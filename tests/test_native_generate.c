/**
 * Native generate test (R10) - mha_decode + KV-cache autoregressive generation.
 *
 * Builds a native single-layer transformer graph and runs a prefill->decode
 * generation loop, exercising the full decode path:
 *   - mha_decode with RoPE fusion (pos=cache_len, cache stores rotated K)
 *   - graph_update_cache_len to advance position each step
 *   - graph_set_kv_cache to persist K/V across decode steps
 *   - graph_set_permanent_fusion to avoid re-fusing
 *
 * This is NOT an ONNX-driven test (the GPT-2 ONNX model has no KV-cache ports
 * and uses absolute position embeddings, not RoPE). It builds the graph in C
 * to validate the decode pipeline that R9-h made possible.
 *
 * Graph: token_embed(Gather) -> mha_decode[RoPE] -> residual(Add)
 *        -> LayerNorm -> FFN(MatMul+GELU+MatMul) -> residual(Add)
 *        -> lm_head(MatMul) -> logits
 */
#include "graph.h"
#include "platform.h"
#include "operator.h"
#include "generate.h"
#include "mha_decode_int.h"
#include "matmul_int.h"
#include "layernorm_int.h"
#include "add_int.h"
#include "gather_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

/* Model dimensions (small for fast test) */
#define VOCAB    16
#define D_MODEL  8    /* hidden_size */
#define N_HEADS  2
#define D_HEAD   4    /* D_MODEL / N_HEADS */
#define D_FF     16   /* FFN inner size */
#define MAX_SEQ  8
#define N_LAYERS 1

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

static void random_fill(float* data, int64_t n, int seed) {
    srand(seed);
    for (int64_t i = 0; i < n; i++) {
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

/* Reference decode (single token): mirrors the graph for verification.
   Computes one forward pass at position cache_len, returns logits[vocab]. */
static void decode_ref(
    int64_t token_id, int64_t cache_len,
    const float* tok_emb,      /* [vocab, D_MODEL] */
    const float* WQ, const float* bQ, const float* WK, const float* bK,
    const float* WV, const float* bV, const float* WO, const float* bO,
    float* K_cache, float* V_cache,  /* [max_seq, N_HEADS, D_HEAD] */
    const float* ln1_g, const float* ln1_b,
    const float* ff1_w, const float* ff1_b,
    const float* ff2_w, const float* ff2_b,
    const float* ln2_g, const float* ln2_b,
    const float* lm_head_w,       /* [D_MODEL, vocab] */
    float* logits,               /* [vocab] output */
    int64_t D, int64_t H, int64_t d)
{
    float x[64];
    memcpy(x, tok_emb + token_id * D, D * sizeof(float));

    /* mha_decode with RoPE (pos=cache_len) */
    float Y[64];
    for (int64_t j = 0; j < D; j++) Y[j] = bO ? bO[j] : 0.0f;
    int64_t kv_dim = H * d;
    for (int64_t kv_h = 0; kv_h < H; kv_h++) {
        int64_t kv_ho = kv_h * d;
        float K_new[64], V_new[64];
        for (int64_t di = 0; di < d; di++) {
            float ka = 0, va = 0;
            for (int64_t j = 0; j < D; j++) { ka += x[j]*WK[j*kv_dim+kv_ho+di]; va += x[j]*WV[j*kv_dim+kv_ho+di]; }
            K_new[di] = ka + (bK ? bK[kv_ho+di] : 0);
            V_new[di] = va + (bV ? bV[kv_ho+di] : 0);
        }
        /* rotate K_new (HALFSPLIT, pos=cache_len) */
        int64_t half_d = d/2;
        for (int64_t i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2*i)/(float)d);
            float ang = (float)cache_len * freq, c = cosf(ang), s = sinf(ang);
            float x0 = K_new[i], x1 = K_new[i+half_d];
            K_new[i] = x0*c - x1*s; K_new[i+half_d] = x0*s + x1*c;
        }
        int64_t cidx = cache_len * H * d + kv_ho;
        for (int64_t di = 0; di < d; di++) { K_cache[cidx+di] = K_new[di]; V_cache[cidx+di] = V_new[di]; }
    }
    float scale = 1.0f / sqrtf((float)d);
    for (int64_t h = 0; h < H; h++) {
        int64_t ho = h*d;
        float Q[64];
        for (int64_t di = 0; di < d; di++) {
            float a = 0; for (int64_t j = 0; j < D; j++) a += x[j]*WQ[j*D+ho+di];
            Q[di] = a + (bQ ? bQ[ho+di] : 0);
        }
        /* rotate Q (HALFSPLIT, pos=cache_len) */
        int64_t half_d = d/2;
        for (int64_t i = 0; i < half_d; i++) {
            float freq = 1.0f / powf(10000.0f, (float)(2*i)/(float)d);
            float ang = (float)cache_len * freq, c = cosf(ang), s = sinf(ang);
            float x0 = Q[i], x1 = Q[i+half_d];
            Q[i] = x0*c - x1*s; Q[i+half_d] = x0*s + x1*c;
        }
        int64_t total = cache_len + 1;
        float scores[16], mx = -1e30f;
        for (int64_t t = 0; t < total; t++) {
            float dot = 0; int64_t ko = t*H*d+ho;
            for (int64_t di = 0; di < d; di++) dot += Q[di]*K_cache[ko+di];
            scores[t] = dot*scale; if (scores[t]>mx) mx=scores[t];
        }
        float se = 0; for (int64_t t = 0; t < total; t++) { scores[t]=expf(scores[t]-mx); se+=scores[t]; }
        if (se < 1e-12f) se = 1e-12f;
        float merged[64] = {0};
        for (int64_t t = 0; t < total; t++) {
            float w = scores[t]/se; int64_t vo = t*H*d+ho;
            for (int64_t di = 0; di < d; di++) merged[di] += w*V_cache[vo+di];
        }
        for (int64_t j = 0; j < D; j++) {
            float c = 0; for (int64_t di = 0; di < d; di++) c += merged[di]*WO[(ho+di)*D+j];
            Y[j] += c;
        }
    }
    /* residual + layernorm1 */
    float h1[64]; for (int64_t j = 0; j < D; j++) h1[j] = x[j] + Y[j];
    float ln1[64];
    {
        float mean = 0; for (int64_t j = 0; j < D; j++) mean += h1[j]; mean /= D;
        float var = 0; for (int64_t j = 0; j < D; j++) { float dt = h1[j]-mean; var += dt*dt; } var /= D;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int64_t j = 0; j < D; j++) ln1[j] = (h1[j]-mean)*inv*ln1_g[j] + ln1_b[j];
    }
    /* FFN: ff1 -> gelu(tanh) -> ff2 */
    float ff1[64];
    for (int64_t j = 0; j < D_FF; j++) {
        float a = ff1_b ? ff1_b[j] : 0;
        for (int64_t i = 0; i < D; i++) a += ln1[i]*ff1_w[i*D_FF+j];
        float t = 0.7978845608f * (a + 0.044715f*a*a*a);  /* tanh GELU */
        ff1[j] = 0.5f*a*(1.0f + tanhf(t));
    }
    float ff2[64];
    for (int64_t j = 0; j < D; j++) {
        float a = ff2_b ? ff2_b[j] : 0;
        for (int64_t i = 0; i < D_FF; i++) a += ff1[i]*ff2_w[i*D+j];
        ff2[j] = a;
    }
    /* residual + layernorm2 */
    float h2[64]; for (int64_t j = 0; j < D; j++) h2[j] = ln1[j] + ff2[j];  /* note: residual to ln1 output */
    float ln2[64];
    {
        float mean = 0; for (int64_t j = 0; j < D; j++) mean += h2[j]; mean /= D;
        float var = 0; for (int64_t j = 0; j < D; j++) { float dt = h2[j]-mean; var += dt*dt; } var /= D;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int64_t j = 0; j < D; j++) ln2[j] = (h2[j]-mean)*inv*ln2_g[j] + ln2_b[j];
    }
    /* lm_head */
    for (int64_t v = 0; v < VOCAB; v++) {
        float a = 0; for (int64_t j = 0; j < D; j++) a += ln2[j]*lm_head_w[j*VOCAB+v];
        logits[v] = a;
    }
}

/* Minimal generate_tokens API test: builds a decode-only graph
   (token_embed -> mha_decode[RoPE] -> lm_head) and calls generate_tokens. */
static int test_generate_api(void) {
    fprintf(stderr, "\n--- generate_tokens API test ---\n");

    const int64_t D = D_MODEL, H = N_HEADS, d = D_HEAD;
    float* tok_emb = (float*)malloc(VOCAB * D * sizeof(float));
    float* WQ = (float*)malloc(D*D*sizeof(float)), *bQ = (float*)malloc(D*sizeof(float));
    float* WK = (float*)malloc(D*D*sizeof(float)), *bK = (float*)malloc(D*sizeof(float));
    float* WV = (float*)malloc(D*D*sizeof(float)), *bV = (float*)malloc(D*sizeof(float));
    float* WO = (float*)malloc(D*D*sizeof(float)), *bO = (float*)malloc(D*sizeof(float));
    float* lm_w = (float*)malloc(D*VOCAB*sizeof(float));
    random_fill(tok_emb, VOCAB*D, 31);
    random_fill(WQ, D*D, 32); random_fill(bQ, D, 33);
    random_fill(WK, D*D, 34); random_fill(bK, D, 35);
    random_fill(WV, D*D, 36); random_fill(bV, D, 37);
    random_fill(WO, D*D, 38); random_fill(bO, D, 39);
    random_fill(lm_w, D*VOCAB, 40);

    inference_graph_t* g = graph_create();
    int64_t id_shape[] = {1};
    int64_t x_shape[] = {1, D};
    int64_t cache_shape[] = {1, MAX_SEQ, H, d};
    int64_t wdd[] = {D, D};
    int64_t wdv[] = {D, VOCAB};
    int64_t bd[] = {D};
    int64_t logit_shape[] = {1, VOCAB};

    tensor_t* tId = tensor_create(DATA_TYPE_I64, 1, id_shape);
    tensor_t* tEmb = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){VOCAB, D});
    tensor_t* tX = tensor_create(DATA_TYPE_F32, 2, x_shape);
    tensor_t* tKc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tY = tensor_create(DATA_TYPE_F32, 2, x_shape);
    tensor_t* tKo = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVo = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tLogits = tensor_create(DATA_TYPE_F32, 2, logit_shape);
    memcpy(tEmb->data, tok_emb, VOCAB*D*sizeof(float));
    tensor_t* tWQ=tensor_create(DATA_TYPE_F32,2,wdd); memcpy(tWQ->data,WQ,D*D*sizeof(float));
    tensor_t* tbQ=tensor_create(DATA_TYPE_F32,1,bd); memcpy(tbQ->data,bQ,D*sizeof(float));
    tensor_t* tWK=tensor_create(DATA_TYPE_F32,2,wdd); memcpy(tWK->data,WK,D*D*sizeof(float));
    tensor_t* tbK=tensor_create(DATA_TYPE_F32,1,bd); memcpy(tbK->data,bK,D*sizeof(float));
    tensor_t* tWV=tensor_create(DATA_TYPE_F32,2,wdd); memcpy(tWV->data,WV,D*D*sizeof(float));
    tensor_t* tbV=tensor_create(DATA_TYPE_F32,1,bd); memcpy(tbV->data,bV,D*sizeof(float));
    tensor_t* tWO=tensor_create(DATA_TYPE_F32,2,wdd); memcpy(tWO->data,WO,D*D*sizeof(float));
    tensor_t* tbO=tensor_create(DATA_TYPE_F32,1,bd); memcpy(tbO->data,bO,D*sizeof(float));
    tensor_t* tLM=tensor_create(DATA_TYPE_F32,2,wdv); memcpy(tLM->data,lm_w,D*VOCAB*sizeof(float));

    int tidId=graph_add_tensor(g,tId), tidEmb=graph_add_tensor(g,tEmb);
    int tidX=graph_add_tensor(g,tX), tidKc=graph_add_tensor(g,tKc), tidVc=graph_add_tensor(g,tVc);
    int tidY=graph_add_tensor(g,tY), tidKo=graph_add_tensor(g,tKo), tidVo=graph_add_tensor(g,tVo);
    int tidLogits=graph_add_tensor(g,tLogits);
    int tidWQ=graph_add_tensor(g,tWQ),tidbQ=graph_add_tensor(g,tbQ);
    int tidWK=graph_add_tensor(g,tWK),tidbK=graph_add_tensor(g,tbK);
    int tidWV=graph_add_tensor(g,tWV),tidbV=graph_add_tensor(g,tbV);
    int tidWO=graph_add_tensor(g,tWO),tidbO=graph_add_tensor(g,tbO);
    int tidLM=graph_add_tensor(g,tLM);

    /* Gather(embed, id) -> X */
    gather_params_t gp; memset(&gp,0,sizeof(gp));
    gp.axis=0; gp.num_indices=1; gp.block_size=D; gp.outer_size=1;
    gp.inner_size=VOCAB*D; gp.out_axis_dim=1;
    { int in[]={tidEmb,tidId},out[]={tidX};
      graph_add_node(g,OP_GATHER,2,in,1,out,0,NULL,&gp,sizeof(gp)); }

    /* mha_decode(X, K_cache, V_cache, weights) -> Y, K_out, V_out */
    mha_decode_params_t mp; memset(&mp,0,sizeof(mp));
    mp.batch_size=1; mp.hidden_size=D; mp.num_heads=H; mp.num_kv_heads=H; mp.head_dim=d;
    mp.scale=1.0f/sqrtf((float)d); mp.cache_len=0; mp.max_seq=MAX_SEQ;
    mp.rope_base=10000.0f; mp.rope_layout=ROPE_LAYOUT_HALFSPLIT; mp.rope_inv_freq=NULL;
    { int in[]={tidX,tidKc,tidVc}, out[]={tidY,tidKo,tidVo};
      tensor_t* wts[]={tWQ,tbQ,tWK,tbK,tWV,tbV,tWO,tbO};
      graph_add_node(g,OP_MHA_DECODE,3,in,3,out,8,wts,&mp,sizeof(mp)); }
    graph_set_kv_cache(g, tidKo, tidVo);

    /* lm_head: MatMul(Y, lm_w) -> logits */
    matmul_params_t mmp; memset(&mmp,0,sizeof(mmp));
    mmp.M=1; mmp.N=VOCAB; mmp.K=D;
    { int in[]={tidY}, out[]={tidLogits};
      tensor_t* wts[]={tLM};
      graph_add_node(g,OP_MATMUL,1,in,1,out,1,wts,&mmp,sizeof(mmp)); }

    int in_nid=graph_add_node(g,OP_INPUT,0,NULL,1,(int[]){tidId},0,NULL,NULL,0);
    graph_set_input(g,in_nid);
    int out_nid=graph_add_node(g,OP_OUTPUT,1,(int[]){tidLogits},0,NULL,0,NULL,NULL,0);
    graph_set_output(g,out_nid);
    graph_set_permanent_fusion(g,1);
    CHECK(graph_build(g)==0, "graph_build (api)");

    /* Call generate_tokens */
    int64_t prompt[] = {5, 2};
    int64_t out[8] = {0};
    generate_config_t cfg = generate_default_config();
    cfg.max_new_tokens = 3;
    cfg.temperature = 0;
    cfg.use_cuda = 0;
    cfg.verbose = 0;

    int gen = generate_tokens(g, prompt, 2, out, &cfg);
    fprintf(stderr, "generate_tokens: gen=%d, tokens:", gen);
    for (int i = 0; i < gen; i++) fprintf(stderr, " %lld", (long long)out[i]);
    fprintf(stderr, "\n");
    CHECK(gen == 3, "generate_tokens wrong count");

    fprintf(stderr, "generate_tokens API (KV-cache): PASS\n");
    graph_destroy(g);
    free(tok_emb);free(WQ);free(bQ);free(WK);free(bK);free(WV);free(bV);free(WO);free(bO);free(lm_w);
    return 0;
}

static int run_tests(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    fprintf(stderr, "\n=== R10: Native Generate (mha_decode + KV-cache) ===\n");

    /* Allocate all model weights */
    float* tok_emb = (float*)malloc(VOCAB * D_MODEL * sizeof(float));
    float* WQ = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    float* bQ = (float*)malloc(D_MODEL * sizeof(float));
    float* WK = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    float* bK = (float*)malloc(D_MODEL * sizeof(float));
    float* WV = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    float* bV = (float*)malloc(D_MODEL * sizeof(float));
    float* WO = (float*)malloc(D_MODEL * D_MODEL * sizeof(float));
    float* bO = (float*)malloc(D_MODEL * sizeof(float));
    float* ln1_g = (float*)malloc(D_MODEL * sizeof(float));
    float* ln1_b = (float*)malloc(D_MODEL * sizeof(float));
    float* ff1_w = (float*)malloc(D_MODEL * D_FF * sizeof(float));
    float* ff1_b = (float*)malloc(D_FF * sizeof(float));
    float* ff2_w = (float*)malloc(D_FF * D_MODEL * sizeof(float));
    float* ff2_b = (float*)malloc(D_MODEL * sizeof(float));
    float* ln2_g = (float*)malloc(D_MODEL * sizeof(float));
    float* ln2_b = (float*)malloc(D_MODEL * sizeof(float));
    float* lm_head_w = (float*)malloc(D_MODEL * VOCAB * sizeof(float));

    random_fill(tok_emb, VOCAB*D_MODEL, 1);
    random_fill(WQ, D_MODEL*D_MODEL, 2); random_fill(bQ, D_MODEL, 3);
    random_fill(WK, D_MODEL*D_MODEL, 4); random_fill(bK, D_MODEL, 5);
    random_fill(WV, D_MODEL*D_MODEL, 6); random_fill(bV, D_MODEL, 7);
    random_fill(WO, D_MODEL*D_MODEL, 8); random_fill(bO, D_MODEL, 9);
    random_fill(ln1_g, D_MODEL, 10); random_fill(ln1_b, D_MODEL, 11);
    random_fill(ff1_w, D_MODEL*D_FF, 12); random_fill(ff1_b, D_FF, 13);
    random_fill(ff2_w, D_FF*D_MODEL, 14); random_fill(ff2_b, D_MODEL, 15);
    random_fill(ln2_g, D_MODEL, 16); random_fill(ln2_b, D_MODEL, 17);
    random_fill(lm_head_w, D_MODEL*VOCAB, 18);

    /* Build native graph */
    inference_graph_t* g = graph_create();
    CHECK(g, "graph_create");

    /* Tensors */
    int64_t id_shape[] = {1};  /* single token id */
    int64_t d_shape[] = {1, D_MODEL};
    int64_t cache_shape[] = {1, MAX_SEQ, N_HEADS, D_HEAD};
    int64_t w_dd[] = {D_MODEL, D_MODEL};
    int64_t w_df[] = {D_MODEL, D_FF};
    int64_t w_fd[] = {D_FF, D_MODEL};
    int64_t w_dv[] = {D_MODEL, VOCAB};
    int64_t b_d[] = {D_MODEL};
    int64_t b_ff[] = {D_FF};
    int64_t b_v[] = {VOCAB};
    int64_t logit_shape[] = {1, VOCAB};

    tensor_t* tId = tensor_create(DATA_TYPE_I64, 1, id_shape);
    tensor_t* tEmb = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){VOCAB, D_MODEL});
    tensor_t* tX = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tKc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tY = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tKo = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVo = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tRes1 = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tLn1 = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tFf1 = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){1, D_FF});
    tensor_t* tFf2 = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tRes2 = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tLn2 = tensor_create(DATA_TYPE_F32, 2, d_shape);
    tensor_t* tLogits = tensor_create(DATA_TYPE_F32, 2, logit_shape);

    memcpy(tEmb->data, tok_emb, VOCAB*D_MODEL*sizeof(float));
    /* weight tensors */
    tensor_t* tWQ = tensor_create(DATA_TYPE_F32, 2, w_dd); memcpy(tWQ->data, WQ, D_MODEL*D_MODEL*sizeof(float));
    tensor_t* tbQ = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tbQ->data, bQ, D_MODEL*sizeof(float));
    tensor_t* tWK = tensor_create(DATA_TYPE_F32, 2, w_dd); memcpy(tWK->data, WK, D_MODEL*D_MODEL*sizeof(float));
    tensor_t* tbK = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tbK->data, bK, D_MODEL*sizeof(float));
    tensor_t* tWV = tensor_create(DATA_TYPE_F32, 2, w_dd); memcpy(tWV->data, WV, D_MODEL*D_MODEL*sizeof(float));
    tensor_t* tbV = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tbV->data, bV, D_MODEL*sizeof(float));
    tensor_t* tWO = tensor_create(DATA_TYPE_F32, 2, w_dd); memcpy(tWO->data, WO, D_MODEL*D_MODEL*sizeof(float));
    tensor_t* tbO = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tbO->data, bO, D_MODEL*sizeof(float));
    tensor_t* tLn1g = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tLn1g->data, ln1_g, D_MODEL*sizeof(float));
    tensor_t* tLn1b = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tLn1b->data, ln1_b, D_MODEL*sizeof(float));
    tensor_t* tFf1w = tensor_create(DATA_TYPE_F32, 2, w_df); memcpy(tFf1w->data, ff1_w, D_MODEL*D_FF*sizeof(float));
    tensor_t* tFf1b = tensor_create(DATA_TYPE_F32, 1, b_ff); memcpy(tFf1b->data, ff1_b, D_FF*sizeof(float));
    tensor_t* tFf2w = tensor_create(DATA_TYPE_F32, 2, w_fd); memcpy(tFf2w->data, ff2_w, D_FF*D_MODEL*sizeof(float));
    tensor_t* tFf2b = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tFf2b->data, ff2_b, D_MODEL*sizeof(float));
    tensor_t* tLn2g = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tLn2g->data, ln2_g, D_MODEL*sizeof(float));
    tensor_t* tLn2b = tensor_create(DATA_TYPE_F32, 1, b_d); memcpy(tLn2b->data, ln2_b, D_MODEL*sizeof(float));
    tensor_t* tLM = tensor_create(DATA_TYPE_F32, 2, w_dv); memcpy(tLM->data, lm_head_w, D_MODEL*VOCAB*sizeof(float));

    /* Add tensors to graph */
    int tidId = graph_add_tensor(g, tId);
    int tidEmb = graph_add_tensor(g, tEmb);
    int tidX = graph_add_tensor(g, tX);
    int tidKc = graph_add_tensor(g, tKc);
    int tidVc = graph_add_tensor(g, tVc);
    int tidY = graph_add_tensor(g, tY);
    int tidKo = graph_add_tensor(g, tKo);
    int tidVo = graph_add_tensor(g, tVo);
    int tidRes1 = graph_add_tensor(g, tRes1);
    int tidLn1 = graph_add_tensor(g, tLn1);
    int tidFf1 = graph_add_tensor(g, tFf1);
    int tidFf2 = graph_add_tensor(g, tFf2);
    int tidRes2 = graph_add_tensor(g, tRes2);
    int tidLn2 = graph_add_tensor(g, tLn2);
    int tidLogits = graph_add_tensor(g, tLogits);
    int tidWQ = graph_add_tensor(g, tWQ), tidbQ = graph_add_tensor(g, tbQ);
    int tidWK = graph_add_tensor(g, tWK), tidbK = graph_add_tensor(g, tbK);
    int tidWV = graph_add_tensor(g, tWV), tidbV = graph_add_tensor(g, tbV);
    int tidWO = graph_add_tensor(g, tWO), tidbO = graph_add_tensor(g, tbO);
    int tidLn1g = graph_add_tensor(g, tLn1g), tidLn1b = graph_add_tensor(g, tLn1b);
    int tidFf1w = graph_add_tensor(g, tFf1w), tidFf1b = graph_add_tensor(g, tFf1b);
    int tidFf2w = graph_add_tensor(g, tFf2w), tidFf2b = graph_add_tensor(g, tFf2b);
    int tidLn2g = graph_add_tensor(g, tLn2g), tidLn2b = graph_add_tensor(g, tLn2b);
    int tidLM = graph_add_tensor(g, tLM);

    /* Node 1: Gather(tok_emb, token_id) -> X */
    gather_params_t gp; memset(&gp, 0, sizeof(gp));
    gp.axis = 0; gp.num_indices = 1; gp.block_size = D_MODEL;
    gp.outer_size = 1; gp.inner_size = VOCAB * D_MODEL; gp.out_axis_dim = 1;
    {
        int in[] = {tidEmb, tidId}, out[] = {tidX};
        graph_add_node(g, OP_GATHER, 2, in, 1, out, 0, NULL, &gp, sizeof(gp));
    }

    /* Node 2: mha_decode(X, K_cache, V_cache, WQ,bQ,...,WO,bO) -> Y, K_out, V_out */
    mha_decode_params_t mp; memset(&mp, 0, sizeof(mp));
    mp.batch_size = 1; mp.hidden_size = D_MODEL;
    mp.num_heads = N_HEADS; mp.num_kv_heads = N_HEADS; mp.head_dim = D_HEAD;
    mp.scale = 1.0f / sqrtf((float)D_HEAD);
    mp.cache_len = 0; mp.max_seq = MAX_SEQ;
    mp.rope_base = 10000.0f;
    mp.rope_layout = ROPE_LAYOUT_HALFSPLIT;
    mp.rope_inv_freq = NULL;
    {
        int in[] = {tidX, tidKc, tidVc};
        int out[] = {tidY, tidKo, tidVo};
        tensor_t* wts[] = {tWQ, tbQ, tWK, tbK, tWV, tbV, tWO, tbO};
        graph_add_node(g, OP_MHA_DECODE, 3, in, 3, out, 8, wts, &mp, sizeof(mp));
    }
    graph_set_kv_cache(g, tidKo, tidVo);

    /* Node 3: Add(X, Y) -> Res1 (residual) */
    add_params_t ap1; ap1.numel = D_MODEL; ap1.B_numel = D_MODEL;
    {
        int in[] = {tidX, tidY}, out[] = {tidRes1};
        graph_add_node(g, OP_ADD, 2, in, 1, out, 0, NULL, &ap1, sizeof(ap1));
    }

    /* Node 4: LayerNorm(Res1) -> Ln1 */
    layernorm_params_t lp1; lp1.N = 1; lp1.normalized_size = D_MODEL; lp1.epsilon = 1e-5f;
    {
        int in[] = {tidRes1}, out[] = {tidLn1};
        tensor_t* wts[] = {tLn1g, tLn1b};
        graph_add_node(g, OP_LAYERNORM, 1, in, 1, out, 2, wts, &lp1, sizeof(lp1));
    }

    /* Node 5: MatMul(Ln1, ff1_w) + ff1_b -> Ff1 */
    matmul_params_t mmp1; memset(&mmp1, 0, sizeof(mmp1));
    mmp1.M = 1; mmp1.N = D_FF; mmp1.K = D_MODEL;
    {
        int in[] = {tidLn1}; int out[] = {tidFf1};
        tensor_t* wts[] = {tFf1w, tFf1b};
        graph_add_node(g, OP_MATMUL, 1, in, 1, out, 2, wts, &mmp1, sizeof(mmp1));
    }

    /* Node 6: GELU(Ff1) in-place */
    {
        int in[] = {tidFf1}; int out[] = {tidFf1};
        graph_add_node(g, OP_GELU, 1, in, 1, out, 0, NULL, NULL, 0);
    }

    /* Node 7: MatMul(Ff1, ff2_w) + ff2_b -> Ff2 */
    matmul_params_t mmp2; memset(&mmp2, 0, sizeof(mmp2));
    mmp2.M = 1; mmp2.N = D_MODEL; mmp2.K = D_FF;
    {
        int in[] = {tidFf1}; int out[] = {tidFf2};
        tensor_t* wts[] = {tFf2w, tFf2b};
        graph_add_node(g, OP_MATMUL, 1, in, 1, out, 2, wts, &mmp2, sizeof(mmp2));
    }

    /* Node 8: Add(Ln1, Ff2) -> Res2 (residual to ln1) */
    add_params_t ap2; ap2.numel = D_MODEL; ap2.B_numel = D_MODEL;
    {
        int in[] = {tidLn1, tidFf2}, out[] = {tidRes2};
        graph_add_node(g, OP_ADD, 2, in, 1, out, 0, NULL, &ap2, sizeof(ap2));
    }

    /* Node 9: LayerNorm(Res2) -> Ln2 */
    layernorm_params_t lp2; lp2.N = 1; lp2.normalized_size = D_MODEL; lp2.epsilon = 1e-5f;
    {
        int in[] = {tidRes2}, out[] = {tidLn2};
        tensor_t* wts[] = {tLn2g, tLn2b};
        graph_add_node(g, OP_LAYERNORM, 1, in, 1, out, 2, wts, &lp2, sizeof(lp2));
    }

    /* Node 10: MatMul(Ln2, lm_head_w) -> Logits */
    matmul_params_t mmp3; memset(&mmp3, 0, sizeof(mmp3));
    mmp3.M = 1; mmp3.N = VOCAB; mmp3.K = D_MODEL;
    {
        int in[] = {tidLn2}; int out[] = {tidLogits};
        tensor_t* wts[] = {tLM};
        graph_add_node(g, OP_MATMUL, 1, in, 1, out, 1, wts, &mmp3, sizeof(mmp3));
    }

    /* I/O wrappers */
    int in_nid = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tidId}, 0, NULL, NULL, 0);
    graph_set_input(g, in_nid);
    int out_nid = graph_add_node(g, OP_OUTPUT, 1, (int[]){tidLogits}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_output(g, out_nid);

    graph_set_permanent_fusion(g, 1);
    CHECK(graph_build(g) == 0, "graph_build");

    /* === Generation loop: prefill 2 tokens, decode 2 tokens === */
    int64_t prompt[] = {3, 7};
    int n_prompt = 2;
    int n_gen = 2;
    int64_t generated[8];
    int n_generated = 0;

    /* Reference caches (independent copy for verification) */
    float* ref_K = (float*)calloc(MAX_SEQ * N_HEADS * D_HEAD, sizeof(float));
    float* ref_V = (float*)calloc(MAX_SEQ * N_HEADS * D_HEAD, sizeof(float));

    for (int step = 0; step < n_prompt + n_gen; step++) {
        int64_t token_id = (step < n_prompt) ? prompt[step] : generated[n_generated - 1];

        /* Set input token id */
        ((int64_t*)tId->data)[0] = token_id;

        /* Update cache_len for mha_decode nodes */
        graph_update_cache_len(g, step);

        /* Execute */
        tensor_t* inputs[] = {tId};
        tensor_t* outputs[] = {tLogits};
        int rc = graph_execute(g, inputs, outputs, false);
        CHECK(rc == 0, "graph_execute failed");

        /* Copy K/V cache back for next step (graph_execute may have swapped pointers) */
        memcpy(tKc->data, tKo->data, MAX_SEQ*N_HEADS*D_HEAD*sizeof(float));
        memcpy(tVc->data, tVo->data, MAX_SEQ*N_HEADS*D_HEAD*sizeof(float));

        /* Reference */
        float ref_logits[16];
        decode_ref(token_id, step, tok_emb, WQ, bQ, WK, bK, WV, bV, WO, bO,
                   ref_K, ref_V, ln1_g, ln1_b, ff1_w, ff1_b, ff2_w, ff2_b,
                   ln2_g, ln2_b, lm_head_w, ref_logits, D_MODEL, N_HEADS, D_HEAD);

        float diff = max_abs_diff((float*)tLogits->data, ref_logits, VOCAB);
        fprintf(stderr, "Step %d (token=%lld, cache_len=%d): logits max_diff=%.2e\n",
                step, (long long)token_id, step, diff);
        CHECK(diff < 1e-3f, "logits mismatch");

        /* Greedy: argmax */
        int best = 0; float bestv = ((float*)tLogits->data)[0];
        for (int v = 1; v < VOCAB; v++) {
            if (((float*)tLogits->data)[v] > bestv) { bestv = ((float*)tLogits->data)[v]; best = v; }
        }
        if (step >= n_prompt) {
            generated[n_generated++] = best;
        }
        fprintf(stderr, "  -> next token: %d\n", best);
    }

    fprintf(stderr, "Generated tokens:");
    for (int i = 0; i < n_generated; i++) fprintf(stderr, " %lld", (long long)generated[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "R10 Native Generate (manual loop): PASS\n");

    graph_destroy(g);
    free(ref_K); free(ref_V);
    free(tok_emb); free(WQ); free(bQ); free(WK); free(bK); free(WV); free(bV); free(WO); free(bO);
    free(ln1_g); free(ln1_b); free(ff1_w); free(ff1_b); free(ff2_w); free(ff2_b);
    free(ln2_g); free(ln2_b); free(lm_head_w);

    /* Test generate_tokens API with a minimal decode-only graph */
    test_generate_api();

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();
    fprintf(stderr, "\n=== R10 Native Generate Tests Done ===\n");
    return 0;
}

int main(void) {
    return run_tests();
}
