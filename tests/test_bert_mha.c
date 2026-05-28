/**
 * MHA fused kernel verification test (manual graph construction).
 *
 * Constructs an OP_MHA_FUSED node directly with known weights,
 * executes CPU + CUDA paths, and compares with a reference
 * unfused computation.
 */
#include "graph.h"
#include "mha_fused_int.h"
#include "matmul_int.h"
#include "add_int.h"
#include "reshape_int.h"
#include "transpose_int.h"
#include "mul_int.h"
#include "softmax_int.h"
#include "layernorm_int.h"
#include "platform.h"
#include "operator.h"
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


#define B  1
#define S  4
#define H  2
#define d  8
#define D  (H * d)  /* 16 */

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

static float mean_abs_diff(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++)
        sum += fabs((double)a[i] - (double)b[i]);
    return (float)(sum / (double)n);
}

static void random_fill(float* data, int64_t n, unsigned seed) {
    unsigned state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = ((float)(state & 0x7FFFFFFF) / 2147483648.0f) - 1.0f;
    }
}

/* Unfused reference: performs the full MHA computation step-by-step */
static void mha_unfused_ref(
    const float* X,        /* (B, S, D) — LayerNorm output */
    const float* R,        /* (B, S, D) — residual */
    const float* WQ, const float* bQ,
    const float* WK, const float* bK,
    const float* WV, const float* bV,
    const float* WO, const float* bO,
    float* Y,              /* (B, S, D) — output */
    float scale, int has_residual)
{
    int64_t BS = B * S;

    /* QKV projections */
    float* Q = (float*)calloc((size_t)BS * D, sizeof(float));
    float* K = (float*)calloc((size_t)BS * D, sizeof(float));
    float* V = (float*)calloc((size_t)BS * D, sizeof(float));

    for (int64_t bs = 0; bs < BS; bs++) {
        for (int64_t j = 0; j < D; j++) {
            float sq = 0, sk = 0, sv = 0;
            for (int64_t i = 0; i < D; i++) {
                float xv = X[bs * D + i];
                sq += xv * WQ[i * D + j];
                sk += xv * WK[i * D + j];
                sv += xv * WV[i * D + j];
            }
            Q[bs * D + j] = sq + (bQ ? bQ[j] : 0);
            K[bs * D + j] = sk + (bK ? bK[j] : 0);
            V[bs * D + j] = sv + (bV ? bV[j] : 0);
        }
    }

    /* Multi-head attention */
    float* attn = (float*)calloc((size_t)B * H * S * d, sizeof(float));
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            float* attn_bh = attn + (b * H + h) * S * d;
            int head_off = (int)(h * d);

            for (int64_t si = 0; si < S; si++) {
                float* scores = (float*)calloc((size_t)S, sizeof(float));
                float max_s = -1e38f;
                for (int64_t sj = 0; sj < S; sj++) {
                    float dot = 0;
                    for (int64_t di = 0; di < d; di++) {
                        dot += Q[(b * S + si) * D + head_off + di]
                             * K[(b * S + sj) * D + head_off + di];
                    }
                    scores[sj] = dot * scale;
                    if (scores[sj] > max_s) max_s = scores[sj];
                }
                float sum = 0;
                for (int64_t sj = 0; sj < S; sj++) {
                    scores[sj] = expf(scores[sj] - max_s);
                    sum += scores[sj];
                }
                if (sum < 1e-12f) sum = 1e-12f;
                for (int64_t di = 0; di < d; di++) {
                    float acc = 0;
                    for (int64_t sj = 0; sj < S; sj++) {
                        acc += scores[sj] * V[(b * S + sj) * D + head_off + di];
                    }
                    attn_bh[si * d + di] = acc / sum;
                }
                free(scores);
            }
        }
    }

    /* Merge heads: (B, H, S, d) → (B, S, D) */
    float* merged = (float*)calloc((size_t)BS * D, sizeof(float));
    for (int64_t b = 0; b < B; b++) {
        for (int64_t si = 0; si < S; si++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t di = 0; di < d; di++) {
                    merged[(b * S + si) * D + h * d + di] =
                        attn[(b * H + h) * S * d + si * d + di];
                }
            }
        }
    }

    /* Output projection + residual */
    for (int64_t bs = 0; bs < BS; bs++) {
        for (int64_t j = 0; j < D; j++) {
            float acc = 0;
            for (int64_t i = 0; i < D; i++) {
                acc += merged[bs * D + i] * WO[i * D + j];
            }
            Y[bs * D + j] = acc + (bO ? bO[j] : 0);
            if (has_residual && R) {
                Y[bs * D + j] += R[bs * D + j];
            }
        }
    }

    free(Q); free(K); free(V); free(attn); free(merged);
}

/* =========================================================================
 * Test: MHA fused (CPU fallback) vs unfused reference
 * ========================================================================= */
static void test_mha_fused_cpu(void) {
    fprintf(stderr, "\n=== MHA Fused CPU Test ===\n");

    int64_t BS = B * S;
    int64_t n_elem = BS * D;

    /* Allocate and fill all tensors */
    float *X = (float*)malloc(n_elem * sizeof(float));
    float *R = (float*)malloc(n_elem * sizeof(float));
    float *WQ = (float*)malloc(D * D * sizeof(float));
    float *bQ = (float*)malloc(D * sizeof(float));
    float *WK = (float*)malloc(D * D * sizeof(float));
    float *bK = (float*)malloc(D * sizeof(float));
    float *WV = (float*)malloc(D * D * sizeof(float));
    float *bV = (float*)malloc(D * sizeof(float));
    float *WO = (float*)malloc(D * D * sizeof(float));
    float *bO = (float*)malloc(D * sizeof(float));
    float *Y_fused = (float*)calloc(n_elem, sizeof(float));
    float *Y_ref   = (float*)calloc(n_elem, sizeof(float));

    random_fill(X,  n_elem, 42);
    random_fill(R,  n_elem, 99);
    random_fill(WQ, D * D, 123);
    random_fill(bQ, D,     456);
    random_fill(WK, D * D, 789);
    random_fill(bK, D,     234);
    random_fill(WV, D * D, 567);
    random_fill(bV, D,     890);
    random_fill(WO, D * D, 345);
    random_fill(bO, D,     678);

    float scale = 1.0f / sqrtf((float)d);

    /* Build graph with a single MHA_Fused node */
    inference_graph_t* g = graph_create();
    CHECK(g != NULL, "graph_create");

    /* Create tensors */
    tensor_t* tX = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* tR = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* tY = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* twQ = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbQ = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twK = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbK = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twV = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbV = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twO = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbO = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    CHECK(tX != NULL, "tX"); CHECK(tR != NULL, "tR"); CHECK(tY != NULL, "tY");
    CHECK(twQ != NULL, "twQ"); CHECK(tbQ != NULL, "tbQ");
    CHECK(twK != NULL, "twK"); CHECK(tbK != NULL, "tbK");
    CHECK(twV != NULL, "twV"); CHECK(tbV != NULL, "tbV");
    CHECK(twO != NULL, "twO"); CHECK(tbO != NULL, "tbO");

    /* Copy data */
    memcpy(tX->data, X, n_elem * sizeof(float));
    memcpy(tR->data, R, n_elem * sizeof(float));
    memcpy(twQ->data, WQ, D * D * sizeof(float));
    memcpy(tbQ->data, bQ, D * sizeof(float));
    memcpy(twK->data, WK, D * D * sizeof(float));
    memcpy(tbK->data, bK, D * sizeof(float));
    memcpy(twV->data, WV, D * D * sizeof(float));
    memcpy(tbV->data, bV, D * sizeof(float));
    memcpy(twO->data, WO, D * D * sizeof(float));
    memcpy(tbO->data, bO, D * sizeof(float));

    /* Add tensors to graph */
    int tidX  = graph_add_tensor(g, tX);
    int tidR  = graph_add_tensor(g, tR);
    int tidY  = graph_add_tensor(g, tY);
    int tidWQ = graph_add_tensor(g, twQ);
    int tidBQ = graph_add_tensor(g, tbQ);
    int tidWK = graph_add_tensor(g, twK);
    int tidBK = graph_add_tensor(g, tbK);
    int tidWV = graph_add_tensor(g, twV);
    int tidBV = graph_add_tensor(g, tbV);
    int tidWO = graph_add_tensor(g, twO);
    int tidBO = graph_add_tensor(g, tbO);
    CHECK(tidX > -1, "tidX");
    CHECK(tidY > -1, "tidY");

    /* Create MHA_Fused node */
    mha_fused_params_t params;
    memset(&params, 0, sizeof(params));
    params.batch_size   = B;
    params.seq_len      = S;
    params.hidden_size  = D;
    params.num_heads    = H;
    params.head_dim     = d;
    params.scale        = scale;
    params.has_residual = true;

    int input_tids[]  = {tidX, tidR};
    int output_tids[] = {tidY};
    tensor_t* weights[] = {twQ, tbQ, twK, tbK, twV, tbV, twO, tbO};

    int nid = graph_add_node(g, OP_MHA_FUSED,
                              2, input_tids,
                              1, output_tids,
                              8, weights,
                              &params, sizeof(params));
    CHECK(nid > -1, "nid");

    /* Add INPUT/OUTPUT wrapper nodes */
    int in_nid = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tidX}, 0, NULL, NULL, 0);
    graph_set_input(g, in_nid);
    int out_nid = graph_add_node(g, OP_OUTPUT, 1, (int[]){tidY}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_output(g, out_nid);

    CHECK(graph_build(g) == 0, "graph_build");

    /* Execute CPU */
    {
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        int rc = graph_execute(g, inputs, outputs, false);
        CHECK(rc == 0, "graph_execute");
        memcpy(Y_fused, tY->data, n_elem * sizeof(float));
    }

    /* Compute ref */
    mha_unfused_ref(X, R, WQ, bQ, WK, bK, WV, bV, WO, bO,
                    Y_ref, scale, 1);

    float max_diff = max_abs_diff(Y_fused, Y_ref, n_elem);
    float mean_diff = mean_abs_diff(Y_fused, Y_ref, n_elem);
    fprintf(stderr, "CPU fused vs ref: max_diff=%.2e  mean_diff=%.2e\n",
            max_diff, mean_diff);

    int non_zero = 0;
    for (int64_t i = 0; i < n_elem; i++)
        if (fabsf(Y_fused[i]) > 1e-6f) non_zero++;
    fprintf(stderr, "Fused non_zero: %d/%lld\n", non_zero, (long long)n_elem);
    CHECK(non_zero > (int)n_elem / 2, "non_zero");

    CHECK(max_diff < 1e-3f, "max_diff cpu");
    fprintf(stderr, "MHA Fused CPU: PASS (max_diff=%.2e)\n", max_diff);

    free(X); free(R); free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);
    free(Y_fused); free(Y_ref);
    graph_destroy(g);
}

/* =========================================================================
 * Test: MHA fused CUDA vs CPU
 * ========================================================================= */
static void test_mha_fused_cuda(void) {
#ifdef USE_CUDA
    fprintf(stderr, "\n=== MHA Fused CUDA Test ===\n");

    int64_t BS = B * S;
    int64_t n_elem = BS * D;

    float *X = (float*)malloc(n_elem * sizeof(float));
    float *R = (float*)malloc(n_elem * sizeof(float));
    float *WQ = (float*)malloc(D * D * sizeof(float));
    float *bQ = (float*)malloc(D * sizeof(float));
    float *WK = (float*)malloc(D * D * sizeof(float));
    float *bK = (float*)malloc(D * sizeof(float));
    float *WV = (float*)malloc(D * D * sizeof(float));
    float *bV = (float*)malloc(D * sizeof(float));
    float *WO = (float*)malloc(D * D * sizeof(float));
    float *bO = (float*)malloc(D * sizeof(float));
    float *Y_cpu = (float*)calloc(n_elem, sizeof(float));
    float *Y_cuda = (float*)calloc(n_elem, sizeof(float));

    random_fill(X,  n_elem, 12345);
    random_fill(R,  n_elem, 67890);
    random_fill(WQ, D * D, 111);
    random_fill(bQ, D,     222);
    random_fill(WK, D * D, 333);
    random_fill(bK, D,     444);
    random_fill(WV, D * D, 555);
    random_fill(bV, D,     666);
    random_fill(WO, D * D, 777);
    random_fill(bO, D,     888);

    float scale = 1.0f / sqrtf((float)d);

    /* Build graph */
    inference_graph_t* g = graph_create();
    CHECK(g != NULL, "graph_create");

    tensor_t* tX = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* tR = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* tY = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, S, D});
    tensor_t* twQ = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbQ = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twK = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbK = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twV = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbV = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    tensor_t* twO = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
    tensor_t* tbO = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
    CHECK(tX != NULL, "tX cuda"); CHECK(tY != NULL, "tY cuda");

    memcpy(tX->data, X, n_elem * sizeof(float));
    memcpy(tR->data, R, n_elem * sizeof(float));
    memcpy(twQ->data, WQ, D * D * sizeof(float));
    memcpy(tbQ->data, bQ, D * sizeof(float));
    memcpy(twK->data, WK, D * D * sizeof(float));
    memcpy(tbK->data, bK, D * sizeof(float));
    memcpy(twV->data, WV, D * D * sizeof(float));
    memcpy(tbV->data, bV, D * sizeof(float));
    memcpy(twO->data, WO, D * D * sizeof(float));
    memcpy(tbO->data, bO, D * sizeof(float));

    int tidX  = graph_add_tensor(g, tX);
    int tidR  = graph_add_tensor(g, tR);
    int tidY  = graph_add_tensor(g, tY);
    graph_add_tensor(g, twQ); graph_add_tensor(g, tbQ);
    graph_add_tensor(g, twK); graph_add_tensor(g, tbK);
    graph_add_tensor(g, twV); graph_add_tensor(g, tbV);
    graph_add_tensor(g, twO); graph_add_tensor(g, tbO);

    mha_fused_params_t params;
    memset(&params, 0, sizeof(params));
    params.batch_size   = B;
    params.seq_len      = S;
    params.hidden_size  = D;
    params.num_heads    = H;
    params.head_dim     = d;
    params.scale        = scale;
    params.has_residual = true;

    int input_tids[]  = {tidX, tidR};
    int output_tids[] = {tidY};
    tensor_t* weights[] = {twQ, tbQ, twK, tbK, twV, tbV, twO, tbO};

    int nid = graph_add_node(g, OP_MHA_FUSED,
                              2, input_tids,
                              1, output_tids,
                              8, weights,
                              &params, sizeof(params));
    CHECK(nid > -1, "nid");

    int in_id = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tidX}, 0, NULL, NULL, 0);
    graph_set_input(g, in_id);
    int out_id = graph_add_node(g, OP_OUTPUT, 1, (int[]){tidY}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_output(g, out_id);

    CHECK(graph_build(g) == 0, "graph_build");

    /* CPU execution */
    {
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        int rc = graph_execute(g, inputs, outputs, false);
        CHECK(rc == 0, "graph_execute");
        memcpy(Y_cpu, tY->data, n_elem * sizeof(float));
    }

    /* CUDA execution */
    {
        memset(tY->data, 0, n_elem * sizeof(float));
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        fprintf(stderr, "Running CUDA fused kernel...\n");
        int rc = graph_execute(g, inputs, outputs, true);
        CHECK(rc == 0, "graph_execute");
        memcpy(Y_cuda, tY->data, n_elem * sizeof(float));
    }

    float max_diff = max_abs_diff(Y_cuda, Y_cpu, n_elem);
    float mean_diff = mean_abs_diff(Y_cuda, Y_cpu, n_elem);
    fprintf(stderr, "CUDA vs CPU: max_diff=%.2e  mean_diff=%.2e\n",
            max_diff, mean_diff);

    int non_zero = 0;
    for (int64_t i = 0; i < n_elem; i++)
        if (fabsf(Y_cuda[i]) > 1e-6f) non_zero++;
    fprintf(stderr, "CUDA non_zero: %d/%lld\n", non_zero, (long long)n_elem);
    CHECK(non_zero > (int)n_elem / 2, "non_zero");

    CHECK(max_diff < 1e-3f, "max_diff cpu");
    fprintf(stderr, "MHA Fused CUDA: PASS (max_diff=%.2e)\n", max_diff);

    /* Second run to verify non-destructive fusion */
    {
        memset(tY->data, 0, n_elem * sizeof(float));
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        int rc = graph_execute(g, inputs, outputs, true);
        CHECK(rc == 0, "graph_execute");
        float max_diff2 = max_abs_diff((float*)tY->data, Y_cpu, n_elem);
        fprintf(stderr, "CUDA run 2 vs CPU: max_diff=%.2e\n", max_diff2);
        CHECK(max_diff2 < 1e-3f, "max_diff cuda run2");
        fprintf(stderr, "MHA Fused CUDA run 2: PASS\n");
    }

    free(X); free(R); free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);
    free(Y_cpu); free(Y_cuda);
    graph_destroy(g);
#else
    (void)0;
#endif
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_mha_fused_cpu();
    fprintf(stderr, "\n=== MHA CPU DONE ===\n");

#ifdef USE_CUDA
    test_mha_fused_cuda();
    fprintf(stderr, "\n=== MHA CUDA DONE ===\n");
    cuda_platform_finalize();
#endif

    platform_finalize();
    return 0;
}
