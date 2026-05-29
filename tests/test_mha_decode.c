/**
 * MHA decode kernel verification test.
 *
 * Tests the single-token decode kernel with KV-cache:
 *   - CPU reference correctness
 *   - CUDA vs CPU comparison
 *   - KV-cache persistence across multiple decode steps
 */
#include "graph.h"
#include "mha_decode_int.h"
#include "causal_mask_int.h"
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
int mha_decode_f32(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream);
int causal_mask_f32(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

/* Small test dimensions */
#define TB  1
#define TH  2
#define Td  4
#define TD  (TH * Td)  /* 8 */
#define TMAX_SEQ 8

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

/* ============================================================
 * CPU reference: manual mha_decode computation
 * ============================================================ */
static void decode_ref(
    const float* X_new,    /* (B, 1, D) */
    const float* K_cache,  /* (B, max_seq, H, d) */
    const float* V_cache,  /* (B, max_seq, H, d) */
    const float* WQ, const float* bQ,
    const float* WK, const float* bK,
    const float* WV, const float* bV,
    const float* WO, const float* bO,
    float* Y,              /* (B, 1, D) */
    float* K_out, float* V_out,
    int64_t cache_len, float scale)
{
    /* Copy caches */
    memcpy(K_out, K_cache, TB * TMAX_SEQ * TH * Td * sizeof(float));
    memcpy(V_out, V_cache, TB * TMAX_SEQ * TH * Td * sizeof(float));

    for (int64_t b = 0; b < TB; b++) {
        const float* x = X_new + b * TD;
        float* y = Y + b * TD;

        /* Init Y with bias */
        for (int64_t j = 0; j < TD; j++) y[j] = bO ? bO[j] : 0.0f;

        for (int64_t h = 0; h < TH; h++) {
            int64_t ho = h * Td;

            /* Q = x · WQ + bQ */
            float Q[16];
            for (int64_t di = 0; di < Td; di++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < TD; j++) acc += x[j] * WQ[j * TD + ho + di];
                Q[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
            }

            /* K_new, V_new → cache */
            for (int64_t di = 0; di < Td; di++) {
                float k_acc = 0.0f, v_acc = 0.0f;
                for (int64_t j = 0; j < TD; j++) {
                    k_acc += x[j] * WK[j * TD + ho + di];
                    v_acc += x[j] * WV[j * TD + ho + di];
                }
                int64_t idx = (b * TMAX_SEQ + cache_len) * TH * Td + ho + di;
                K_out[idx] = k_acc + (bK ? bK[ho + di] : 0.0f);
                V_out[idx] = v_acc + (bV ? bV[ho + di] : 0.0f);
            }

            /* Attention scores */
            int64_t total_len = cache_len + 1;
            float scores[16];
            float max_s = -1e38f;
            for (int64_t t = 0; t < total_len; t++) {
                float dot = 0.0f;
                int64_t k_off = (b * TMAX_SEQ + t) * TH * Td + ho;
                for (int64_t di = 0; di < Td; di++) dot += Q[di] * K_out[k_off + di];
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }

            /* Softmax */
            float sum_e = 0.0f;
            for (int64_t t = 0; t < total_len; t++) {
                scores[t] = expf(scores[t] - max_s);
                sum_e += scores[t];
            }
            if (sum_e < 1e-12f) sum_e = 1e-12f;

            /* Weighted V sum */
            float merged[16] = {0};
            for (int64_t t = 0; t < total_len; t++) {
                float w = scores[t] / sum_e;
                int64_t v_off = (b * TMAX_SEQ + t) * TH * Td + ho;
                for (int64_t di = 0; di < Td; di++) merged[di] += w * V_out[v_off + di];
            }

            /* Output projection — accumulate per-head to match GPU atomicAdd order */
            for (int64_t j = 0; j < TD; j++) {
                float contrib = 0.0f;
                for (int64_t di = 0; di < Td; di++) contrib += merged[di] * WO[(ho + di) * TD + j];
                y[j] += contrib;
            }
        }
    }
}

/* ============================================================
 * Test: mha_decode CPU correctness
 * ============================================================ */
static int test_mha_decode_cpu(void) {
    fprintf(stderr, "\n=== MHA Decode CPU Test ===\n");

    /* Allocate tensors */
    float* X_new   = (float*)malloc(TB * 1 * TD * sizeof(float));
    float* K_cache = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_cache = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* WQ = (float*)malloc(TD * TD * sizeof(float));
    float* bQ = (float*)malloc(TD * sizeof(float));
    float* WK = (float*)malloc(TD * TD * sizeof(float));
    float* bK = (float*)malloc(TD * sizeof(float));
    float* WV = (float*)malloc(TD * TD * sizeof(float));
    float* bV = (float*)malloc(TD * sizeof(float));
    float* WO = (float*)malloc(TD * TD * sizeof(float));
    float* bO = (float*)malloc(TD * sizeof(float));

    float* Y_cpu   = (float*)calloc(1, TB * 1 * TD * sizeof(float));
    float* Y_ref   = (float*)calloc(1, TB * 1 * TD * sizeof(float));
    float* K_ref   = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_ref   = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* K_cpu   = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_cpu   = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));

    /* Fill with deterministic random data */
    random_fill(X_new, TB * TD, 42);
    random_fill(WQ, TD * TD, 100);
    random_fill(bQ, TD, 101);
    random_fill(WK, TD * TD, 102);
    random_fill(bK, TD, 103);
    random_fill(WV, TD * TD, 104);
    random_fill(bV, TD, 105);
    random_fill(WO, TD * TD, 106);
    random_fill(bO, TD, 107);

    float scale = 1.0f / sqrtf((float)Td);

    /* --- Step 1: Decode at cache_len=0 --- */
    mha_decode_params_t p1;
    memset(&p1, 0, sizeof(p1));
    p1.batch_size = TB; p1.hidden_size = TD;
    p1.num_heads = TH; p1.num_kv_heads = TH; p1.head_dim = Td;
    p1.scale = scale; p1.cache_len = 0; p1.max_seq = TMAX_SEQ;

    const void* cpu_inputs1[] = {
        X_new, K_cache, V_cache, WQ, bQ, WK, bK, WV, bV, WO, bO
    };
    void* cpu_outputs1[] = { Y_cpu, K_cpu, V_cpu };

    int ret = mha_decode_f32(cpu_inputs1, cpu_outputs1,
                             (const operator_params_t*)&p1, NULL);
    CHECK(ret == 0, "mha_decode_f32 step 1 returned error");

    /* Reference */
    decode_ref(X_new, K_cache, V_cache, WQ, bQ, WK, bK, WV, bV, WO, bO,
               Y_ref, K_ref, V_ref, 0, scale);

    float diff = max_abs_diff(Y_cpu, Y_ref, TB * TD);
    fprintf(stderr, "Step 1 (cache_len=0): Y max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "Step 1 Y mismatch");

    /* Check K/V cache at position 0 */
    diff = max_abs_diff(K_cpu, K_ref, TB * TMAX_SEQ * TH * Td);
    fprintf(stderr, "Step 1 K_cache max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "Step 1 K_cache mismatch");

    diff = max_abs_diff(V_cpu, V_ref, TB * TMAX_SEQ * TH * Td);
    fprintf(stderr, "Step 1 V_cache max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "Step 1 V_cache mismatch");

    /* --- Step 2: Decode at cache_len=1 (use updated cache) --- */
    float* X_new2 = (float*)malloc(TB * TD * sizeof(float));
    random_fill(X_new2, TB * TD, 200);

    float* Y_cpu2 = (float*)calloc(1, TB * TD * sizeof(float));
    float* Y_ref2 = (float*)calloc(1, TB * TD * sizeof(float));
    float* K_ref2 = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_ref2 = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* K_cpu2 = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_cpu2 = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));

    mha_decode_params_t p2;
    memset(&p2, 0, sizeof(p2));
    p2.batch_size = TB; p2.hidden_size = TD;
    p2.num_heads = TH; p2.num_kv_heads = TH; p2.head_dim = Td;
    p2.scale = scale; p2.cache_len = 1; p2.max_seq = TMAX_SEQ;

    const void* cpu_inputs2[] = {
        X_new2, K_cpu, V_cpu, WQ, bQ, WK, bK, WV, bV, WO, bO
    };
    void* cpu_outputs2[] = { Y_cpu2, K_cpu2, V_cpu2 };

    ret = mha_decode_f32(cpu_inputs2, cpu_outputs2,
                         (const operator_params_t*)&p2, NULL);
    CHECK(ret == 0, "mha_decode_f32 step 2 returned error");

    decode_ref(X_new2, K_cpu, V_cpu, WQ, bQ, WK, bK, WV, bV, WO, bO,
               Y_ref2, K_ref2, V_ref2, 1, scale);

    diff = max_abs_diff(Y_cpu2, Y_ref2, TB * TD);
    fprintf(stderr, "Step 2 (cache_len=1): Y max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "Step 2 Y mismatch");

    fprintf(stderr, "MHA Decode CPU: PASS\n");

    /* Cleanup */
    free(X_new); free(X_new2);
    free(K_cache); free(V_cache);
    free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);
    free(Y_cpu); free(Y_ref); free(Y_cpu2); free(Y_ref2);
    free(K_ref); free(V_ref); free(K_ref2); free(V_ref2);
    free(K_cpu); free(V_cpu); free(K_cpu2); free(V_cpu2);
    return 0;
}

/* ============================================================
 * Test: mha_decode CUDA vs CPU
 * ============================================================ */
static int test_mha_decode_cuda(void) {
#ifdef USE_CUDA
    fprintf(stderr, "\n=== MHA Decode CUDA Test ===\n");

    /* Allocate host tensors */
    float* X_new   = (float*)malloc(TB * TD * sizeof(float));
    float* K_cache = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_cache = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* WQ = (float*)malloc(TD * TD * sizeof(float));
    float* bQ = (float*)malloc(TD * sizeof(float));
    float* WK = (float*)malloc(TD * TD * sizeof(float));
    float* bK = (float*)malloc(TD * sizeof(float));
    float* WV = (float*)malloc(TD * TD * sizeof(float));
    float* bV = (float*)malloc(TD * sizeof(float));
    float* WO = (float*)malloc(TD * TD * sizeof(float));
    float* bO = (float*)malloc(TD * sizeof(float));

    float* Y_cpu = (float*)calloc(1, TB * TD * sizeof(float));
    float* Y_gpu = (float*)calloc(1, TB * TD * sizeof(float));
    float* K_cpu = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_cpu = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* K_gpu = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));
    float* V_gpu = (float*)calloc(1, TB * TMAX_SEQ * TH * Td * sizeof(float));

    random_fill(X_new, TB * TD, 42);
    random_fill(WQ, TD * TD, 100); random_fill(bQ, TD, 101);
    random_fill(WK, TD * TD, 102); random_fill(bK, TD, 103);
    random_fill(WV, TD * TD, 104); random_fill(bV, TD, 105);
    random_fill(WO, TD * TD, 106); random_fill(bO, TD, 107);

    float scale = 1.0f / sqrtf((float)Td);

    mha_decode_params_t p;
    memset(&p, 0, sizeof(p));
    p.batch_size = TB; p.hidden_size = TD;
    p.num_heads = TH; p.num_kv_heads = TH; p.head_dim = Td;
    p.scale = scale; p.cache_len = 0; p.max_seq = TMAX_SEQ;

    /* CPU reference */
    const void* cpu_in[] = { X_new, K_cache, V_cache, WQ, bQ, WK, bK, WV, bV, WO, bO };
    void* cpu_out[] = { Y_cpu, K_cpu, V_cpu };
    mha_decode_f32(cpu_in, cpu_out, (const operator_params_t*)&p, NULL);

    /* CUDA */
    /* Create tensors for CUDA dispatch */
    int64_t x_shape[] = {TB, 1, TD};
    int64_t cache_shape[] = {TB, TMAX_SEQ, TH, Td};
    int64_t w_shape[] = {TD, TD};
    int64_t b_shape[] = {TD};

    tensor_t* tX = tensor_create(DATA_TYPE_F32, 3, x_shape);
    tensor_t* tKc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVc = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tWQ = tensor_create(DATA_TYPE_F32, 2, w_shape);
    tensor_t* tbQ = tensor_create(DATA_TYPE_F32, 1, b_shape);
    tensor_t* tWK = tensor_create(DATA_TYPE_F32, 2, w_shape);
    tensor_t* tbK = tensor_create(DATA_TYPE_F32, 1, b_shape);
    tensor_t* tWV = tensor_create(DATA_TYPE_F32, 2, w_shape);
    tensor_t* tbV = tensor_create(DATA_TYPE_F32, 1, b_shape);
    tensor_t* tWO = tensor_create(DATA_TYPE_F32, 2, w_shape);
    tensor_t* tbO = tensor_create(DATA_TYPE_F32, 1, b_shape);
    tensor_t* tY = tensor_create(DATA_TYPE_F32, 3, x_shape);
    tensor_t* tKo = tensor_create(DATA_TYPE_F32, 4, cache_shape);
    tensor_t* tVo = tensor_create(DATA_TYPE_F32, 4, cache_shape);

    /* Copy data to tensors */
    memcpy(tX->data, X_new, TB * TD * sizeof(float));
    memcpy(tKc->data, K_cache, TB * TMAX_SEQ * TH * Td * sizeof(float));
    memcpy(tVc->data, V_cache, TB * TMAX_SEQ * TH * Td * sizeof(float));
    memcpy(tWQ->data, WQ, TD * TD * sizeof(float));
    memcpy(tbQ->data, bQ, TD * sizeof(float));
    memcpy(tWK->data, WK, TD * TD * sizeof(float));
    memcpy(tbK->data, bK, TD * sizeof(float));
    memcpy(tWV->data, WV, TD * TD * sizeof(float));
    memcpy(tbV->data, bV, TD * sizeof(float));
    memcpy(tWO->data, WO, TD * TD * sizeof(float));
    memcpy(tbO->data, bO, TD * sizeof(float));

    /* Copy to device */
    tensor_copy_to_device(tX); tensor_copy_to_device(tKc); tensor_copy_to_device(tVc);
    tensor_copy_to_device(tWQ); tensor_copy_to_device(tbQ);
    tensor_copy_to_device(tWK); tensor_copy_to_device(tbK);
    tensor_copy_to_device(tWV); tensor_copy_to_device(tbV);
    tensor_copy_to_device(tWO); tensor_copy_to_device(tbO);
    tensor_copy_to_device(tY); tensor_copy_to_device(tKo); tensor_copy_to_device(tVo);

    /* Look up CUDA operator */
    const operator_registry_t* op = operator_find("mha_decode_f32_cuda");
    if (!op || !op->func) {
        fprintf(stderr, "SKIP: mha_decode_f32_cuda not registered\n");
        goto cleanup_cuda;
    }

    const void* gpu_in[] = {
        tX->data_device, tKc->data_device, tVc->data_device,
        tWQ->data_device, tbQ->data_device,
        tWK->data_device, tbK->data_device,
        tWV->data_device, tbV->data_device,
        tWO->data_device, tbO->data_device
    };
    void* gpu_out[] = { tY->data_device, tKo->data_device, tVo->data_device };

    int ret = op->func(gpu_in, gpu_out, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "mha_decode_f32_cuda returned error");

    /* Copy back */
    tensor_copy_to_host(tY); tensor_copy_to_host(tKo); tensor_copy_to_host(tVo);
    g_cuda.stream_synchronize(0);

    memcpy(Y_gpu, tY->data, TB * TD * sizeof(float));
    memcpy(K_gpu, tKo->data, TB * TMAX_SEQ * TH * Td * sizeof(float));
    memcpy(V_gpu, tVo->data, TB * TMAX_SEQ * TH * Td * sizeof(float));

    /* Compare */
    float diff;
    diff = max_abs_diff(K_gpu, K_cpu, TB * TMAX_SEQ * TH * Td);
    fprintf(stderr, "CUDA vs CPU: K_cache max_diff=%.2e\n", diff);
    CHECK(diff < 1e-3f, "CUDA K_cache mismatch");

    diff = max_abs_diff(V_gpu, V_cpu, TB * TMAX_SEQ * TH * Td);
    fprintf(stderr, "CUDA vs CPU: V_cache max_diff=%.2e\n", diff);
    CHECK(diff < 1e-3f, "CUDA V_cache mismatch");

    diff = max_abs_diff(Y_gpu, Y_cpu, TB * TD);
    fprintf(stderr, "CUDA vs CPU: Y max_diff=%.2e\n", diff);
    CHECK(diff < 1e-3f, "CUDA Y mismatch");

    diff = max_abs_diff(K_gpu, K_cpu, TB * TMAX_SEQ * TH * Td);
    fprintf(stderr, "CUDA vs CPU: K_cache max_diff=%.2e\n", diff);
    CHECK(diff < 1e-3f, "CUDA K_cache mismatch");

    fprintf(stderr, "MHA Decode CUDA: PASS\n");

cleanup_cuda:
    tensor_destroy(tX); tensor_destroy(tKc); tensor_destroy(tVc);
    tensor_destroy(tWQ); tensor_destroy(tbQ);
    tensor_destroy(tWK); tensor_destroy(tbK);
    tensor_destroy(tWV); tensor_destroy(tbV);
    tensor_destroy(tWO); tensor_destroy(tbO);
    tensor_destroy(tY); tensor_destroy(tKo); tensor_destroy(tVo);
    free(X_new); free(K_cache); free(V_cache);
    free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);
    free(Y_cpu); free(Y_gpu); free(K_cpu); free(V_cpu);
    free(K_gpu); free(V_gpu);
#else
    (void)0;
#endif
    return 0;
}

/* ============================================================
 * Test: CausalMask operator
 * ============================================================ */
static int test_causal_mask(void) {
    fprintf(stderr, "\n=== Causal Mask Test ===\n");

    int64_t S = 4;
    int64_t shape[] = {S, S};
    tensor_t* tMask = tensor_create(DATA_TYPE_F32, 2, shape);

    causal_mask_params_t mp;
    mp.seq_len = S;

    /* CPU */
    const void* cpu_in[] = { NULL };
    void* cpu_out[] = { tMask->data };
    int ret = causal_mask_f32(cpu_in, cpu_out, (const operator_params_t*)&mp, NULL);
    CHECK(ret == 0, "causal_mask_f32 returned error");

    /* Verify lower triangular: 0 on/below diagonal, -inf above */
    float* mask = (float*)tMask->data;
    int ok = 1;
    for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
            float val = mask[i * S + j];
            if (j <= i) {
                if (val != 0.0f) { ok = 0; fprintf(stderr, "FAIL: mask[%lld][%lld]=%f, expected 0\n", (long long)i, (long long)j, val); }
            } else {
                if (isfinite(val)) { ok = 0; fprintf(stderr, "FAIL: mask[%lld][%lld]=%f, expected -inf\n", (long long)i, (long long)j, val); }
            }
        }
    }
    CHECK(ok, "Causal mask values incorrect");

    fprintf(stderr, "Causal Mask: PASS\n");
    tensor_destroy(tMask);
    return 0;
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_causal_mask();
    test_mha_decode_cpu();
    test_mha_decode_cuda();

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();

    fprintf(stderr, "\n=== All MHA Decode Tests Done ===\n");
    return 0;
}
