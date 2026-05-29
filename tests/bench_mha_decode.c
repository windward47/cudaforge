/**
 * MHA decode benchmark — measures per-token decode latency with KV-cache.
 * Compares: CPU vs CUDA for single-token attention with growing cache.
 */
#include "mha_decode_int.h"
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

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* BERT-base-like dimensions */
#define B  1
#define D  768
#define H  12
#define d  64
#define MAX_SEQ 512
#define WARMUP_ITERS 5
#define BENCH_ITERS  50

static void random_fill(float* data, int64_t n, int seed) {
    srand(seed);
    for (int64_t i = 0; i < n; i++) data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    fprintf(stderr, "=== MHA Decode Benchmark ===\n");
    fprintf(stderr, "B=%d  D=%d  H=%d  d=%d  max_seq=%d\n", B, D, H, d, MAX_SEQ);

    float scale = 1.0f / sqrtf((float)d);

    /* Allocate tensors */
    float* X_new   = (float*)malloc(B * D * sizeof(float));
    float* K_cache = (float*)calloc(1, B * MAX_SEQ * H * d * sizeof(float));
    float* V_cache = (float*)calloc(1, B * MAX_SEQ * H * d * sizeof(float));
    float* WQ = (float*)malloc(D * D * sizeof(float));
    float* bQ = (float*)malloc(D * sizeof(float));
    float* WK = (float*)malloc(D * D * sizeof(float));
    float* bK = (float*)malloc(D * sizeof(float));
    float* WV = (float*)malloc(D * D * sizeof(float));
    float* bV = (float*)malloc(D * sizeof(float));
    float* WO = (float*)malloc(D * D * sizeof(float));
    float* bO = (float*)malloc(D * sizeof(float));
    float* Y = (float*)calloc(1, B * D * sizeof(float));
    float* K_out = (float*)calloc(1, B * MAX_SEQ * H * d * sizeof(float));
    float* V_out = (float*)calloc(1, B * MAX_SEQ * H * d * sizeof(float));

    random_fill(X_new, B * D, 42);
    random_fill(WQ, D * D, 100); random_fill(bQ, D, 101);
    random_fill(WK, D * D, 102); random_fill(bK, D, 103);
    random_fill(WV, D * D, 104); random_fill(bV, D, 105);
    random_fill(WO, D * D, 106); random_fill(bO, D, 107);

    /* Test at different cache lengths */
    int cache_lens[] = {0, 32, 128, 256, 511};
    int num_tests = sizeof(cache_lens) / sizeof(cache_lens[0]);

    for (int t = 0; t < num_tests; t++) {
        int64_t cl = cache_lens[t];
        mha_decode_params_t p;
        memset(&p, 0, sizeof(p));
        p.batch_size = B; p.hidden_size = D;
        p.num_heads = H; p.head_dim = d;
        p.scale = scale; p.cache_len = cl; p.max_seq = MAX_SEQ;

        /* CPU benchmark */
        {
            for (int i = 0; i < WARMUP_ITERS; i++) {
                const void* in[] = { X_new, K_cache, V_cache, WQ, bQ, WK, bK, WV, bV, WO, bO };
                void* out[] = { Y, K_out, V_out };
                mha_decode_f32(in, out, (const operator_params_t*)&p, NULL);
            }
            double t0 = now_ms();
            for (int i = 0; i < BENCH_ITERS; i++) {
                const void* in[] = { X_new, K_cache, V_cache, WQ, bQ, WK, bK, WV, bV, WO, bO };
                void* out[] = { Y, K_out, V_out };
                mha_decode_f32(in, out, (const operator_params_t*)&p, NULL);
            }
            double t1 = now_ms();
            fprintf(stderr, "CPU   cache_len=%3lld: %.3f ms/iter\n",
                    (long long)cl, (t1 - t0) / BENCH_ITERS);
        }

#ifdef USE_CUDA
        /* CUDA benchmark */
        {
            const operator_registry_t* op = operator_find("mha_decode_f32_cuda");
            if (op && op->func) {
                /* Create CUDA tensors */
                int64_t x_shape[] = {B, 1, D};
                int64_t cache_shape[] = {B, MAX_SEQ, H, d};
                int64_t w_shape[] = {D, D};
                int64_t b_shape[] = {D};

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

                memcpy(tX->data, X_new, B * D * sizeof(float));
                memcpy(tKc->data, K_cache, B * MAX_SEQ * H * d * sizeof(float));
                memcpy(tVc->data, V_cache, B * MAX_SEQ * H * d * sizeof(float));
                memcpy(tWQ->data, WQ, D * D * sizeof(float)); memcpy(tbQ->data, bQ, D * sizeof(float));
                memcpy(tWK->data, WK, D * D * sizeof(float)); memcpy(tbK->data, bK, D * sizeof(float));
                memcpy(tWV->data, WV, D * D * sizeof(float)); memcpy(tbV->data, bV, D * sizeof(float));
                memcpy(tWO->data, WO, D * D * sizeof(float)); memcpy(tbO->data, bO, D * sizeof(float));

                tensor_copy_to_device(tX); tensor_copy_to_device(tKc); tensor_copy_to_device(tVc);
                tensor_copy_to_device(tWQ); tensor_copy_to_device(tbQ);
                tensor_copy_to_device(tWK); tensor_copy_to_device(tbK);
                tensor_copy_to_device(tWV); tensor_copy_to_device(tbV);
                tensor_copy_to_device(tWO); tensor_copy_to_device(tbO);
                tensor_copy_to_device(tY); tensor_copy_to_device(tKo); tensor_copy_to_device(tVo);

                const void* gpu_in[] = {
                    tX->data_device, tKc->data_device, tVc->data_device,
                    tWQ->data_device, tbQ->data_device,
                    tWK->data_device, tbK->data_device,
                    tWV->data_device, tbV->data_device,
                    tWO->data_device, tbO->data_device
                };
                void* gpu_out[] = { tY->data_device, tKo->data_device, tVo->data_device };

                for (int i = 0; i < WARMUP_ITERS; i++) {
                    op->func(gpu_in, gpu_out, (const operator_params_t*)&p, NULL);
                }
                g_cuda.stream_synchronize(0);
                double t0 = now_ms();
                for (int i = 0; i < BENCH_ITERS; i++) {
                    op->func(gpu_in, gpu_out, (const operator_params_t*)&p, NULL);
                }
                g_cuda.stream_synchronize(0);
                double t1 = now_ms();
                fprintf(stderr, "CUDA  cache_len=%3lld: %.3f ms/iter\n",
                        (long long)cl, (t1 - t0) / BENCH_ITERS);

                tensor_destroy(tX); tensor_destroy(tKc); tensor_destroy(tVc);
                tensor_destroy(tWQ); tensor_destroy(tbQ);
                tensor_destroy(tWK); tensor_destroy(tbK);
                tensor_destroy(tWV); tensor_destroy(tbV);
                tensor_destroy(tWO); tensor_destroy(tbO);
                tensor_destroy(tY); tensor_destroy(tKo); tensor_destroy(tVo);
            }
        }
#endif
        fprintf(stderr, "\n");
    }

    free(X_new); free(K_cache); free(V_cache);
    free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);
    free(Y); free(K_out); free(V_out);

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();
    fprintf(stderr, "Bench done.\n");
    return 0;
}
