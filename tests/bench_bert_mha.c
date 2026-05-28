/**
 * BERT-base MHA fused kernel benchmark.
 *
 * Measures execution time of the MHA_Fused CUDA kernel vs CPU reference,
 * using BERT-base dimensions: B=1, S=8, D=768, H=12, d=64.
 */
#include "graph.h"
#include "mha_fused_int.h"
#include "platform.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

#define B  1
#define S  8
#define H  12
#define d  64
#define D  768

#define WARMUP_ITERS 5
#define BENCH_ITERS   100

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static void random_fill(float* data, int64_t n, unsigned seed) {
    unsigned state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = ((float)(state & 0x7FFFFFFF) / 2147483648.0f) - 1.0f;
    }
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    int64_t BS = B * S;
    int64_t n_elem = BS * D;

    fprintf(stderr, "=== BERT-base MHA Fused Benchmark ===\n");
    fprintf(stderr, "B=%d  S=%d  D=%d  H=%d  d=%d\n", B, S, D, H, d);
    fprintf(stderr, "Input: %.1f KB,  WQKV: %.1f KB,  WO: %.1f KB\n",
            n_elem * 4.0 / 1024.0,
            4 * D * D * 4.0 / 1024.0,
            D * D * 4.0 / 1024.0);

    /* Allocate data */
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

    /* Build graph */
    inference_graph_t* g = graph_create();
    if (!g) { fprintf(stderr, "graph_create failed\n"); return 1; }

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

    graph_add_node(g, OP_MHA_FUSED, 2, input_tids, 1, output_tids, 8, weights,
                   &params, sizeof(params));
    int in_id  = graph_add_node(g, OP_INPUT,  0, NULL, 1, (int[]){tidX}, 0, NULL, NULL, 0);
    int out_id = graph_add_node(g, OP_OUTPUT,  1, (int[]){tidY}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_input(g, in_id);
    graph_set_output(g, out_id);

    if (graph_build(g) != 0) {
        fprintf(stderr, "graph_build failed\n"); return 1;
    }

    /* ---- Verify correctness first ---- */
#ifdef USE_CUDA
    {
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        float* Y_cpu  = (float*)malloc(n_elem * sizeof(float));
        float* Y_cuda = (float*)malloc(n_elem * sizeof(float));

        graph_execute(g, inputs, outputs, false);
        memcpy(Y_cpu, tY->data, n_elem * sizeof(float));

        memset(tY->data, 0, n_elem * sizeof(float));
        graph_execute(g, inputs, outputs, true);
        memcpy(Y_cuda, tY->data, n_elem * sizeof(float));

        float max_abs = 0.0f;
        double sum_abs = 0.0;
        float max_rel = 0.0f;
        float max_cpu = 0.0f;
        for (int64_t i = 0; i < n_elem; i++) {
            float diff = fabsf(Y_cpu[i] - Y_cuda[i]);
            if (diff > max_abs) max_abs = diff;
            sum_abs += (double)diff;
            float cpu_abs = fabsf(Y_cpu[i]);
            if (cpu_abs > max_cpu) max_cpu = cpu_abs;
            float rel = cpu_abs > 1e-8f ? diff / cpu_abs : diff;
            if (rel > max_rel) max_rel = rel;
        }
        fprintf(stderr, "CUDA vs CPU: max_abs=%.2e  mean_abs=%.2e  max_rel=%.2e  max_val=%.1f\n",
                max_abs, sum_abs / n_elem, max_rel, max_cpu);
        if (max_rel > 1e-4f)
            fprintf(stderr, "WARNING: large relative difference (%.2e)\n\n", max_rel);

        free(Y_cpu); free(Y_cuda);
    }
#endif

    /* ---- CPU benchmark ---- */
    fprintf(stderr, "\nCPU (%d warmup + %d iters):\n", WARMUP_ITERS, BENCH_ITERS);
    {
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        for (int i = 0; i < WARMUP_ITERS; i++) {
            graph_execute(g, inputs, outputs, false);
        }
        double t0 = now_ms();
        for (int i = 0; i < BENCH_ITERS; i++) {
            graph_execute(g, inputs, outputs, false);
        }
        double t1 = now_ms();
        double total = t1 - t0;
        fprintf(stderr, "  total: %.2f ms,  avg: %.3f ms/iter\n",
                total, total / BENCH_ITERS);
    }

#ifdef USE_CUDA
    /* ---- CUDA benchmark ---- */
    fprintf(stderr, "\nCUDA (%d warmup + %d iters):\n", WARMUP_ITERS, BENCH_ITERS);
    {
        tensor_t* inputs[]  = {tX};
        tensor_t* outputs[] = {tY};
        for (int i = 0; i < WARMUP_ITERS; i++) {
            graph_execute(g, inputs, outputs, true);
        }
        double t0 = now_ms();
        for (int i = 0; i < BENCH_ITERS; i++) {
            graph_execute(g, inputs, outputs, true);
        }
        double t1 = now_ms();
        double total = t1 - t0;
        fprintf(stderr, "  total: %.2f ms,  avg: %.3f ms/iter\n",
                total, total / BENCH_ITERS);
    }

    cuda_platform_finalize();
#endif

    graph_destroy(g);
    free(X); free(R); free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);

    platform_finalize();
    fprintf(stderr, "\nBench done.\n");
    return 0;
}
