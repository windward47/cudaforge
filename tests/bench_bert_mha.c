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

    /* ---- FP16 benchmark ---- */
    {
        const operator_registry_t* fp16_reg = operator_find("mha_fused_f16_cuda");
        if (fp16_reg && fp16_reg->func) {
            fprintf(stderr, "\nCUDA FP16 (%d warmup + %d iters):\n", WARMUP_ITERS, BENCH_ITERS);

            /* Prepare device pointers */
            tensor_copy_to_device(tX); tensor_copy_to_device(tR);
            tensor_copy_to_device(twQ); tensor_copy_to_device(tbQ);
            tensor_copy_to_device(twK); tensor_copy_to_device(tbK);
            tensor_copy_to_device(twV); tensor_copy_to_device(tbV);
            tensor_copy_to_device(twO); tensor_copy_to_device(tbO);
            tensor_copy_to_device(tY);

            mha_fused_params_t fp;
            memset(&fp, 0, sizeof(fp));
            fp.batch_size = B; fp.seq_len = S;
            fp.hidden_size = D; fp.num_heads = H;
            fp.head_dim = d; fp.scale = scale;
            fp.has_residual = true;

            const void* fp16_inputs[] = {
                tX->data_device, tR->data_device,
                twQ->data_device, tbQ->data_device,
                twK->data_device, tbK->data_device,
                twV->data_device, tbV->data_device,
                twO->data_device, tbO->data_device,
            };
            void* fp16_outputs[] = { tY->data_device };

            for (int i = 0; i < WARMUP_ITERS; i++) {
                fp16_reg->func(fp16_inputs, fp16_outputs, (const operator_params_t*)&fp, NULL);
            }
            /* Sync via platform API to ensure kernel completes */
            g_cuda.stream_synchronize(NULL);
            double t0 = now_ms();
            for (int i = 0; i < BENCH_ITERS; i++) {
                fp16_reg->func(fp16_inputs, fp16_outputs, (const operator_params_t*)&fp, NULL);
            }
            g_cuda.stream_synchronize(NULL);
            double t1 = now_ms();
            double total = t1 - t0;
            fprintf(stderr, "  total: %.2f ms,  avg: %.3f ms/iter\n",
                    total, total / BENCH_ITERS);
        } else {
            fprintf(stderr, "\nCUDA FP16: SKIP (not registered)\n");
        }
    }

    cuda_platform_finalize();
#endif

    graph_destroy(g);
    free(X); free(R); free(WQ); free(bQ); free(WK); free(bK);
    free(WV); free(bV); free(WO); free(bO);

    /* ---- Long sequence benchmark (Flash Attention path) ---- */
#ifdef USE_CUDA
    cuda_platform_init(0);
    fprintf(stderr, "\n=== Flash Attention Long Sequence Benchmark ===\n");

    int64_t long_S_vals[] = {128, 256, 512};
    int num_long = sizeof(long_S_vals) / sizeof(long_S_vals[0]);

    for (int si = 0; si < num_long; si++) {
        int64_t lS = long_S_vals[si];
        int64_t lBS = B * lS;
        int64_t l_elem = lBS * D;
        int l_iters = (lS <= 256) ? 50 : 20;

        float *lX = (float*)malloc(l_elem * sizeof(float));
        float *lR = (float*)malloc(l_elem * sizeof(float));
        float *lWQ = (float*)malloc(D * D * sizeof(float));
        float *lbQ = (float*)malloc(D * sizeof(float));
        float *lWK = (float*)malloc(D * D * sizeof(float));
        float *lbK = (float*)malloc(D * sizeof(float));
        float *lWV = (float*)malloc(D * D * sizeof(float));
        float *lbV = (float*)malloc(D * sizeof(float));
        float *lWO = (float*)malloc(D * D * sizeof(float));
        float *lbO = (float*)malloc(D * sizeof(float));

        random_fill(lX, l_elem, 42); random_fill(lR, l_elem, 99);
        random_fill(lWQ, D*D, 123); random_fill(lbQ, D, 456);
        random_fill(lWK, D*D, 789); random_fill(lbK, D, 234);
        random_fill(lWV, D*D, 567); random_fill(lbV, D, 890);
        random_fill(lWO, D*D, 345); random_fill(lbO, D, 678);

        inference_graph_t* lg = graph_create();
        tensor_t* ltX = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, lS, D});
        tensor_t* ltR = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, lS, D});
        tensor_t* ltY = tensor_create(DATA_TYPE_F32, 3, (int64_t[]){B, lS, D});
        tensor_t* ltwQ = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
        tensor_t* ltbQ = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
        tensor_t* ltwK = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
        tensor_t* ltbK = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
        tensor_t* ltwV = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
        tensor_t* ltbV = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});
        tensor_t* ltwO = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){D, D});
        tensor_t* ltbO = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){D});

        memcpy(ltX->data, lX, l_elem * sizeof(float));
        memcpy(ltR->data, lR, l_elem * sizeof(float));
        memcpy(ltwQ->data, lWQ, D*D*sizeof(float)); memcpy(ltbQ->data, lbQ, D*sizeof(float));
        memcpy(ltwK->data, lWK, D*D*sizeof(float)); memcpy(ltbK->data, lbK, D*sizeof(float));
        memcpy(ltwV->data, lWV, D*D*sizeof(float)); memcpy(ltbV->data, lbV, D*sizeof(float));
        memcpy(ltwO->data, lWO, D*D*sizeof(float)); memcpy(ltbO->data, lbO, D*sizeof(float));

        int ltidX = graph_add_tensor(lg, ltX);
        int ltidR = graph_add_tensor(lg, ltR);
        int ltidY = graph_add_tensor(lg, ltY);
        graph_add_tensor(lg, ltwQ); graph_add_tensor(lg, ltbQ);
        graph_add_tensor(lg, ltwK); graph_add_tensor(lg, ltbK);
        graph_add_tensor(lg, ltwV); graph_add_tensor(lg, ltbV);
        graph_add_tensor(lg, ltwO); graph_add_tensor(lg, ltbO);

        mha_fused_params_t lp;
        memset(&lp, 0, sizeof(lp));
        lp.batch_size = B; lp.seq_len = lS; lp.hidden_size = D;
        lp.num_heads = H; lp.head_dim = d; lp.scale = scale;
        lp.has_residual = true;

        int lin_tids[] = {ltidX, ltidR};
        int lout_tids[] = {ltidY};
        tensor_t* lw[] = {ltwQ, ltbQ, ltwK, ltbK, ltwV, ltbV, ltwO, ltbO};
        graph_add_node(lg, OP_MHA_FUSED, 2, lin_tids, 1, lout_tids, 8, lw, &lp, sizeof(lp));
        int lin_id = graph_add_node(lg, OP_INPUT, 0, NULL, 1, (int[]){ltidX}, 0, NULL, NULL, 0);
        int lout_id = graph_add_node(lg, OP_OUTPUT, 1, (int[]){ltidY}, 0, NULL, 0, NULL, NULL, 0);
        graph_set_input(lg, lin_id);
        graph_set_output(lg, lout_id);
        graph_build(lg);

        /* Warmup + bench CUDA */
        tensor_t* linputs[] = {ltX};
        tensor_t* loutputs[] = {ltY};
        for (int i = 0; i < 3; i++) {
            memset(ltY->data, 0, l_elem * sizeof(float));
            graph_execute(lg, linputs, loutputs, true);
        }
        double lt0 = now_ms();
        for (int i = 0; i < l_iters; i++) {
            memset(ltY->data, 0, l_elem * sizeof(float));
            graph_execute(lg, linputs, loutputs, true);
        }
        double lt1 = now_ms();
        double ltotal = lt1 - lt0;
        const char* path = (lS <= 512) ? "flash_v2" : "split_kv";
        fprintf(stderr, "  S=%-4lld CUDA: %.3f ms/iter  (%d iters, %s)\n",
                (long long)lS, ltotal / l_iters, l_iters, path);

        free(lX); free(lR); free(lWQ); free(lbQ); free(lWK); free(lbK);
        free(lWV); free(lbV); free(lWO); free(lbO);
        graph_destroy(lg);
    }
    cuda_platform_finalize();
#endif

    platform_finalize();
    fprintf(stderr, "\nBench done.\n");
    return 0;
}
