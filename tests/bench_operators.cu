extern "C" {
#include "platform.h"
int operator_init_all(void);
}

#include "operator.h"
#include "matmul_int.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "batchnorm_int.h"
#include "cuda_ops.h"
#include "cuda_platform.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define WARMUP 3
#define REPEAT 10

static double now_ms(void) {
    return (double)clock() / (CLOCKS_PER_SEC / 1000.0);
}

/* ============================================================
 * MatMul benchmark
 * ============================================================ */
static void bench_matmul(int M, int N, int K) {
    size_t size_a = M * K, size_b = K * N, size_c = M * N;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size_a * sizeof(float));
    h_B = (float*)malloc(size_b * sizeof(float));
    h_C = (float*)malloc(size_c * sizeof(float));
    for (size_t i = 0; i < size_a; i++) h_A[i] = (float)(i % 7 - 3);
    for (size_t i = 0; i < size_b; i++) h_B[i] = (float)(i % 5 - 2);

    d_A = (float*)g_cuda.device_alloc(size_a * sizeof(float));
    d_B = (float*)g_cuda.device_alloc(size_b * sizeof(float));
    d_C = (float*)g_cuda.device_alloc(size_c * sizeof(float));
    g_cuda.memcpy_h2d(d_A, h_A, size_a * sizeof(float), 0);
    g_cuda.memcpy_h2d(d_B, h_B, size_b * sizeof(float), 0);

    matmul_params_t p = {.M = M, .N = N, .K = K};

    /* --- CPU --- */
    const operator_registry_t* cpu_op = operator_find("matmul_f32");
    {
        const void* inputs[]  = {h_A, h_B};
        void*       outputs[] = {h_C};
        for (int w = 0; w < WARMUP; w++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
        double t1 = now_ms();
        printf("  matmul(%d,%d,%d)  CPU: %8.2f ms\n", M, N, K, (t1 - t0) / REPEAT);
    }

    /* --- CUDA --- */
    const operator_registry_t* cuda_op = operator_find("matmul_f32_cuda");
    if (cuda_op) {
        const void* inputs[]  = {d_A, d_B};
        void*       outputs[] = {d_C};
        for (int w = 0; w < WARMUP; w++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t1 = now_ms();
        printf("  matmul(%d,%d,%d)  GPU: %8.2f ms\n", M, N, K, (t1 - t0) / REPEAT);
    }

    g_cuda.device_free(d_A); g_cuda.device_free(d_B); g_cuda.device_free(d_C);
    free(h_A); free(h_B); free(h_C);
}

/* ============================================================
 * Conv2D benchmark
 * ============================================================ */
static void bench_conv2d(int H, int W, int C, int K, int KH, int KW) {
    int64_t OH = H - KH + 1, OW = W - KW + 1;
    size_t size_in = C * H * W, size_w = K * C * KH * KW, size_out = K * OH * OW;
    float *h_in = (float*)malloc(size_in * sizeof(float));
    float *h_w  = (float*)malloc(size_w  * sizeof(float));
    float *h_out = (float*)malloc(size_out * sizeof(float));
    for (size_t i = 0; i < size_in; i++) h_in[i] = (float)(i % 7 - 3);
    for (size_t i = 0; i < size_w;  i++) h_w[i]  = (float)((i % 3) - 1);

    float *d_in, *d_w, *d_out;
    d_in = (float*)g_cuda.device_alloc(size_in  * sizeof(float));
    d_w = (float*)g_cuda.device_alloc(size_w   * sizeof(float));
    d_out = (float*)g_cuda.device_alloc(size_out * sizeof(float));
    g_cuda.memcpy_h2d(d_in, h_in, size_in * sizeof(float), 0);
    g_cuda.memcpy_h2d(d_w, h_w, size_w  * sizeof(float), 0);

    conv_params_t cp;
    memset(&cp, 0, sizeof(cp));
    cp.N = 1; cp.C = C; cp.H = H; cp.W = W; cp.K = K;
    cp.kernel_h = KH; cp.kernel_w = KW;
    cp.stride_h = 1; cp.stride_w = 1;
    cp.dilation_h = 1; cp.dilation_w = 1;

    /* --- CPU --- */
    const operator_registry_t* cpu_op = operator_find("conv2d_f32");
    {
        const void* inputs[]  = {h_in, h_w};
        void*       outputs[] = {h_out};
        for (int w = 0; w < WARMUP; w++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&cp, NULL);
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&cp, NULL);
        double t1 = now_ms();
        printf("  conv2d(%dx%d,c=%d,k=%d,%dx%d) CPU: %8.2f ms\n",
               H, W, C, K, KH, KW, (t1 - t0) / REPEAT);
    }

    /* --- CUDA --- */
    const operator_registry_t* cuda_op = operator_find("conv2d_f32_cuda");
    if (cuda_op) {
        const void* inputs[]  = {d_in, d_w};
        void*       outputs[] = {d_out};
        for (int w = 0; w < WARMUP; w++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&cp, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&cp, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t1 = now_ms();
        printf("  conv2d(%dx%d,c=%d,k=%d,%dx%d) GPU: %8.2f ms\n",
               H, W, C, K, KH, KW, (t1 - t0) / REPEAT);
    }

    g_cuda.device_free(d_in); g_cuda.device_free(d_w); g_cuda.device_free(d_out);
    free(h_in); free(h_w); free(h_out);
}

/* ============================================================
 * Activation benchmark (ReLU, Sigmoid, GELU)
 * ============================================================ */
static void bench_activation(const char* name, const char* op_cpu, const char* op_cuda, int64_t n) {
    float *h_in  = (float*)malloc(n * sizeof(float));
    float *h_out = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) h_in[i] = (float)((i % 5) - 2.0f);

    float *d_in, *d_out;
    d_in = (float*)g_cuda.device_alloc(n * sizeof(float));
    d_out = (float*)g_cuda.device_alloc(n * sizeof(float));
    g_cuda.memcpy_h2d(d_in, h_in, n * sizeof(float), 0);

    /* --- CPU --- */
    const operator_registry_t* cpu_op = operator_find(op_cpu);
    if (cpu_op) {
        const void* inputs[]  = {h_in, &n};
        void*       outputs[] = {h_out};
        for (int w = 0; w < WARMUP; w++)
            cpu_op->func(inputs, outputs, NULL, NULL);
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++)
            cpu_op->func(inputs, outputs, NULL, NULL);
        double t1 = now_ms();
        printf("  %-10s (n=%-8lld) CPU: %8.2f ms\n", name, (long long)n, (t1 - t0) / REPEAT);
    }

    /* --- CUDA --- */
    const operator_registry_t* cuda_op = operator_find(op_cuda);
    if (cuda_op) {
        const void* inputs[]  = {d_in, &n};
        void*       outputs[] = {d_out};
        for (int w = 0; w < WARMUP; w++) {
            cuda_op->func(inputs, outputs, NULL, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++) {
            cuda_op->func(inputs, outputs, NULL, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t1 = now_ms();
        printf("  %-10s (n=%-8lld) GPU: %8.2f ms\n", name, (long long)n, (t1 - t0) / REPEAT);
    }

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
    free(h_in); free(h_out);
}

/* ============================================================
 * Pooling benchmark
 * ============================================================ */
static void bench_pool(const char* name, const char* op_cpu, const char* op_cuda,
                       int H, int W, int KH, int KW, int stride) {
    int64_t OH = (H - KH) / stride + 1, OW = (W - KW) / stride + 1;
    size_t n_in = H * W, n_out = OH * OW;
    float *h_in  = (float*)malloc(n_in  * sizeof(float));
    float *h_out = (float*)malloc(n_out * sizeof(float));
    for (size_t i = 0; i < n_in; i++) h_in[i] = (float)(i % 10);

    float *d_in, *d_out;
    d_in = (float*)g_cuda.device_alloc(n_in  * sizeof(float));
    d_out = (float*)g_cuda.device_alloc(n_out * sizeof(float));
    g_cuda.memcpy_h2d(d_in, h_in, n_in * sizeof(float), 0);

    pool_params_t pp = {.N=1,.C=1,.H=H,.W=W,.kernel_h=KH,.kernel_w=KW,.stride_h=stride,.stride_w=stride};

    /* --- CPU --- */
    const operator_registry_t* cpu_op = operator_find(op_cpu);
    if (cpu_op) {
        const void* inputs[]  = {h_in};
        void*       outputs[] = {h_out};
        for (int w = 0; w < WARMUP; w++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&pp, NULL);
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++)
            cpu_op->func(inputs, outputs, (const operator_params_t*)&pp, NULL);
        double t1 = now_ms();
        printf("  %-12s (%dx%d,%dx%d) CPU: %8.2f ms\n", name, H, W, KH, KW, (t1 - t0) / REPEAT);
    }

    /* --- CUDA --- */
    const operator_registry_t* cuda_op = operator_find(op_cuda);
    if (cuda_op) {
        const void* inputs[]  = {d_in};
        void*       outputs[] = {d_out};
        for (int w = 0; w < WARMUP; w++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&pp, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t0 = now_ms();
        for (int r = 0; r < REPEAT; r++) {
            cuda_op->func(inputs, outputs, (const operator_params_t*)&pp, NULL);
            g_cuda.stream_synchronize(0);
        }
        double t1 = now_ms();
        printf("  %-12s (%dx%d,%dx%d) GPU: %8.2f ms\n", name, H, W, KH, KW, (t1 - t0) / REPEAT);
    }

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
    free(h_in); free(h_out);
}

int main(void) {
    platform_init();
    cuda_platform_init(0);
    operator_init_all();

    printf("=== Operator Benchmarks (CPU vs CUDA) ===\n");
    printf("Warmup: %d iterations, Repeat: %d iterations\n\n", WARMUP, REPEAT);

    /* MatMul */
    printf("-- MatMul --\n");
    bench_matmul(128, 128, 128);
    bench_matmul(256, 256, 256);
    bench_matmul(512, 512, 512);
    bench_matmul(1024, 1024, 1024);
    printf("\n");

    /* Conv2D */
    printf("-- Conv2D --\n");
    bench_conv2d(32, 32, 3, 16, 3, 3);
    bench_conv2d(64, 64, 3, 32, 3, 3);
    bench_conv2d(128, 128, 3, 16, 5, 5);
    printf("\n");

    /* Activations */
    printf("-- Activations --\n");
    size_t n_act = 1024 * 1024;
    bench_activation("ReLU",    "relu_f32",    "relu_f32_cuda",    n_act);
    bench_activation("Sigmoid", "sigmoid_f32", "sigmoid_f32_cuda", n_act);
    bench_activation("GELU",    "gelu_f32",    "gelu_f32_cuda",    n_act);
    printf("\n");

    /* Pooling */
    printf("-- Pooling --\n");
    bench_pool("MaxPool2D",  "maxpool2d_f32",  "maxpool2d_f32_cuda",  128, 128, 2, 2, 2);
    bench_pool("AvgPool2D",  "avgpool2d_f32",  "avgpool2d_f32_cuda",  128, 128, 2, 2, 2);
    bench_pool("MaxPool2D",  "maxpool2d_f32",  "maxpool2d_f32_cuda",  256, 256, 3, 3, 2);
    printf("\n");

    /* BatchNorm */
    printf("-- BatchNorm --\n");
    {
        int64_t C = 64, hw = 256;
        size_t n = C * hw;
        float *x = (float*)malloc(n * sizeof(float));
        float *g = (float*)malloc(C * sizeof(float));
        float *b = (float*)malloc(C * sizeof(float));
        float *m = (float*)malloc(C * sizeof(float));
        float *v = (float*)malloc(C * sizeof(float));
        float *o = (float*)malloc(n * sizeof(float));
        for (size_t i = 0; i < n; i++) x[i] = (float)(i % 7 - 3);
        for (int64_t i = 0; i < C; i++) { g[i] = 1; b[i] = 0; m[i] = 0; v[i] = 1; }

        float *dx,*dg,*db,*dm,*dv,*dd;
        dx = (float*)g_cuda.device_alloc(n * sizeof(float));
        dg = (float*)g_cuda.device_alloc(C * sizeof(float));
        db = (float*)g_cuda.device_alloc(C * sizeof(float));
        dm = (float*)g_cuda.device_alloc(C * sizeof(float));
        dv = (float*)g_cuda.device_alloc(C * sizeof(float));
        dd = (float*)g_cuda.device_alloc(n * sizeof(float));
        g_cuda.memcpy_h2d(dx, x, n * sizeof(float), 0);
        g_cuda.memcpy_h2d(dg, g, C * sizeof(float), 0);
        g_cuda.memcpy_h2d(db, b, C * sizeof(float), 0);
        g_cuda.memcpy_h2d(dm, m, C * sizeof(float), 0);
        g_cuda.memcpy_h2d(dv, v, C * sizeof(float), 0);

        batchnorm_params_t bp = {.C = C, .epsilon = 1e-5f};

        /* CPU */
        const operator_registry_t* cpu_op = operator_find("batchnorm_f32");
        {
            const void* inputs[]  = {x, g, b, m, v, &hw};
            void*       outputs[] = {o};
            for (int w = 0; w < WARMUP; w++)
                cpu_op->func(inputs, outputs, (const operator_params_t*)&bp, NULL);
            double t0 = now_ms();
            for (int r = 0; r < REPEAT; r++)
                cpu_op->func(inputs, outputs, (const operator_params_t*)&bp, NULL);
            double t1 = now_ms();
            printf("  batchnorm   (c=%lld,n=%lld) CPU: %8.2f ms\n", (long long)C, (long long)n, (t1 - t0) / REPEAT);
        }

        /* CUDA */
        const operator_registry_t* cuda_op = operator_find("batchnorm_f32_cuda");
        if (cuda_op) {
            const void* inputs[]  = {dx, dg, db, dm, dv, &hw};
            void*       outputs[] = {dd};
            for (int w = 0; w < WARMUP; w++) {
                cuda_op->func(inputs, outputs, (const operator_params_t*)&bp, NULL);
                g_cuda.stream_synchronize(0);
            }
            double t0 = now_ms();
            for (int r = 0; r < REPEAT; r++) {
                cuda_op->func(inputs, outputs, (const operator_params_t*)&bp, NULL);
                g_cuda.stream_synchronize(0);
            }
            double t1 = now_ms();
            printf("  batchnorm   (c=%lld,n=%lld) GPU: %8.2f ms\n", (long long)C, (long long)n, (t1 - t0) / REPEAT);
        }

        g_cuda.device_free(dx); g_cuda.device_free(dg); g_cuda.device_free(db); g_cuda.device_free(dm); g_cuda.device_free(dv); g_cuda.device_free(dd);
        free(x); free(g); free(b); free(m); free(v); free(o);
    }

    printf("\n=== Done ===\n");
    cuda_platform_finalize();
    platform_finalize();
    return 0;
}
