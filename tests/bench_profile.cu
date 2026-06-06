/* ============================================================
 * bench_profile.cu — 算子级 CUDA 性能 Profile
 *
 * 使用 CUDA event 精确计时，输出 CSV 格式结果用于回归检测。
 * 用法：
 *   ./bench_profile.exe                    # 输出到 stdout
 *   ./bench_profile.exe > profile.csv      # 保存到文件
 *   ./bench_profile.exe --json             # JSON 格式输出
 *
 * 输出格式（CSV）：
 *   op_name,shape,latency_ms
 *
 * 参考 CUDAForge 的 NCU 硬件指标驱动优化思路。
 * ============================================================ */
extern "C" {
#include "platform.h"
int operator_init_all(void);
}

#include "operator.h"
#include "matmul_int.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "batchnorm_int.h"
#include "layernorm_int.h"
#include "softmax_int.h"
#include "reduce_int.h"
#include "add_int.h"
#include "cuda_ops.h"
#include "cuda_platform.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- 配置 ---- */
#define WARMUP_ITERS  5
#define BENCH_ITERS  20

/* ---- CUDA event 计时辅助 ---- */
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} bench_timer_t;

static void timer_create(bench_timer_t* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static void timer_destroy(bench_timer_t* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static void timer_start(bench_timer_t* t, cudaStream_t stream) {
    cudaEventRecord(t->start, stream);
}

static float timer_stop(bench_timer_t* t, cudaStream_t stream) {
    cudaEventRecord(t->stop, stream);
    cudaEventSynchronize(t->stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

/* ---- 输出格式 ---- */
static int g_use_json = 0;
static int g_first_output = 1;

static void print_header(void) {
    if (g_use_json) {
        printf("[\n");
    } else {
        printf("op_name,shape,latency_ms\n");
    }
}

static void print_result(const char* op, const char* shape, float ms) {
    if (g_use_json) {
        if (!g_first_output) printf(",\n");
        printf("  {\"op\":\"%s\",\"shape\":\"%s\",\"latency_ms\":%.4f}", op, shape, ms);
        g_first_output = 0;
    } else {
        printf("%s,%s,%.4f\n", op, shape, ms);
    }
}

static void print_footer(void) {
    if (g_use_json) printf("\n]\n");
}

/* ============================================================
 * Benchmark helpers
 * ============================================================ */

/* 通用 CUDA 算子 benchmark：返回平均 latency (ms) */
static float bench_cuda_op(const operator_registry_t* op,
                           const void* inputs[], void* outputs[],
                           const operator_params_t* params,
                           int warmup, int repeat) {
    if (!op) return -1.0f;
    cudaStream_t stream = 0;
    bench_timer_t timer;
    timer_create(&timer);

    for (int i = 0; i < warmup; i++) {
        op->func(inputs, outputs, params, NULL);
        g_cuda.stream_synchronize(stream);
    }

    float total = 0;
    for (int i = 0; i < repeat; i++) {
        timer_start(&timer, stream);
        op->func(inputs, outputs, params, NULL);
        total += timer_stop(&timer, stream);
    }

    timer_destroy(&timer);
    return total / repeat;
}

/* ============================================================
 * MatMul profile
 * ============================================================ */
static void profile_matmul(int M, int N, int K) {
    size_t sa = (size_t)M * K, sb = (size_t)K * N, sc = (size_t)M * N;
    float *dA = (float*)g_cuda.device_alloc(sa * sizeof(float));
    float *dB = (float*)g_cuda.device_alloc(sb * sizeof(float));
    float *dC = (float*)g_cuda.device_alloc(sc * sizeof(float));

    matmul_params_t p = {.M = M, .N = N, .K = K};
    const void* inputs[]  = {dA, dB, NULL};
    void*       outputs[] = {dC};

    const operator_registry_t* op = operator_find("matmul_f32_cuda");
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&p, WARMUP_ITERS, BENCH_ITERS);

    char shape[64];
    snprintf(shape, sizeof(shape), "%dx%dx%d", M, N, K);
    print_result("matmul_f32", shape, ms);

    g_cuda.device_free(dA); g_cuda.device_free(dB); g_cuda.device_free(dC);
}

/* ============================================================
 * Conv2D profile
 * ============================================================ */
static void profile_conv2d(int H, int W, int C_in, int K, int KH, int KW) {
    int64_t OH = H - KH + 1, OW = W - KW + 1;
    size_t s_in = (size_t)C_in * H * W;
    size_t s_w  = (size_t)K * C_in * KH * KW;
    size_t s_out = (size_t)K * OH * OW;

    float *d_in  = (float*)g_cuda.device_alloc(s_in  * sizeof(float));
    float *d_w   = (float*)g_cuda.device_alloc(s_w   * sizeof(float));
    float *d_out = (float*)g_cuda.device_alloc(s_out * sizeof(float));

    conv_params_t cp;
    memset(&cp, 0, sizeof(cp));
    cp.N = 1; cp.C = C_in; cp.H = H; cp.W = W; cp.K = K;
    cp.kernel_h = KH; cp.kernel_w = KW;
    cp.stride_h = 1; cp.stride_w = 1;
    cp.dilation_h = 1; cp.dilation_w = 1;

    const void* inputs[]  = {d_in, d_w, NULL};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find("conv2d_f32_cuda");
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&cp, WARMUP_ITERS, BENCH_ITERS);

    char shape[64];
    snprintf(shape, sizeof(shape), "%dx%dx%dx%dx%dx%d", 1, C_in, H, W, K, KH);
    print_result("conv2d_f32", shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_w); g_cuda.device_free(d_out);
}

/* ============================================================
 * Element-wise activation profile
 * ============================================================ */
static void profile_activation(const char* name, const char* op_name, int64_t n) {
    float *d_in  = (float*)g_cuda.device_alloc(n * sizeof(float));
    float *d_out = (float*)g_cuda.device_alloc(n * sizeof(float));

    const void* inputs[]  = {d_in, &n};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find(op_name);
    float ms = bench_cuda_op(op, inputs, outputs, NULL, WARMUP_ITERS, BENCH_ITERS);

    char shape[32];
    snprintf(shape, sizeof(shape), "%lld", (long long)n);
    print_result(name, shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
}

/* ============================================================
 * Softmax profile
 * ============================================================ */
static void profile_softmax(int rows, int cols) {
    size_t n = (size_t)rows * cols;
    float *d_in  = (float*)g_cuda.device_alloc(n * sizeof(float));
    float *d_out = (float*)g_cuda.device_alloc(n * sizeof(float));

    softmax_params_t sp = {.num_classes = cols, .num_blocks = rows};
    const void* inputs[]  = {d_in};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find("softmax_f32_cuda");
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&sp, WARMUP_ITERS, BENCH_ITERS);

    char shape[32];
    snprintf(shape, sizeof(shape), "%dx%d", rows, cols);
    print_result("softmax_f32", shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
}

/* ============================================================
 * LayerNorm profile
 * ============================================================ */
static void profile_layernorm(int rows, int cols) {
    size_t n = (size_t)rows * cols;
    float *d_in    = (float*)g_cuda.device_alloc(n * sizeof(float));
    float *d_out   = (float*)g_cuda.device_alloc(n * sizeof(float));
    float *d_gamma = (float*)g_cuda.device_alloc(cols * sizeof(float));
    float *d_beta  = (float*)g_cuda.device_alloc(cols * sizeof(float));

    layernorm_params_t lp;
    lp.N = rows;
    lp.normalized_size = cols;
    lp.epsilon = 1e-5f;

    const void* inputs[]  = {d_in, d_gamma, d_beta};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find("layernorm_f32_cuda");
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&lp, WARMUP_ITERS, BENCH_ITERS);

    char shape[32];
    snprintf(shape, sizeof(shape), "%dx%d", rows, cols);
    print_result("layernorm_f32", shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
    g_cuda.device_free(d_gamma); g_cuda.device_free(d_beta);
}

/* ============================================================
 * Reduce profile
 * ============================================================ */
static void profile_reduce(const char* name, const char* op_name, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    float *d_in  = (float*)g_cuda.device_alloc(n * sizeof(float));
    float *d_out = (float*)g_cuda.device_alloc(rows * sizeof(float));

    reduce_params_t rp;
    rp.reduce_size = cols;
    rp.num_blocks = rows;
    rp.total_elems = n;
    rp.op = (strcmp(name, "reduce_max_f32") == 0) ? 1 : 0;

    const void* inputs[]  = {d_in};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find(op_name);
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&rp, WARMUP_ITERS, BENCH_ITERS);

    char shape[32];
    snprintf(shape, sizeof(shape), "%dx%d", rows, cols);
    print_result(name, shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
}

/* ============================================================
 * Pooling profile
 * ============================================================ */
static void profile_pooling(const char* name, const char* op_name,
                            int H, int W, int KH, int KW, int stride) {
    int64_t OH = (H - KH) / stride + 1, OW = (W - KW) / stride + 1;
    size_t s_in = (size_t)H * W, s_out = (size_t)OH * OW;

    float *d_in  = (float*)g_cuda.device_alloc(s_in  * sizeof(float));
    float *d_out = (float*)g_cuda.device_alloc(s_out * sizeof(float));

    pool_params_t pp;
    memset(&pp, 0, sizeof(pp));
    pp.N = 1; pp.C = 1; pp.H = H; pp.W = W;
    pp.kernel_h = KH; pp.kernel_w = KW;
    pp.stride_h = stride; pp.stride_w = stride;

    const void* inputs[]  = {d_in};
    void*       outputs[] = {d_out};

    const operator_registry_t* op = operator_find(op_name);
    float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&pp, WARMUP_ITERS, BENCH_ITERS);

    char shape[32];
    snprintf(shape, sizeof(shape), "%dx%dx%dx%d", H, W, KH, KW);
    print_result(name, shape, ms);

    g_cuda.device_free(d_in); g_cuda.device_free(d_out);
}

/* ============================================================
 * main
 * ============================================================ */
int main(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) g_use_json = 1;
    }

    platform_init();
    cuda_platform_init(0);
    operator_init_all();

    print_header();

    /* --- MatMul --- */
    profile_matmul(128, 128, 128);
    profile_matmul(256, 256, 256);
    profile_matmul(512, 512, 512);
    profile_matmul(1024, 1024, 1024);

    /* --- Conv2D --- */
    profile_conv2d(32, 32, 3, 16, 3, 3);
    profile_conv2d(64, 64, 3, 32, 3, 3);
    profile_conv2d(128, 128, 64, 64, 3, 3);
    profile_conv2d(56, 56, 64, 128, 3, 3);

    /* --- Activations (1M elements) --- */
    int64_t n_act = 1024 * 1024;
    profile_activation("relu_f32",    "relu_f32_cuda",    n_act);
    profile_activation("sigmoid_f32", "sigmoid_f32_cuda", n_act);
    profile_activation("gelu_f32",    "gelu_f32_cuda",    n_act);
    profile_activation("silu_f32",    "silu_f32_cuda",    n_act);
    /* --- Add (binary element-wise) --- */
    {
        int64_t n = n_act;
        float *dA = (float*)g_cuda.device_alloc(n * sizeof(float));
        float *dB = (float*)g_cuda.device_alloc(n * sizeof(float));
        float *dC = (float*)g_cuda.device_alloc(n * sizeof(float));
        add_params_t ap = {.numel = n, .B_numel = n};
        const void* inputs[]  = {dA, dB};
        void*       outputs[] = {dC};
        const operator_registry_t* op = operator_find("add_f32_cuda");
        float ms = bench_cuda_op(op, inputs, outputs, (const operator_params_t*)&ap, WARMUP_ITERS, BENCH_ITERS);
        char shape[32]; snprintf(shape, sizeof(shape), "%lld", (long long)n);
        print_result("add_f32", shape, ms);
        g_cuda.device_free(dA); g_cuda.device_free(dB); g_cuda.device_free(dC);
    }

    /* --- Softmax --- */
    profile_softmax(32, 512);
    profile_softmax(128, 1024);
    profile_softmax(256, 4096);

    /* --- LayerNorm --- */
    profile_layernorm(32, 768);
    profile_layernorm(128, 768);
    profile_layernorm(128, 3072);

    /* --- Reduce --- */
    profile_reduce("reduce_sum_f32", "reduce_f32_cuda", 128, 1024);
    profile_reduce("reduce_max_f32", "reduce_f32_cuda", 128, 1024);
    profile_reduce("reduce_sum_f32", "reduce_f32_cuda", 256, 4096);

    /* --- Pooling --- */
    profile_pooling("maxpool_f32", "maxpool2d_f32_cuda", 128, 128, 2, 2, 2);
    profile_pooling("avgpool_f32", "avgpool2d_f32_cuda", 128, 128, 2, 2, 2);
    profile_pooling("maxpool_f32", "maxpool2d_f32_cuda", 256, 256, 3, 3, 2);

    print_footer();

    cuda_platform_finalize();
    platform_finalize();
    return 0;
}
