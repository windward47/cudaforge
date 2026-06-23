/**
 * INT8 block quantization test.
 *
 * Tests:
 *   1. CPU quantize/dequantize roundtrip accuracy
 *   2. CUDA quantize/dequantize correctness
 *   3. INT8 MatMul vs FP32 MatMul accuracy
 */
#include "platform.h"
#include "operator.h"
#include "quantize_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
extern "C" int quantize_f32_q8_cuda(const float* src, block_q8_t* dst, int64_t n, stream_t* stream);
extern "C" int dequantize_q8_f32_cuda(const block_q8_t* src, float* dst, int64_t n, stream_t* stream);
extern "C" int matmul_q8_f32_cuda(const block_q8_t* W_q8, const float* X, float* out,
                               int M, int K, int N, stream_t* stream);
#endif

extern "C" {
extern int operator_init_all(void);
extern void quantize_f32_to_q8(const float* src, block_q8_t* dst, int64_t n);
extern void dequantize_q8_to_f32(const block_q8_t* src, float* dst, int64_t n);
}

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

static void random_fill(float* data, int64_t n, unsigned seed) {
    unsigned state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = ((float)(state & 0x7FFFFFFF) / 2147483648.0f) - 1.0f;
    }
}

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

/* ---- Test 1: CPU roundtrip ---- */
static void test_quantize_cpu(void) {
    fprintf(stderr, "\n=== INT8 Quantize CPU Test ===\n");
    int64_t n = 1024;
    float* src = (float*)malloc(n * sizeof(float));
    float* dst = (float*)malloc(n * sizeof(float));
    block_q8_t* q = (block_q8_t*)malloc(Q8_PACKED_BYTES(n));

    random_fill(src, n, 42);
    quantize_f32_to_q8(src, q, n);
    dequantize_q8_to_f32(q, dst, n);

    float maxd = max_abs_diff(src, dst, n);
    fprintf(stderr, "Roundtrip max_diff=%.2e (n=%lld, %lld blocks)\n",
            maxd, (long long)n, (long long)Q8_NUM_BLOCKS(n));
    /* INT8 quantization has ~0.5% relative error for uniform data */
    CHECK(maxd < 0.02f, "CPU roundtrip accuracy");

    free(src); free(dst); free(q);
    fprintf(stderr, "INT8 Quantize CPU: PASS\n");
}

#ifdef USE_CUDA
/* ---- Test 2: CUDA quantize/dequantize ---- */
static void test_quantize_cuda(void) {
    fprintf(stderr, "\n=== INT8 Quantize CUDA Test ===\n");
    int64_t n = 4096;
    float* src = (float*)malloc(n * sizeof(float));
    float* ref = (float*)malloc(n * sizeof(float));
    float* gpu_out = (float*)malloc(n * sizeof(float));
    block_q8_t* q_ref = (block_q8_t*)malloc(Q8_PACKED_BYTES(n));

    random_fill(src, n, 123);

    /* CPU reference */
    quantize_f32_to_q8(src, q_ref, n);
    dequantize_q8_to_f32(q_ref, ref, n);

    /* GPU: upload src, quantize, dequantize, download */
    float* d_src = NULL;
    block_q8_t* d_q = NULL;
    float* d_out = NULL;
    cudaMalloc(&d_src, n * sizeof(float));
    cudaMalloc(&d_q, Q8_PACKED_BYTES(n));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_src, src, n * sizeof(float), cudaMemcpyHostToDevice);

    quantize_f32_q8_cuda(d_src, d_q, n, NULL);
    dequantize_q8_f32_cuda(d_q, d_out, n, NULL);
    cudaMemcpy(gpu_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    /* Compare GPU dequant vs CPU dequant */
    float maxd = max_abs_diff(ref, gpu_out, n);
    fprintf(stderr, "CUDA vs CPU dequant max_diff=%.2e\n", maxd);
    CHECK(maxd < 1e-5f, "CUDA quantize/dequantize correctness");

    cudaFree(d_src); cudaFree(d_q); cudaFree(d_out);
    free(src); free(ref); free(gpu_out); free(q_ref);
    fprintf(stderr, "INT8 Quantize CUDA: PASS\n");
}

/* ---- Test 3: INT8 MatMul vs FP32 ---- */
static void test_matmul_q8(void) {
    fprintf(stderr, "\n=== INT8 MatMul Test ===\n");
    int M = 64, K = 768, N = 128;

    float* W = (float*)malloc(M * K * sizeof(float));
    float* X = (float*)malloc(K * N * sizeof(float));
    float* out_fp32 = (float*)malloc(M * N * sizeof(float));
    float* out_q8 = (float*)malloc(M * N * sizeof(float));

    random_fill(W, M * K, 456);
    random_fill(X, K * N, 789);

    /* FP32 reference matmul */
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < N; c++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += W[r * K + k] * X[k * N + c];
            out_fp32[r * N + c] = acc;
        }
    }

    /* Quantize W to INT8 */
    block_q8_t* W_q8 = (block_q8_t*)malloc(Q8_PACKED_BYTES(M * K));
    quantize_f32_to_q8(W, W_q8, M * K);

    /* GPU INT8 matmul */
    block_q8_t* d_W = NULL;
    float* d_X = NULL;
    float* d_out = NULL;
    cudaMalloc(&d_W, Q8_PACKED_BYTES(M * K));
    cudaMalloc(&d_X, K * N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));
    cudaMemcpy(d_W, W_q8, Q8_PACKED_BYTES(M * K), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, K * N * sizeof(float), cudaMemcpyHostToDevice);

    matmul_q8_f32_cuda(d_W, d_X, d_out, M, K, N, NULL);
    cudaMemcpy(out_q8, d_out, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxd = max_abs_diff(out_fp32, out_q8, M * N);
    /* Relative error */
    float max_val = 0.0f;
    for (int i = 0; i < M * N; i++)
        if (fabsf(out_fp32[i]) > max_val) max_val = fabsf(out_fp32[i]);
    float rel_err = max_val > 1e-6f ? maxd / max_val : maxd;
    fprintf(stderr, "INT8 vs FP32 MatMul: max_abs=%.2e, max_rel=%.2e\n", maxd, rel_err);
    CHECK(rel_err < 0.05f, "INT8 MatMul accuracy (5%% relative)");

    cudaFree(d_W); cudaFree(d_X); cudaFree(d_out);
    free(W); free(X); free(out_fp32); free(out_q8); free(W_q8);
    fprintf(stderr, "INT8 MatMul: PASS\n");
}
#endif

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_quantize_cpu();

#ifdef USE_CUDA
    test_quantize_cuda();
    test_matmul_q8();
    cuda_platform_finalize();
#endif

    platform_finalize();
    fprintf(stderr, "\nAll quantize tests done.\n");
    return 0;
}
