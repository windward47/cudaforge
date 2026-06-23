/**
 * @file quantize_cuda.cu
 * @brief INT8 block quantization — CUDA implementation.
 */
#include "operator.h"
#include "cuda_ops.h"
#include "quantize_int.h"
#include <math.h>

/* ============================================================
 * Kernel 1: FP32 → INT8 block quantization
 * One block per Q8 block (64 elements). 64 threads.
 * Uses shared memory for reduction, broadcast via smem.
 * ============================================================ */
__global__ void quantize_f32_q8_kernel(const float* __restrict__ src,
                                        block_q8_t* __restrict__ dst,
                                        int64_t n) {
    int b = blockIdx.x;
    int64_t offset = (int64_t)b * Q8_BLOCK_SIZE;
    if (offset >= n) return;

    int tid = threadIdx.x;  /* 0..63 */
    int64_t count = n - offset;
    if (count > Q8_BLOCK_SIZE) count = Q8_BLOCK_SIZE;

    __shared__ float sdata[Q8_BLOCK_SIZE];

    /* Step 1: Each thread finds abs value of its element */
    float val = 0.0f;
    if (tid < count) {
        float v = src[offset + tid];
        val = v < 0.0f ? -v : v;
    }
    sdata[tid] = val;
    __syncthreads();

    /* Step 2: Parallel reduction for max */
    for (int s = Q8_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }

    /* Step 3: Compute scale (thread 0) and broadcast via smem */
    if (tid == 0)
        sdata[0] = (sdata[0] > 1e-12f) ? (sdata[0] / 127.0f) : 1e-12f;
    __syncthreads();

    float inv_scale = 1.0f / sdata[0];

    /* Step 4: Write scale to global memory */
    if (tid == 0)
        dst[b].scale = sdata[0];

    /* Step 5: Quantize — one element per thread */
    if (tid < count) {
        float v = src[offset + tid] * inv_scale;
        int q;
        if (v >= 0.0f) q = (int)(v + 0.5f);
        else           q = (int)(v - 0.5f);
        if (q > 127)  q = 127;
        if (q < -128) q = -128;
        dst[b].values[tid] = (int8_t)q;
    } else if (tid < Q8_BLOCK_SIZE) {
        dst[b].values[tid] = 0;
    }
}

/* ============================================================
 * Kernel 2: INT8 block → FP32 dequantization
 * ============================================================ */
__global__ void dequantize_q8_f32_kernel(const block_q8_t* __restrict__ src,
                                          float* __restrict__ dst,
                                          int64_t n) {
    int b = blockIdx.x;
    int64_t offset = (int64_t)b * Q8_BLOCK_SIZE;
    if (offset >= n) return;

    int tid = threadIdx.x;
    int64_t count = n - offset;
    if (count > Q8_BLOCK_SIZE) count = Q8_BLOCK_SIZE;

    float scale = src[b].scale;
    if (tid < count)
        dst[offset + tid] = (float)src[b].values[tid] * scale;
}

/* ============================================================
 * Kernel 3: INT8 weight × FP32 activation → FP32 output
 * ============================================================ */
__global__ void matmul_q8_f32_kernel(const block_q8_t* __restrict__ W_q8,
                                      const float* __restrict__ X,
                                      float* __restrict__ out,
                                      int M, int K, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int num_blocks = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;

    for (int col = tid; col < N; col += blockDim.x) {
        float acc = 0.0f;
        const block_q8_t* w_row = W_q8 + (int64_t)row * num_blocks;

        for (int b = 0; b < num_blocks; b++) {
            float scale = w_row[b].scale;
            int k_base = b * Q8_BLOCK_SIZE;
            int k_count = K - k_base;
            if (k_count > Q8_BLOCK_SIZE) k_count = Q8_BLOCK_SIZE;

            float block_acc = 0.0f;
            for (int ki = 0; ki < k_count; ki++)
                block_acc += (float)w_row[b].values[ki] * scale * X[(k_base + ki) * N + col];
            acc += block_acc;
        }
        out[row * N + col] = acc;
    }
}

/* ============================================================
 * Host entry points
 * ============================================================ */
extern "C" int quantize_f32_q8_cuda(const float* src, block_q8_t* dst, int64_t n, stream_t* stream) {
    if (!src || !dst || n <= 0) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t num_blocks = Q8_NUM_BLOCKS(n);
    dim3 grid((unsigned int)num_blocks);
    dim3 block(Q8_BLOCK_SIZE);
    return CUDA_KERNEL_LAUNCH(quantize_f32_q8_kernel, grid, block, 0, s, src, dst, n);
}

extern "C" int dequantize_q8_f32_cuda(const block_q8_t* src, float* dst, int64_t n, stream_t* stream) {
    if (!src || !dst || n <= 0) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t num_blocks = Q8_NUM_BLOCKS(n);
    dim3 grid((unsigned int)num_blocks);
    dim3 block(Q8_BLOCK_SIZE);
    return CUDA_KERNEL_LAUNCH(dequantize_q8_f32_kernel, grid, block, 0, s, src, dst, n);
}

extern "C" int matmul_q8_f32_cuda(const block_q8_t* W_q8, const float* X, float* out,
                                    int M, int K, int N, stream_t* stream) {
    if (!W_q8 || !X || !out || M <= 0 || K <= 0 || N <= 0) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    dim3 grid((unsigned int)M);
    dim3 block(256);
    return CUDA_KERNEL_LAUNCH(matmul_q8_f32_kernel, grid, block, 0, s, W_q8, X, out, M, K, N);
}
