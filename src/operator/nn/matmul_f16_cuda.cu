/* FP16 MatMul CUDA kernel.
   Reads FP16 inputs, accumulates in FP32, writes FP16 output.
   Registered as "matmul_f16_cuda". */
#include "operator.h"
#include "cuda_ops.h"
#include "matmul_int.h"
#include <cuda_fp16.h>

__global__ void matmul_f16_kernel(
    const __half* __restrict__ A,  /* (M, K) */
    const __half* __restrict__ B,  /* (K, N) */
    __half* __restrict__ C,        /* (M, N) */
    int64_t M, int64_t N, int64_t K)
{
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int64_t k = 0; k < K; k++) {
        acc += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
    }
    C[row * N + col] = __float2half(acc);
}

int matmul_f16_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;

    const matmul_params_t* p = (const matmul_params_t*)params;
    const __half* A = (const __half*)inputs[0];
    const __half* B = (const __half*)inputs[1];
    __half* C = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t M = p->M, K = p->K, N = p->N;

    dim3 block(16, 16, 1);
    dim3 grid((unsigned int)((N + 15) / 16), (unsigned int)((M + 15) / 16), 1);

    return CUDA_KERNEL_LAUNCH(matmul_f16_kernel, grid, block, 0, s, A, B, C, M, N, K);
}

extern "C" int register_matmul_f16_cuda(void) {
    static operator_registry_t reg = {
        .name = "matmul_f16_cuda", .data_type = "f16",
        .func = matmul_f16_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
