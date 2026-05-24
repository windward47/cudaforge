#include "operator.h"
#include "cuda_ops.h"
#include "matmul_int.h"

/* ============================================================
 * Naive: each thread computes one output element
 * ============================================================ */
__global__ void matmul_f32_naive(const float* A, const float* B, float* C,
                                  int64_t M, int64_t N, int64_t K) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* ============================================================
 * Tiled: shared memory, MATMUL_TILE_SIZE (16) × MATMUL_TILE_SIZE
 * ============================================================ */
__global__ void matmul_f32_tiled(const float* A, const float* B, float* C,
                                  int64_t M, int64_t N, int64_t K) {
    __shared__ float As[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
    __shared__ float Bs[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * MATMUL_TILE_SIZE + ty;
    int col = blockIdx.x * MATMUL_TILE_SIZE + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE; t++) {
        int tiled_k = t * MATMUL_TILE_SIZE;

        if (row < M && tiled_k + tx < K)
            As[ty][tx] = A[row * K + tiled_k + tx];
        else
            As[ty][tx] = 0.0f;

        if (tiled_k + ty < K && col < N)
            Bs[ty][tx] = B[(tiled_k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < MATMUL_TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/* ============================================================
 * Warp-tiled: 32×32 tile with bank-conflict-avoiding padding.
 * Larger tile reduces global memory round-trips;
 * padding column (+1) avoids shared memory bank conflicts
 * between consecutive rows on the 32-bit bank architecture.
 * ============================================================ */
#define WARP_TILE 32

__global__ void matmul_f32_warp(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int64_t M, int64_t N, int64_t K) {
    __shared__ float As[WARP_TILE][WARP_TILE + 1];
    __shared__ float Bs[WARP_TILE][WARP_TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * WARP_TILE + ty;
    int col = blockIdx.x * WARP_TILE + tx;

    float sum = 0.0f;
    int num_tiles = (int)((K + WARP_TILE - 1) / WARP_TILE);

    for (int t = 0; t < num_tiles; t++) {
        int tk = t * WARP_TILE;

        if (row < M && tk + tx < K)
            As[ty][tx] = A[row * K + tk + tx];
        else
            As[ty][tx] = 0.0f;

        if (tk + ty < K && col < N)
            Bs[ty][tx] = B[(tk + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < WARP_TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

/* ============================================================
 * Dispatch: selects the best kernel for the given dimensions
 * ============================================================ */
int matmul_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const matmul_params_t* p = (const matmul_params_t*)params;
    const float* A = (const float*)inputs[0];
    const float* B = (const float*)inputs[1];
    float* C       = (float*)outputs[0];

    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t M = p->M, N = p->N, K = p->K;
    dim3 block, grid;

    if (M >= 64 && N >= 64) {
        block = dim3(WARP_TILE, WARP_TILE, 1);
        grid  = dim3((unsigned int)((N + WARP_TILE - 1) / WARP_TILE),
                     (unsigned int)((M + WARP_TILE - 1) / WARP_TILE), 1);
        CUDA_KERNEL_LAUNCH(matmul_f32_warp, grid, block, 0, s,
                           A, B, C, M, N, K);
    } else {
        block = dim3(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
        grid  = dim3((unsigned int)((N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE),
                     (unsigned int)((M + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE), 1);
        CUDA_KERNEL_LAUNCH(matmul_f32_tiled, grid, block, 0, s,
                           A, B, C, M, N, K);
    }
    return 0;
}

extern "C" int register_matmul_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "matmul_f32_cuda", .data_type = "f32",
        .func = matmul_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
