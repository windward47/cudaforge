#include "operator.h"
#include "cuda_ops.h"
#include "matmul_int.h"
#include <mma.h>

/* ============================================================
 * Transpose kernel — used when transpose_a / transpose_b is set.
 * dst[M][N] = src[N][M]  (row-major)
 * ============================================================ */
__global__ void transpose_f32_kernel(const float* src, float* dst,
                                     int64_t src_rows, int64_t src_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = src_rows * src_cols;
    if (idx >= total) return;
    int64_t r = idx / src_cols;
    int64_t c = idx % src_cols;
    dst[c * src_rows + r] = src[idx];
}

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
 * Tensor Core kernel (sm_70+).
 *
 * Uses nvcuda::wmma with m16n16k16 FP16→FP32 MMA operations.
 * Each block (1 warp, 32 threads) computes a 16×16 output tile.
 * FP32 inputs are converted to FP16 on load for Tensor Core compute;
 * accumulation and output remain FP32 for full precision.
 * ============================================================ */
#define TC_TILE 16

__global__ void matmul_f32_tc_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int64_t M, int64_t N, int64_t K) {
    using namespace nvcuda;

    __shared__ half As[TC_TILE][TC_TILE];
    __shared__ half Bs[TC_TILE][TC_TILE];

    /* Warp-local: this block is a single warp (32 threads) */
    int tid = threadIdx.x;

    /* Compute the 16×16 output tile this warp owns */
    int tile_row = blockIdx.y * TC_TILE;
    int tile_col = blockIdx.x * TC_TILE;

    wmma::fragment<wmma::matrix_a, TC_TILE, TC_TILE, TC_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TC_TILE, TC_TILE, TC_TILE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TC_TILE, TC_TILE, TC_TILE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int num_k_tiles = (int)((K + TC_TILE - 1) / TC_TILE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k0 = kt * TC_TILE;

        /* Cooperative load A tile: TC_TILE×TC_TILE = 256 halfs, 8 per thread */
        const float* A_base = A + tile_row * K + k0;
        #pragma unroll
        for (int i = tid; i < TC_TILE * TC_TILE; i += 32) {
            int r = i / TC_TILE;
            int c = i % TC_TILE;
            float val = 0.0f;
            if (tile_row + r < M && k0 + c < K)
                val = A_base[r * K + c];
            As[r][c] = __float2half(val);
        }

        /* Cooperative load B tile: TC_TILE×TC_TILE = 256 halfs, 8 per thread */
        const float* B_base = B + k0 * N + tile_col;
        #pragma unroll
        for (int i = tid; i < TC_TILE * TC_TILE; i += 32) {
            int r = i / TC_TILE;
            int c = i % TC_TILE;
            float val = 0.0f;
            if (k0 + r < K && tile_col + c < N)
                val = B_base[r * N + c];
            Bs[r][c] = __float2half(val);
        }

        __syncthreads();

        wmma::load_matrix_sync(a_frag, (const half*)As, TC_TILE);
        wmma::load_matrix_sync(b_frag, (const half*)Bs, TC_TILE);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    /* Store.  wmma::store_matrix_sync writes a full 16×16 tile —
     * the dispatch below only routes to this kernel when M and N are
     * multiples of TC_TILE, so no out-of-bounds store is possible. */
    float* C_out = C + tile_row * N + tile_col;
    wmma::store_matrix_sync(C_out, c_frag, (unsigned)N, wmma::mem_row_major);
}

/* ============================================================
 * Dispatch: selects the best kernel for the given dimensions.
 *
 * Strategy (sm_86 / RTX 2050):
 *   - Tiny matrices (max dim <= 32): naive kernel (less launch overhead)
 *   - Small/medium (M,N < 64):    16x16 tiled shared-memory kernel
 *   - Medium (M,N 64..511):       32x32 warp-tiled kernel with padding
 *   - Large (M,N >= 512):         FP16 Tensor Core kernel (MMA m16n16k16)
 *
 * The 32×32 warp kernel uses +1 padding on shared memory columns to
 * avoid bank conflicts between consecutive rows.
 * ============================================================ */
/* Bias addition kernel: C[i*N + j] += bias[j] for all rows i */
__global__ void matmul_bias_add_kernel(float* C, const float* bias,
                                        int64_t M, int64_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx >= total) return;
    int j = (int)(idx % N);
    C[idx] += bias[j];
}

int matmul_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const matmul_params_t* p = (const matmul_params_t*)params;
    const float* A    = (const float*)inputs[0];
    const float* B    = (const float*)inputs[1];
    const float* bias = (const float*)inputs[2];
    float* C          = (float*)outputs[0];

    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t M = p->M, N = p->N, K = p->K;
    float* A_trans = NULL;
    float* B_trans = NULL;
    dim3 block(1, 1, 1), grid(1, 1, 1);
    int ret = 0;

    /* Transpose A if needed (A is K×M, need M×K for standard kernel) */
    if (p->transpose_a) {
        size_t bytes = (size_t)M * K * sizeof(float);
        A_trans = (float*)g_cuda.device_alloc(bytes);
        if (!A_trans) { ret = -1; goto matmul_cleanup; }
        int64_t total = M * K;
        dim3 tb(256, 1, 1);
        dim3 tg((unsigned int)((total + 255) / 256), 1, 1);
        CUDA_KERNEL_LAUNCH(transpose_f32_kernel, tg, tb, 0, s,
                           A, A_trans, K, M);
        A = A_trans;
    }

    /* Transpose B if needed (B is N×K, need K×N for standard kernel) */
    if (p->transpose_b) {
        size_t bytes = (size_t)K * N * sizeof(float);
        B_trans = (float*)g_cuda.device_alloc(bytes);
        if (!B_trans) { ret = -1; goto matmul_cleanup; }
        int64_t total = K * N;
        dim3 tb(256, 1, 1);
        dim3 tg((unsigned int)((total + 255) / 256), 1, 1);
        CUDA_KERNEL_LAUNCH(transpose_f32_kernel, tg, tb, 0, s,
                           B, B_trans, N, K);
        B = B_trans;
    }

    if (M <= 32 && N <= 32) {
        block = dim3(16, 16, 1);
        grid  = dim3((unsigned int)((N + 15) / 16),
                     (unsigned int)((M + 15) / 16), 1);
        CUDA_KERNEL_LAUNCH(matmul_f32_naive, grid, block, 0, s,
                           A, B, C, M, N, K);
    } else if (M >= 512 && N >= 512 && K >= 512
               && M % TC_TILE == 0 && N % TC_TILE == 0) {
        block = dim3(32, 1, 1);
        grid  = dim3((unsigned int)((N + TC_TILE - 1) / TC_TILE),
                     (unsigned int)((M + TC_TILE - 1) / TC_TILE), 1);
        CUDA_KERNEL_LAUNCH(matmul_f32_tc_kernel, grid, block, 0, s,
                           A, B, C, M, N, K);
    } else if (M >= 64 && N >= 64) {
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

    /* Add bias if present */
    if (bias) {
        int64_t total = M * N;
        int block_sz = 256;
        int grid_sz = (int)((total + block_sz - 1) / block_sz);
        CUDA_KERNEL_LAUNCH(matmul_bias_add_kernel, grid_sz, block_sz, 0, s,
                           C, bias, M, N);
    }

matmul_cleanup:
    if (A_trans) g_cuda.device_free(A_trans);
    if (B_trans) g_cuda.device_free(B_trans);
    return ret;
}

extern "C" int register_matmul_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "matmul_f32_cuda", .data_type = "f32",
        .func = matmul_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
