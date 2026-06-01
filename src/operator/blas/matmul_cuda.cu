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
 * Naive: each thread computes one output element (batched)
 * ============================================================ */
__global__ void matmul_f32_naive(const float* A, const float* B, float* C,
                                  int64_t M, int64_t N, int64_t K,
                                  int64_t batch_size,
                                  int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.z;
    if (row < M && col < N && batch < batch_size) {
        const float* Ab = A + batch * stride_a;
        const float* Bb = B + batch * stride_b;
        float*       Cb = C + batch * stride_c;
        float sum = 0.0f;
        for (int64_t k = 0; k < K; k++) {
            sum += Ab[row * K + k] * Bb[k * N + col];
        }
        Cb[row * N + col] = sum;
    }
}

/* ============================================================
 * Tiled: shared memory, MATMUL_TILE_SIZE (16) × MATMUL_TILE_SIZE (batched)
 * ============================================================ */
__global__ void matmul_f32_tiled(const float* A, const float* B, float* C,
                                  int64_t M, int64_t N, int64_t K,
                                  int64_t batch_size,
                                  int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    __shared__ float As[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];
    __shared__ float Bs[MATMUL_TILE_SIZE][MATMUL_TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * MATMUL_TILE_SIZE + ty;
    int col = blockIdx.x * MATMUL_TILE_SIZE + tx;
    int64_t batch = blockIdx.z;

    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float*       Cb = C + batch * stride_c;

    float sum = 0.0f;
    for (int t = 0; t < (K + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE; t++) {
        int tiled_k = t * MATMUL_TILE_SIZE;

        if (row < M && tiled_k + tx < K)
            As[ty][tx] = Ab[row * K + tiled_k + tx];
        else
            As[ty][tx] = 0.0f;

        if (tiled_k + ty < K && col < N)
            Bs[ty][tx] = Bb[(tiled_k + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < MATMUL_TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N && batch < batch_size) {
        Cb[row * N + col] = sum;
    }
}

/* ============================================================
 * Warp-tiled: 32×32 tile with bank-conflict-avoiding padding (batched)
 * ============================================================ */
#define WARP_TILE 32

__global__ void matmul_f32_warp(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int64_t M, int64_t N, int64_t K,
                                 int64_t batch_size,
                                 int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    __shared__ float As[WARP_TILE][WARP_TILE + 1];
    __shared__ float Bs[WARP_TILE][WARP_TILE + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * WARP_TILE + ty;
    int col = blockIdx.x * WARP_TILE + tx;
    int64_t batch = blockIdx.z;

    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float*       Cb = C + batch * stride_c;

    float sum = 0.0f;
    int num_tiles = (int)((K + WARP_TILE - 1) / WARP_TILE);

    for (int t = 0; t < num_tiles; t++) {
        int tk = t * WARP_TILE;

        if (row < M && tk + tx < K)
            As[ty][tx] = Ab[row * K + tk + tx];
        else
            As[ty][tx] = 0.0f;

        if (tk + ty < K && col < N)
            Bs[ty][tx] = Bb[(tk + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < WARP_TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N && batch < batch_size) {
        Cb[row * N + col] = sum;
    }
}

/* ============================================================
 * Tensor Core kernel (sm_70+, batched).
 * ============================================================ */
#define TC_TILE 16

__global__ void matmul_f32_tc_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int64_t M, int64_t N, int64_t K,
                                      int64_t batch_size,
                                      int64_t stride_a, int64_t stride_b, int64_t stride_c) {
    using namespace nvcuda;

    __shared__ half As[TC_TILE][TC_TILE];
    __shared__ half Bs[TC_TILE][TC_TILE];

    int tid = threadIdx.x;
    int64_t batch = blockIdx.z;

    const float* Ab = A + batch * stride_a;
    const float* Bb = B + batch * stride_b;
    float*       Cb = C + batch * stride_c;

    int tile_row = blockIdx.y * TC_TILE;
    int tile_col = blockIdx.x * TC_TILE;

    wmma::fragment<wmma::matrix_a, TC_TILE, TC_TILE, TC_TILE, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TC_TILE, TC_TILE, TC_TILE, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TC_TILE, TC_TILE, TC_TILE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int num_k_tiles = (int)((K + TC_TILE - 1) / TC_TILE);

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int k0 = kt * TC_TILE;

        const float* A_base = Ab + tile_row * K + k0;
        #pragma unroll
        for (int i = tid; i < TC_TILE * TC_TILE; i += 32) {
            int r = i / TC_TILE;
            int c = i % TC_TILE;
            float val = 0.0f;
            if (tile_row + r < M && k0 + c < K)
                val = A_base[r * K + c];
            As[r][c] = __float2half(val);
        }

        const float* B_base = Bb + k0 * N + tile_col;
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

    if (batch < batch_size) {
        float* C_out = Cb + tile_row * N + tile_col;
        wmma::store_matrix_sync(C_out, c_frag, (unsigned)N, wmma::mem_row_major);
    }
}

/* ============================================================
 * Bias addition kernel: C[b,i*N + j] += bias[j] for all batches/rows
 * ============================================================ */
__global__ void matmul_bias_add_kernel(float* C, const float* bias,
                                        int64_t M, int64_t N, int64_t batch_size,
                                        int64_t stride_c) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = batch_size * M * N;
    if (idx >= total) return;
    int64_t j = idx % N;
    C[idx] += bias[j];
}

/* ============================================================
 * Dispatch: selects the best kernel for the given dimensions.
 * ============================================================ */
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
    int64_t batch_size = p->batch_size > 0 ? p->batch_size : 1;
    int64_t stride_a = p->stride_a;
    int64_t stride_b = p->stride_b;
    int64_t stride_c = p->stride_c;
    float* A_trans = NULL;
    float* B_trans = NULL;
    int ret = 0;

    /* Transpose A if needed (A is K×M, need M×K for standard kernel) */
    if (p->transpose_a) {
        size_t bytes = (size_t)batch_size * M * K * sizeof(float);
        A_trans = (float*)g_cuda.device_alloc(bytes);
        if (!A_trans) { ret = -1; goto matmul_cleanup; }
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t total = M * K;
            dim3 tb(OPS_THREADS_PER_BLOCK, 1, 1);
            dim3 tg((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);
            ret = CUDA_KERNEL_LAUNCH(transpose_f32_kernel, tg, tb, 0, s,
                                      A + b * stride_a, A_trans + b * M * K, K, M);
            if (ret != 0) goto matmul_cleanup;
        }
        A = A_trans;
        stride_a = M * K;
    }

    /* Transpose B if needed (B is N×K, need K×N for standard kernel) */
    if (p->transpose_b) {
        size_t bytes = (size_t)batch_size * K * N * sizeof(float);
        B_trans = (float*)g_cuda.device_alloc(bytes);
        if (!B_trans) { ret = -1; goto matmul_cleanup; }
        for (int64_t b = 0; b < batch_size; b++) {
            int64_t total = K * N;
            dim3 tb(OPS_THREADS_PER_BLOCK, 1, 1);
            dim3 tg((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);
            ret = CUDA_KERNEL_LAUNCH(transpose_f32_kernel, tg, tb, 0, s,
                                      B + b * stride_b, B_trans + b * K * N, N, K);
            if (ret != 0) goto matmul_cleanup;
        }
        B = B_trans;
        stride_b = K * N;
    }

    if (M <= 32 && N <= 32) {
        dim3 block(16, 16, 1);
        dim3 grid((unsigned int)((N + 15) / 16),
                  (unsigned int)((M + 15) / 16),
                  (unsigned int)batch_size);
        ret = CUDA_KERNEL_LAUNCH(matmul_f32_naive, grid, block, 0, s,
                                  A, B, C, M, N, K,
                                  batch_size, stride_a, stride_b, stride_c);
    } else if (M >= 512 && N >= 512 && K >= 512
               && M % TC_TILE == 0 && N % TC_TILE == 0 && batch_size == 1) {
        dim3 block(32, 1, 1);
        dim3 grid((unsigned int)((N + TC_TILE - 1) / TC_TILE),
                  (unsigned int)((M + TC_TILE - 1) / TC_TILE), 1);
        ret = CUDA_KERNEL_LAUNCH(matmul_f32_tc_kernel, grid, block, 0, s,
                                  A, B, C, M, N, K,
                                  batch_size, stride_a, stride_b, stride_c);
    } else if (M >= 64 && N >= 64) {
        dim3 block(WARP_TILE, WARP_TILE, 1);
        dim3 grid((unsigned int)((N + WARP_TILE - 1) / WARP_TILE),
                  (unsigned int)((M + WARP_TILE - 1) / WARP_TILE),
                  (unsigned int)batch_size);
        ret = CUDA_KERNEL_LAUNCH(matmul_f32_warp, grid, block, 0, s,
                                  A, B, C, M, N, K,
                                  batch_size, stride_a, stride_b, stride_c);
    } else {
        dim3 block(MATMUL_TILE_SIZE, MATMUL_TILE_SIZE, 1);
        dim3 grid((unsigned int)((N + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE),
                  (unsigned int)((M + MATMUL_TILE_SIZE - 1) / MATMUL_TILE_SIZE),
                  (unsigned int)batch_size);
        ret = CUDA_KERNEL_LAUNCH(matmul_f32_tiled, grid, block, 0, s,
                                  A, B, C, M, N, K,
                                  batch_size, stride_a, stride_b, stride_c);
    }
    if (ret != 0) goto matmul_cleanup;

    /* Add bias if present */
    if (bias) {
        int64_t total = batch_size * M * N;
        int block_sz = 256;
        int grid_sz = (int)((total + block_sz - 1) / block_sz);
        ret = CUDA_KERNEL_LAUNCH(matmul_bias_add_kernel, grid_sz, block_sz, 0, s,
                                  C, bias, M, N, batch_size, stride_c);
        if (ret != 0) goto matmul_cleanup;
    }

matmul_cleanup:
    if (A_trans) g_cuda.device_free(A_trans);
    if (B_trans) g_cuda.device_free(B_trans);
    return ret;
}

extern "C" int register_matmul_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "matmul_f32_cuda", .data_type = "f32",
        .func = matmul_f32_cuda, .version = 3, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
