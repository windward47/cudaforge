/* M3: Full WMMA FP16 MHA kernel.
   X loaded once as FP16 into shared memory. QKV and output projections
   use WMMA 16×16×16 Tensor Cores. WO loaded in tiles (d×WMMA_N) to fit
   in shared memory. Attention/softmax in FP32.

   Shared memory (BERT-base, ~41KB):
     X_h:    S_pad × D = 16 × 768  = 12288 half = 24KB
     W_tile: D × d     = 768 × 64  = 49152 half = 96KB — TOO LARGE!
     → Use D × WMMA_N = 768 × 16  = 12288 half = 24KB tile instead.
     → But even 24KB + 24KB = 48KB leaves no room for K/V/M/tbuf.

     Revised: W_tile = WMMA_K × WMMA_N = 16 × 16 = 512 half = 1KB (per-tile loading)
     X_h:    24KB, K: 1KB, V: 1KB, M: 2KB, W_tile: 1KB, tbuf: 1KB = ~30KB */
#include "operator.h"
#include "cuda_ops.h"
#include "mha_fused_int.h"
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define MAX_SEQ_LEN 64

__global__ void mha_fused_f16_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ WK, const float* __restrict__ bK,
    const float* __restrict__ WV, const float* __restrict__ bV,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y, const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H, int64_t d,
    float scale, int has_residual)
{
    int bs = blockIdx.x;
    int b  = bs / (int)S;
    int si = bs % (int)S;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    if (b >= B) return;

    const float* X_b = X + b * S * D;
    float* Y_bs = Y + (b * S + si) * D;
    const float* R_bs = (has_residual && R) ? (R + (b * S + si) * D) : NULL;
    int S_int = (int)S, d_int = (int)d, D_int = (int)D;
    int S_pad = ((S_int + WMMA_M - 1) / WMMA_M) * WMMA_M;

    /* Shared memory:
       X_h: S_pad × D (FP16, loaded once)
       K_smem: S × d (FP16)
       V_smem: S × d (FP16)
       M_smem: S_pad × d (FP16, merged)
       W_tile: WMMA_K × WMMA_N (FP16, per-tile weight)
       tbuf: WMMA_M × WMMA_N (FP32, WMMA store)
       Total: (S_pad*D + 2*S*d + S_pad*d + WMMA_K*WMMA_N)*2 + 16*16*4
       BERT-base: (12288 + 1024 + 1024 + 256)*2 + 1024 = 29184 + 1024 = 30208 bytes ≈ 30KB */
    extern __shared__ __half smem[];
    __half* X_h    = smem;
    __half* K_smem = X_h + S_pad * D_int;
    __half* V_smem = K_smem + S_int * d_int;
    __half* M_smem = V_smem + S_int * d_int;
    __half* W_tile = M_smem + S_pad * d_int;
    float*  tbuf   = (float*)(W_tile + WMMA_K * WMMA_N);

    /* Load X (S×D) → X_h (S_pad×D) as FP16, once */
    for (int i = tid; i < S_pad * D_int; i += nthreads) {
        int r = i / D_int, c = i % D_int;
        X_h[r * D_int + c] = __float2half((r < S_int) ? X_b[r * D_int + c] : 0.0f);
    }
    __syncthreads();

    /* Initialize output */
    for (int j = tid; j < D_int; j += nthreads) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* ---- Helper: load weight tile (WMMA_K × WMMA_N) as FP16 ----
       Loads W[ho+_k+r, _tc+c] into W_tile as col-major for WMMA matrix_b. */
    #define LOAD_W_TILE(W_SRC, ho, _k, _tc) \
        for (int _i = tid; _i < WMMA_K * WMMA_N; _i += nthreads) { \
            int _r = _i / WMMA_N, _cc = _i % WMMA_N; \
            int _gr = (ho) + (_k) + _r, _gc = (_tc) + _cc; \
            W_tile[_r * WMMA_N + _cc] = (_gr < D_int && _gc < D_int) ? \
                __float2half(W_SRC[(_gr) * D_int + (_gc)]) : __float2half(0.0f); \
        } \
        __syncthreads()

    /* ---- Helper: WMMA tile loop for projection ----
       X_h(S_pad×D, row-major) × weight(D×d, loaded per tile as col-major)
       → out(S_pad×d, FP16), with bias */
    #define WMMA_PROJ(W_SRC, ho, OUT_h, bias_ptr, bias_off) \
        for (int _tr = 0; _tr < S_pad; _tr += WMMA_M) { \
            for (int _tc = 0; _tc < d_int; _tc += WMMA_N) { \
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> _c; \
                wmma::fill_fragment(_c, 0.0f); \
                for (int _k = 0; _k < D_int; _k += WMMA_K) { \
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> _a; \
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> _b; \
                    wmma::load_matrix_sync(_a, X_h + _tr * D_int + _k, D_int); \
                    LOAD_W_TILE(W_SRC, ho, _k, _tc); \
                    wmma::load_matrix_sync(_b, W_tile, WMMA_N); \
                    wmma::mma_sync(_c, _a, _b, _c); \
                } \
                wmma::store_matrix_sync(tbuf, _c, WMMA_N, wmma::mem_row_major); \
                __syncthreads(); \
                for (int _i = tid; _i < WMMA_M * WMMA_N; _i += nthreads) { \
                    int _r = _i / WMMA_N, _cc = _i % WMMA_N; \
                    int _gr = _tr + _r, _gc = _tc + _cc; \
                    if (_gr < S_pad && _gc < d_int) { \
                        float _v = tbuf[_r * WMMA_N + _cc]; \
                        _v += (bias_ptr) ? (bias_ptr)[(bias_off) + _gc] : 0.0f; \
                        (OUT_h)[_gr * d_int + _gc] = __float2half(_v); \
                    } \
                } \
                __syncthreads(); \
            } \
        }

    /* Process each head */
    for (int h = 0; h < (int)H; h++) {
        int ho = h * d_int;

        /* ---- QKV projections via WMMA ---- */
        WMMA_PROJ(WQ, ho, M_smem, bQ, ho);  /* Q → M_smem */
        WMMA_PROJ(WK, ho, K_smem, bK, ho);  /* K → K_smem */
        WMMA_PROJ(WV, ho, V_smem, bV, ho);  /* V → V_smem */

        /* ---- Read Q for our query position → Q_reg (FP32) ---- */
        float Q_reg[64];
        for (int di = 0; di < d_int; di++) {
            Q_reg[di] = __half2float(M_smem[si * d_int + di]);
        }

        /* ---- Attention scores = Q · K^T → softmax (FP32) ---- */
        float scores[MAX_SEQ_LEN];
        for (int sj = 0; sj < S_int; sj++) {
            float dot = 0.0f;
            for (int di = 0; di < d_int; di++) {
                dot += Q_reg[di] * __half2float(K_smem[sj * d_int + di]);
            }
            scores[sj] = dot * scale;
        }

        float max_val = -1e38f;
        for (int sj = 0; sj < S_int; sj++)
            if (scores[sj] > max_val) max_val = scores[sj];
        float sum_val = 0.0f;
        for (int sj = 0; sj < S_int; sj++) {
            scores[sj] = __expf(scores[sj] - max_val);
            sum_val += scores[sj];
        }
        if (sum_val < 1e-12f) sum_val = 1e-12f;
        float inv_sum = 1.0f / sum_val;

        /* ---- Weighted V sum → M_smem (FP16) ---- */
        for (int i = tid; i < S_pad * d_int; i += nthreads) {
            int r = i / d_int, c = i % d_int;
            if (r < S_int) {
                float acc = 0.0f;
                for (int sj = 0; sj < S_int; sj++) {
                    acc += scores[sj] * inv_sum * __half2float(V_smem[sj * d_int + c]);
                }
                M_smem[i] = __float2half(acc);
            } else {
                M_smem[i] = __float2half(0.0f);
            }
        }
        __syncthreads();

        /* ---- Output projection: Y += M_smem · WO (WMMA, tiled) ----
           M_smem(S_pad×d) × WO(d×D) → accumulate into Y_bs.
           WO loaded per tile (WMMA_K × WMMA_N) into W_tile. */
        for (int _tr = 0; _tr < S_pad; _tr += WMMA_M) {
            for (int _tc = 0; _tc < D_int; _tc += WMMA_N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                for (int _k = 0; _k < d_int; _k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

                    wmma::load_matrix_sync(a_frag, M_smem + _tr * d_int + _k, d_int);
                    LOAD_W_TILE(WO, ho, _k, _tc);
                    wmma::load_matrix_sync(b_frag, W_tile, WMMA_N);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                wmma::store_matrix_sync(tbuf, c_frag, WMMA_N, wmma::mem_row_major);
                __syncthreads();

                if (_tr <= si && si < _tr + WMMA_M) {
                    int local_row = si - _tr;
                    for (int _i = tid; _i < WMMA_N && _tc + _i < D_int; _i += nthreads) {
                        Y_bs[_tc + _i] += tbuf[local_row * WMMA_N + _i];
                    }
                }
                __syncthreads();
            }
        }
    }
}

int mha_fused_f16_cuda(const void* inputs[], void* outputs[],
                        const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const mha_fused_params_t* p = (const mha_fused_params_t*)params;
    const float* X  = (const float*)inputs[0];
    const float* R  = (const float*)inputs[1];
    const float* WQ = (const float*)inputs[2];
    const float* bQ = (const float*)inputs[3];
    const float* WK = (const float*)inputs[4];
    const float* bK = (const float*)inputs[5];
    const float* WV = (const float*)inputs[6];
    const float* bV = (const float*)inputs[7];
    const float* WO = (const float*)inputs[8];
    const float* bO = (const float*)inputs[9];
    float* Y        = (float*)outputs[0];

    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t B = p->batch_size, S = p->seq_len, D = p->hidden_size;
    int64_t H = p->num_heads, d = p->head_dim;

    dim3 grid((unsigned int)(B * S), 1, 1);
    dim3 block(256, 1, 1);

    /* X_h(S_pad×D) + K(S×d) + V(S×d) + M(S_pad×d) as __half
       + W_tile(WMMA_K×WMMA_N) as __half + tbuf(16×16) as float */
    int S_pad = ((int)S + WMMA_M - 1) / WMMA_M * WMMA_M;
    size_t smem_bytes = (size_t)(S_pad * D + 2 * S * d + S_pad * d + WMMA_K * WMMA_N) * sizeof(__half)
                      + (size_t)(WMMA_M * WMMA_N) * sizeof(float);

    return CUDA_KERNEL_LAUNCH(mha_fused_f16_kernel, grid, block, smem_bytes, s,
        X, WQ, bQ, WK, bK, WV, bV, WO, bO, Y, R,
        B, S, D, H, d, p->scale, p->has_residual && R ? 1 : 0);
}

extern "C" int register_mha_fused_f16_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_fused_f16_cuda", .data_type = "f32",
        .func = mha_fused_f16_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
