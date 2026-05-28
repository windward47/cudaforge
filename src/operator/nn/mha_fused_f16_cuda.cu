/* M3: FP16 MHA kernel with WMMA for output projection.
   Scalar FP16 for QKV projections, WMMA 16×16×16 for output projection.
   Pre-loads WO slice into shared memory to avoid large local arrays. */
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

    /* Dynamic shared memory layout:
       K(S×d) + V(S×d) + merged(S_pad×d) as __half
       W_tile(WMMA_K×WMMA_N) as __half, tbuf(WMMA_M×WMMA_N) as float
       BERT-base: (2*8*64 + 16*64)*2 + 16*16*2 + 16*16*4 = 6144 + 512 + 1024 = 7680 bytes */
    extern __shared__ __half smem[];
    __half* K_smem = smem;
    __half* V_smem = smem + S_int * d_int;
    __half* M_smem = smem + 2 * S_int * d_int;  /* merged: S_pad × d */
    __half* W_tile = M_smem + S_pad * d_int;     /* WMMA_K × WMMA_N */
    float*  tbuf   = (float*)(W_tile + WMMA_K * WMMA_N);  /* WMMA_M × WMMA_N */

    /* Initialize output */
    for (int j = tid; j < D_int; j += nthreads) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* Process each head */
    for (int h = 0; h < (int)H; h++) {
        int ho = h * d_int;

        /* ---- 1. Q projection: X·WQ → Q_reg (FP16 mul, FP32 acc) ---- */
        float Q_reg[64];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                acc += __half2float(__hmul(
                    __float2half(X_b[si * D_int + j]),
                    __float2half(WQ[j * D_int + ho + di])));
            }
            Q_reg[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
        }

        /* ---- 2. K = X·WK → K_smem (FP16) ---- */
        for (int i = tid; i < S_int * d_int; i += nthreads) {
            int sj = i / d_int, dk = i % d_int;
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                acc += __half2float(__hmul(
                    __float2half(X_b[sj * D_int + j]),
                    __float2half(WK[j * D_int + ho + dk])));
            }
            K_smem[i] = __float2half(acc + (bK ? bK[ho + dk] : 0.0f));
        }

        /* ---- 3. V = X·WV → V_smem (FP16) ---- */
        for (int i = tid; i < S_int * d_int; i += nthreads) {
            int sj = i / d_int, dk = i % d_int;
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                acc += __half2float(__hmul(
                    __float2half(X_b[sj * D_int + j]),
                    __float2half(WV[j * D_int + ho + dk])));
            }
            V_smem[i] = __float2half(acc + (bV ? bV[ho + dk] : 0.0f));
        }
        __syncthreads();

        /* ---- 4. Scores = Q·K^T → softmax (FP32) ---- */
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

        /* ---- 5. Weighted V sum → M_smem (FP16) ---- */
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

        /* ---- 6. Output projection: Y += M_smem · WO (WMMA) ---- */
        /* M_smem: (S_pad, d) FP16, WO: (D, D) FP32 → load WO tile as FP16 into tbuf */
        for (int _tr = 0; _tr < S_pad; _tr += WMMA_M) {
            for (int _tc = 0; _tc < D_int; _tc += WMMA_N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                for (int _k = 0; _k < d_int; _k += WMMA_K) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

                    /* Load A from M_smem (S_pad × d, row-major, ldim = d) */
                    wmma::load_matrix_sync(a_frag, M_smem + _tr * d_int + _k, d_int);

                    /* Load B: WO slice (d × D) as FP16 into W_tile (col-major). */
                    for (int i = tid; i < WMMA_K * WMMA_N; i += nthreads) {
                        int r = i / WMMA_N, c = i % WMMA_N;
                        int gr = ho + _k + r, gc = _tc + c;
                        W_tile[r + c * WMMA_K] = (gr < D_int && gc < D_int) ?
                            __float2half(WO[gr * D_int + gc]) : __float2half(0.0f);
                    }
                    __syncthreads();
                    wmma::load_matrix_sync(b_frag, W_tile, WMMA_K);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    __syncthreads();
                }

                /* Store FP32 result to tbuf, accumulate into Y_bs */
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

    /* K(S*d) + V(S*d) + merged(S_pad*d) as __half + W_tile(WMMA_K*WMMA_N) as __half + tbuf(16*16 as float) */
    int S_pad = ((int)S + WMMA_M - 1) / WMMA_M * WMMA_M;
    size_t smem_bytes = (size_t)(2 * S * d + S_pad * d + WMMA_K * WMMA_N) * sizeof(__half)
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
