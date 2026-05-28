/* M3: Tensor Core FP16 MHA kernel.
   Uses scalar FP16 for projections, FP32 for softmax and accumulation.
   Separate file to avoid extern __shared__ type conflicts with FP32 kernel. */
#include "operator.h"
#include "cuda_ops.h"
#include "mha_fused_int.h"
#include <cuda_fp16.h>

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

    /* Dynamic shared memory:
       K: S × d (FP16) + V: S × d (FP16)
       BERT-base: 2 * 8 * 64 * 2 = 2KB */
    extern __shared__ __half smem[];
    __half* K_smem = smem;
    __half* V_smem = smem + S_int * d_int;

    /* Initialize output */
    for (int j = tid; j < D_int; j += nthreads) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* Loop over heads */
    for (int h = 0; h < (int)H; h++) {
        int ho = h * d_int;

        /* ---- 1. Compute Q_h[si, :] in FP16, then convert to FP32 for scores ---- */
        /* Q_reg[di] = sum_j X[si,j] * WQ[j*D+ho+di] + bQ[ho+di] */
        float Q_reg[64];  /* FP32 for attention computation */
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                float x_f32 = X_b[si * D_int + j];
                float w_f32 = WQ[j * D_int + ho + di];
                __half x_h = __float2half(x_f32);
                __half w_h = __float2half(w_f32);
                /* FP16 multiply, accumulate in FP32 */
                acc += __half2float(__hmul(x_h, w_h));
            }
            Q_reg[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
        }

        /* ---- 2. K = X·WK → K_smem (FP16) ---- */
        int total_kv = S_int * d_int;
        for (int i = tid; i < total_kv; i += nthreads) {
            int sj = i / d_int, dk = i % d_int;
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                __half x_h = __float2half(X_b[sj * D_int + j]);
                __half w_h = __float2half(WK[j * D_int + ho + dk]);
                acc += __half2float(__hmul(x_h, w_h));
            }
            K_smem[i] = __float2half(acc + (bK ? bK[ho + dk] : 0.0f));
        }

        /* ---- 3. V = X·WV → V_smem (FP16) ---- */
        for (int i = tid; i < total_kv; i += nthreads) {
            int sj = i / d_int, dk = i % d_int;
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                __half x_h = __float2half(X_b[sj * D_int + j]);
                __half w_h = __float2half(WV[j * D_int + ho + dk]);
                acc += __half2float(__hmul(x_h, w_h));
            }
            V_smem[i] = __float2half(acc + (bV ? bV[ho + dk] : 0.0f));
        }
        __syncthreads();

        /* ---- 4. Scores = Q·K^T → softmax (FP32) ---- */
        /* Each thread handles one score row (for our query position si) */
        float scores[64];  /* S_max */
        for (int sj = 0; sj < S_int; sj++) {
            float dot = 0.0f;
            for (int di = 0; di < d_int; di++) {
                /* Q_reg[di] is FP32, K_smem[sj*d+di] is FP16 */
                dot += Q_reg[di] * __half2float(K_smem[sj * d_int + di]);
            }
            scores[sj] = dot * scale;
        }

        /* Softmax in FP32 */
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

        /* ---- 5. Weighted V sum → merged (FP32) ---- */
        float merged[64];  /* d_max */
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int sj = 0; sj < S_int; sj++) {
                acc += scores[sj] * inv_sum * __half2float(V_smem[sj * d_int + di]);
            }
            merged[di] = acc;
        }

        /* ---- 6. Output projection: Y += merged · WO ---- */
        for (int j = tid; j < D_int; j += nthreads) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++) {
                __half m_h = __float2half(merged[di]);
                __half w_h = __float2half(WO[(ho + di) * D_int + j]);
                contrib += __half2float(__hmul(m_h, w_h));
            }
            Y_bs[j] += contrib;
        }
        __syncthreads();
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

    /* K(S*d) + V(S*d) as __half */
    size_t smem_bytes = (size_t)(2 * S * d) * sizeof(__half);

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
