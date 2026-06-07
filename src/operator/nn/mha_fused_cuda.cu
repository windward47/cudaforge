#include "operator.h"
#include "cuda_ops.h"
#include "mha_fused_int.h"

/* ============================================================
 * Hybrid MHA Dispatch:
 *   S <= 64:  Preloaded K/V kernel (fast, all K/V in smem)
 *   S > 64:   Flash Attention kernel (tiled K/V from global mem)
 *
 * Both kernels support GQA and causal masking.
 * ============================================================ */

/* ============================================================
 * Kernel 1: Preloaded K/V (original, fast for short sequences)
 *
 * All K/V for ALL heads loaded into shared memory upfront.
 * Shared memory: 2 * S * H * d floats
 * Max S ≈ 64 for BERT-base (H=12, d=64) within 48 KB limit.
 * ============================================================ */
__global__ void mha_preloaded_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ,
    const float* __restrict__ bQ,
    const float* __restrict__ WK,
    const float* __restrict__ bK,
    const float* __restrict__ WV,
    const float* __restrict__ bV,
    const float* __restrict__ WO,
    const float* __restrict__ bO,
    float* __restrict__ Y,
    const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int si = bs % (int)S;
    const int tid = threadIdx.x;
    const int NT = blockDim.x;

    if (b >= (int)B) return;

    const int S_int = (int)S;
    const int D_int = (int)D;
    const int d_int = (int)d;
    const int H_q_int = (int)H_q;
    const int H_kv_int = (int)H_kv;
    const int group_size = H_q_int / H_kv_int;
    const int H_d = H_kv_int * d_int;

    const float* X_b  = X + (int64_t)b * S_int * D_int;
    float*       Y_bs = Y + ((int64_t)b * S_int + si) * D_int;
    const float* R_bs = (has_residual && R) ? (R + ((int64_t)b * S_int + si) * D_int) : NULL;

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + S_int * H_d;

    /* ---- Precompute K, V for ALL KV heads into shared memory ---- */
    int total_kv = S_int * H_d;
    for (int i = tid; i < total_kv; i += NT) {
        int dk = i % d_int;
        int tmp = i / d_int;
        int h  = tmp % H_kv_int;
        int sj = tmp / H_kv_int;
        int kv_head_offset = h * d_int;

        float k_acc = 0.0f, v_acc = 0.0f;
        for (int j = 0; j < D_int; j++) {
            float xv = X_b[sj * D_int + j];
            k_acc += xv * WK[j * D_int + kv_head_offset + dk];
            v_acc += xv * WV[j * D_int + kv_head_offset + dk];
        }
        K_smem[i] = k_acc + (bK ? bK[kv_head_offset + dk] : 0.0f);
        V_smem[i] = v_acc + (bV ? bV[kv_head_offset + dk] : 0.0f);
    }
    __syncthreads();

    /* Initialize output */
    for (int j = tid; j < D_int; j += NT) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* ---- Per-head attention ---- */
    for (int h = 0; h < H_q_int; h++) {
        int head_offset = h * d_int;
        int kv_h = h / group_size;
        int kv_head_offset = kv_h * d_int;

        /* Compute Q_h[si, :] into registers */
        float Q_reg[FA_MAX_D];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++)
                acc += X_b[si * D_int + j] * WQ[j * D_int + head_offset + di];
            Q_reg[di] = acc + (bQ ? bQ[head_offset + di] : 0.0f);
        }

        /* Online softmax attention */
        float max_val = -1e38f;
        float sum_val = 0.0f;
        float out_acc[FA_MAX_D] = {0.0f};

        int causal_limit = causal ? (si + 1) : S_int;
        int num_kv_tiles = (causal_limit + 63) / 64;

        for (int kt = 0; kt < num_kv_tiles; kt++) {
            int sj_start = kt * 64;
            int sj_end   = min(sj_start + 64, causal_limit);
            int tile_s   = sj_end - sj_start;

            float tile_scores[64];
            float tile_max = -1e38f;
            for (int sk = 0; sk < tile_s; sk++) {
                float dot = 0.0f;
                int k_idx = (sj_start + sk) * H_d + kv_head_offset;
                for (int di = 0; di < d_int; di++)
                    dot += Q_reg[di] * K_smem[k_idx + di];
                tile_scores[sk] = dot * scale;
                if (tile_scores[sk] > tile_max) tile_max = tile_scores[sk];
            }

            if (tile_max > max_val) {
                float rescale = __expf(max_val - tile_max);
                sum_val *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[di] *= rescale;
                max_val = tile_max;
            }

            for (int sk = 0; sk < tile_s; sk++) {
                float p = __expf(tile_scores[sk] - max_val);
                sum_val += p;
                int v_idx = (sj_start + sk) * H_d + kv_head_offset;
                for (int di = 0; di < d_int; di++)
                    out_acc[di] += p * V_smem[v_idx + di];
            }
        }

        /* Normalize and output projection */
        if (sum_val < 1e-12f) sum_val = 1e-12f;
        float inv_sum = 1.0f / sum_val;

        for (int j = tid; j < D_int; j += NT) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++)
                contrib += out_acc[di] * inv_sum * WO[(head_offset + di) * D_int + j];
            Y_bs[j] += contrib;
        }
        __syncthreads();
    }
}

/* ============================================================
 * Kernel 2: Flash Attention (for long sequences S > 64)
 *
 * K/V computed per-head per-tile from global memory.
 * Shared memory independent of S.
 * ============================================================ */
__global__ void mha_flash_attn_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ,
    const float* __restrict__ bQ,
    const float* __restrict__ WK,
    const float* __restrict__ bK,
    const float* __restrict__ WV,
    const float* __restrict__ bV,
    const float* __restrict__ WO,
    const float* __restrict__ bO,
    float* __restrict__ Y,
    const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int si = bs % (int)S;
    const int tid = threadIdx.x;
    const int NT = blockDim.x;

    if (b >= (int)B) return;

    const int S_int = (int)S;
    const int D_int = (int)D;
    const int d_int = (int)d;
    const int H_q_int = (int)H_q;
    const int H_kv_int = (int)H_kv;
    const int group_size = H_q_int / H_kv_int;
    const int BC = 64;

    const float* X_b  = X + (int64_t)b * S_int * D_int;
    float*       Y_bs = Y + ((int64_t)b * S_int + si) * D_int;
    const float* R_bs = (has_residual && R) ? (R + ((int64_t)b * S_int + si) * D_int) : NULL;

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + NT * d_int;
    float* scores = smem + (NT + BC) * d_int;
    float* red    = smem + (NT + BC) * d_int + BC;

    /* Initialize output */
    for (int j = tid; j < D_int; j += NT) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* Per-head attention */
    for (int h = 0; h < H_q_int; h++) {
        const int head_offset = h * d_int;
        const int kv_h = h / group_size;
        const int kv_head_offset = kv_h * d_int;

        /* Compute Q cooperatively */
        for (int di = tid; di < d_int; di += NT) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++)
                acc += X_b[si * D_int + j] * WQ[j * D_int + head_offset + di];
            K_smem[di] = acc + (bQ ? bQ[head_offset + di] : 0.0f);
        }
        __syncthreads();

        float Q_reg[FA_MAX_D];
        for (int di = 0; di < d_int; di++)
            Q_reg[di] = K_smem[di];
        __syncthreads();

        /* Online softmax attention */
        float max_val = -1e38f;
        float sum_val = 0.0f;
        float out_acc[FA_MAX_D];
        for (int di = 0; di < d_int; di++) out_acc[di] = 0.0f;

        int causal_limit = causal ? (si + 1) : S_int;
        int num_kv_tiles = (causal_limit + BC - 1) / BC;

        for (int kt = 0; kt < num_kv_tiles; kt++) {
            int kv_start = kt * BC;
            int kv_end   = min(kv_start + BC, causal_limit);
            int tile_s   = kv_end - kv_start;

            /* K/V projection into shared memory */
            for (int i = tid; i < tile_s * d_int; i += NT) {
                int sk = i / d_int;
                int dk = i % d_int;
                int gp = kv_start + sk;

                float k_acc = 0.0f, v_acc = 0.0f;
                for (int j = 0; j < D_int; j++) {
                    float xv = X_b[gp * D_int + j];
                    k_acc += xv * WK[j * D_int + kv_head_offset + dk];
                    v_acc += xv * WV[j * D_int + kv_head_offset + dk];
                }
                K_smem[sk * d_int + dk] = k_acc + (bK ? bK[kv_head_offset + dk] : 0.0f);
                V_smem[sk * d_int + dk] = v_acc + (bV ? bV[kv_head_offset + dk] : 0.0f);
            }
            __syncthreads();

            /* Compute scores */
            float tile_max = -1e38f;
            for (int sk = tid; sk < tile_s; sk += NT) {
                float dot = 0.0f;
                for (int di = 0; di < d_int; di++)
                    dot += Q_reg[di] * K_smem[sk * d_int + di];
                float s = dot * scale;
                if (causal && (kv_start + sk) > si) s = -1e38f;
                scores[sk] = s;
                if (s > tile_max) tile_max = s;
            }
            __syncthreads();

            /* Tile max reduction */
            red[tid] = tile_max;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride && red[tid + stride] > red[tid])
                    red[tid] = red[tid + stride];
                __syncthreads();
            }
            tile_max = red[0];
            __syncthreads();

            /* Online softmax rescale */
            if (tile_max > max_val) {
                float rescale = expf(max_val - tile_max);
                sum_val *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[di] *= rescale;
                max_val = tile_max;
            }

            /* Accumulate exp(scores - max) * V */
            float local_sum = 0.0f;
            float local_acc[FA_MAX_D];
            for (int di = 0; di < d_int; di++) local_acc[di] = 0.0f;

            for (int sk = tid; sk < tile_s; sk += NT) {
                float p = expf(scores[sk] - max_val);
                local_sum += p;
                for (int di = 0; di < d_int; di++)
                    local_acc[di] += p * V_smem[sk * d_int + di];
            }

            /* Reduce sum */
            red[tid] = local_sum;
            __syncthreads();
            for (int stride = NT / 2; stride > 0; stride >>= 1) {
                if (tid < stride) red[tid] += red[tid + stride];
                __syncthreads();
            }
            sum_val += red[0];

            /* Reduce out_acc */
            for (int di = 0; di < d_int; di++)
                K_smem[tid * d_int + di] = local_acc[di];
            __syncthreads();

            if (tid == 0) {
                for (int di = 0; di < d_int; di++) {
                    float acc = 0.0f;
                    for (int t = 0; t < NT; t++)
                        acc += K_smem[t * d_int + di];
                    out_acc[di] += acc;
                }
            }
            __syncthreads();
        }

        /* Normalize and output projection */
        if (tid == 0) {
            if (sum_val < 1e-12f) sum_val = 1e-12f;
            float inv_sum = 1.0f / sum_val;

            for (int j = 0; j < D_int; j++) {
                float contrib = 0.0f;
                for (int di = 0; di < d_int; di++)
                    contrib += out_acc[di] * inv_sum * WO[(head_offset + di) * D_int + j];
                Y_bs[j] += contrib;
            }
        }
        __syncthreads();
    }
}

/* ============================================================
 * Host dispatch — hybrid: preloaded for S<=64, flash for S>64
 * ============================================================ */
int mha_fused_f32_cuda(const void* inputs[], void* outputs[],
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

    int64_t B    = p->batch_size;
    int64_t S    = p->seq_len;
    int64_t D    = p->hidden_size;
    int64_t H_q  = p->num_heads;
    int64_t H_kv = p->num_kv_heads > 0 ? p->num_kv_heads : H_q;
    int64_t d    = p->head_dim;

    dim3 grid((unsigned int)(B * S), 1, 1);
    dim3 block(256, 1, 1);
    int has_residual = p->has_residual && R ? 1 : 0;
    int causal_mask  = p->causal ? 1 : 0;

    if (S <= 64) {
        /* ---- Fast path: preloaded K/V (all in shared memory) ---- */
        size_t smem_bytes = (size_t)(2 * S * H_kv * d) * sizeof(float);
        return CUDA_KERNEL_LAUNCH(mha_preloaded_kernel, grid, block, smem_bytes, s,
            X, WQ, bQ, WK, bK, WV, bV, WO, bO, Y, R,
            B, S, D, H_q, H_kv, d, p->scale, has_residual, causal_mask);
    } else {
        /* ---- Flash Attention path: tiled K/V from global memory ---- */
        int NT = 256, BC = 64;
        size_t smem_bytes = (size_t)((NT + BC) * d + BC + NT) * sizeof(float);

        cudaFuncSetAttribute(mha_flash_attn_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem_bytes);

        return CUDA_KERNEL_LAUNCH(mha_flash_attn_kernel, grid, block, smem_bytes, s,
            X, WQ, bQ, WK, bK, WV, bV, WO, bO, Y, R,
            B, S, D, H_q, H_kv, d, p->scale, has_residual, causal_mask);
    }
}

extern "C" int register_mha_fused_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_fused_f32_cuda", .data_type = "f32",
        .func = mha_fused_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
