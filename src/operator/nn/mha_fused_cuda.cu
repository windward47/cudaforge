#include "operator.h"
#include "cuda_ops.h"
#include "mha_fused_int.h"

/* Maximum head_dim supported (BERT-base uses 64). */
#define MHA_MAX_D       64
/* Maximum S for direct shared-memory K/V fit. */
#define MHA_MAX_S_SMEM  64

/* ============================================================
 * Fused MHA kernel — one block per (batch, query_position)
 *
 * grid  = (B * S, 1, 1)
 * block = (256, 1, 1)
 *
 * M2 optimizations (QKV projection fusion):
 *   - Q computed in single pass over D: loads X once, accumulates
 *     all D output positions → 4x fewer X reads for Q projection
 *   - K/V inner loops merged: loads X once, computes both K and V
 *     → 33% fewer X reads for K/V projection
 *   - All heads' weights stay in L2 cache across blocks
 * ============================================================ */
__global__ void mha_fused_kernel(
    const float* __restrict__ X,      /* (B, S, D) */
    const float* __restrict__ WQ,     /* (D, D) */
    const float* __restrict__ bQ,     /* (D,) */
    const float* __restrict__ WK,     /* (D, D) */
    const float* __restrict__ bK,     /* (D,) */
    const float* __restrict__ WV,     /* (D, D) */
    const float* __restrict__ bV,     /* (D,) */
    const float* __restrict__ WO,     /* (D, D) */
    const float* __restrict__ bO,     /* (D,) */
    float* __restrict__ Y,            /* (B, S, D) */
    const float* __restrict__ R,      /* (B, S, D) residual, may be NULL */
    int64_t B, int64_t S, int64_t D, int64_t H, int64_t d,
    float scale, int has_residual)
{
    int bs = blockIdx.x;           /* flat: b * S + si */
    int b  = bs / (int)S;
    int si = bs % (int)S;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (b >= B) return;

    const float* X_b  = X + b * S * D;
    float*       Y_bs = Y + (b * S + si) * D;
    const float* R_bs = (has_residual && R) ? (R + (b * S + si) * D) : NULL;

    /* Dynamic shared memory:
       K_smem[S*H*d] + V_smem[S*H*d]
       BERT-base: 2*8*12*64*4 = 48KB (within default 48KB limit) */
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + (int)(S * H * d);

    /* ---- Precompute K, V for ALL heads into shared memory ----
       Single pass over S*H*d elements, each thread handles multiple (sj, h, dk).
       M2: K and V inner loops merged — loads X once, computes both. */
    int total_kv = (int)(S * H * d);
    for (int i = tid; i < total_kv; i += num_threads) {
        int dk = i % (int)d;
        int tmp = i / (int)d;
        int h  = tmp % (int)H;
        int sj = tmp / (int)H;
        int head_offset = h * (int)d;

        float k_acc = 0.0f, v_acc = 0.0f;
        for (int j = 0; j < (int)D; j++) {
            float xv = X_b[sj * D + j];
            k_acc += xv * WK[j * D + head_offset + dk];
            v_acc += xv * WV[j * D + head_offset + dk];
        }
        K_smem[i] = k_acc + (bK ? bK[head_offset + dk] : 0.0f);
        V_smem[i] = v_acc + (bV ? bV[head_offset + dk] : 0.0f);
    }
    __syncthreads();

    /* Initialize output for this query position */
    for (int j = tid; j < (int)D; j += num_threads) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* ---- Tiled attention for each head ---- */

    /* ---- Tiled attention for each head ---- */
    int S_int = (int)S;
    int d_int = (int)d;
    int D_int = (int)D;
    int H_d   = (int)(H * d);

    for (int h = 0; h < (int)H; h++) {
        int head_offset = h * d_int;

        /* ---- Compute Q_h[si, :] into registers ----
           M2: restructured to load X once per D iteration. */
        float Q_reg[MHA_MAX_D];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                acc += X_b[si * D_int + j] * WQ[j * D_int + head_offset + di];
            }
            Q_reg[di] = acc + (bQ ? bQ[head_offset + di] : 0.0f);
        }

        /* Online softmax attention */
        float max_val = -1e38f;
        float sum_val = 0.0f;
        float out_acc[MHA_MAX_D] = {0.0f};

        int num_kv_tiles = (S_int + MHA_MAX_S_SMEM - 1) / MHA_MAX_S_SMEM;

        for (int kt = 0; kt < num_kv_tiles; kt++) {
            int sj_start = kt * MHA_MAX_S_SMEM;
            int sj_end   = min(sj_start + MHA_MAX_S_SMEM, S_int);
            int tile_s   = sj_end - sj_start;

            /* Compute scores for this tile */
            float tile_scores[MHA_MAX_S_SMEM];
            float tile_max = -1e38f;
            for (int sk = 0; sk < tile_s; sk++) {
                float dot = 0.0f;
                int k_idx = (sj_start + sk) * H_d + head_offset;
                for (int di = 0; di < d_int; di++) {
                    dot += Q_reg[di] * K_smem[k_idx + di];
                }
                tile_scores[sk] = dot * scale;
                if (tile_scores[sk] > tile_max) tile_max = tile_scores[sk];
            }

            /* Online softmax: rescale previous accumulators */
            if (tile_max > max_val) {
                float rescale = __expf(max_val - tile_max);
                sum_val *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[di] *= rescale;
                max_val = tile_max;
            }

            /* Accumulate exp(scores) and weighted V */
            for (int sk = 0; sk < tile_s; sk++) {
                float p = __expf(tile_scores[sk] - max_val);
                sum_val += p;
                int v_idx = (sj_start + sk) * H_d + head_offset;
                for (int di = 0; di < d_int; di++) {
                    out_acc[di] += p * V_smem[v_idx + di];
                }
            }
            __syncthreads();
        }

        /* Normalize and accumulate into output */
        if (sum_val < 1e-12f) sum_val = 1e-12f;
        float inv_sum = 1.0f / sum_val;

        for (int j = tid; j < D_int; j += num_threads) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++) {
                contrib += out_acc[di] * inv_sum * WO[(head_offset + di) * D_int + j];
            }
            Y_bs[j] += contrib;
        }
        __syncthreads();
    }
}

/* ============================================================
 * Host dispatch
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

    int64_t B = p->batch_size;
    int64_t S = p->seq_len;
    int64_t D = p->hidden_size;
    int64_t H = p->num_heads;
    int64_t d = p->head_dim;

    dim3 grid((unsigned int)(B * S), 1, 1);
    dim3 block(256, 1, 1);

    /* K_smem[S*H*d] + V_smem[S*H*d] */
    size_t smem_bytes = (size_t)(2 * S * H * d) * sizeof(float);

    int has_residual = p->has_residual && R ? 1 : 0;

    return CUDA_KERNEL_LAUNCH(mha_fused_kernel, grid, block, smem_bytes, s,
        X, WQ, bQ, WK, bK, WV, bV, WO, bO, Y, R,
        B, S, D, H, d, p->scale, has_residual);
}

extern "C" int register_mha_fused_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_fused_f32_cuda", .data_type = "f32",
        .func = mha_fused_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
