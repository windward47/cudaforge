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
 * block = (min(S, 256), 1, 1)  — threads cooperate on K/V loads
 *
 * Each block handles exactly one output query position.
 * It loops over all H heads sequentially, accumulating
 * the attention-weighted sum into the output.
 *
 * Strategy (per query position si, per head h):
 *   1. Compute Q_h[si, :] into registers
 *   2. Cooperatively load K_h, V_h into shared memory
 *   3. Compute scores = Q·K^T · scale, softmax
 *   4. Weighted V sum → accumulate into output via merged·WO
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

    /* Shared memory for K/V of one head */
    __shared__ float K_smem[MHA_MAX_S_SMEM * MHA_MAX_D];
    __shared__ float V_smem[MHA_MAX_S_SMEM * MHA_MAX_D];

    /* Initialize output for this query position */
    for (int j = tid; j < (int)D; j += num_threads) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    /* Loop over heads */
    for (int h = 0; h < (int)H; h++) {
        int head_offset = h * (int)d;

        /* ---- 1. Compute Q_h[si, :] into registers ---- */
        float Q_reg[MHA_MAX_D];
        for (int di = 0; di < (int)d; di++) {
            float acc = 0.0f;
            for (int j = 0; j < (int)D; j++) {
                acc += X_b[si * D + j] * WQ[j * D + head_offset + di];
            }
            Q_reg[di] = acc + (bQ ? bQ[head_offset + di] : 0.0f);
        }

        if ((int)S <= MHA_MAX_S_SMEM) {
            /* ---- Small S: load full K_h, V_h into shared memory ---- */
            int total_kv = (int)(S * d);
            for (int i = tid; i < total_kv; i += num_threads) {
                int sk = i / (int)d;
                int dk = i % (int)d;
                float k_val = 0.0f, v_val = 0.0f;
                for (int j = 0; j < (int)D; j++) {
                    float xv = X_b[sk * D + j];
                    k_val += xv * WK[j * D + head_offset + dk];
                    v_val += xv * WV[j * D + head_offset + dk];
                }
                K_smem[sk * d + dk] = k_val + (bK ? bK[head_offset + dk] : 0.0f);
                V_smem[sk * d + dk] = v_val + (bV ? bV[head_offset + dk] : 0.0f);
            }
            __syncthreads();

            /* Score row: Q_reg · K_smem^T */
            float scores[MHA_MAX_S_SMEM];
            float max_val = -1e38f;
            for (int sj = 0; sj < (int)S; sj++) {
                float dot = 0.0f;
                for (int di = 0; di < (int)d; di++) {
                    dot += Q_reg[di] * K_smem[sj * d + di];
                }
                scores[sj] = dot * scale;
                if (scores[sj] > max_val) max_val = scores[sj];
            }

            /* Softmax */
            float sum_val = 0.0f;
            for (int sj = 0; sj < (int)S; sj++) {
                scores[sj] = __expf(scores[sj] - max_val);
                sum_val += scores[sj];
            }
            if (sum_val < 1e-12f) sum_val = 1e-12f;

            /* Weighted V sum → merged[di] */
            float merged[MHA_MAX_D];
            for (int di = 0; di < (int)d; di++) {
                float acc = 0.0f;
                for (int sj = 0; sj < (int)S; sj++) {
                    acc += scores[sj] * V_smem[sj * d + di];
                }
                merged[di] = acc / sum_val;
            }

            /* Accumulate into output: Y[si, j] += sum_di merged[di] * WO[head_offset+di, j] */
            for (int j = 0; j < (int)D; j++) {
                float contrib = 0.0f;
                for (int di = 0; di < (int)d; di++) {
                    contrib += merged[di] * WO[(head_offset + di) * D + j];
                }
                Y_bs[j] += contrib;
            }
            __syncthreads();

        } else {
            /* ---- Large S: tiled K/V in S dimension ---- */
            float max_val = -1e38f;
            float sum_val = 0.0f;
            float out_acc[MHA_MAX_D] = {0.0f};

            int num_kv_tiles = ((int)S + MHA_MAX_S_SMEM - 1) / MHA_MAX_S_SMEM;

            for (int kt = 0; kt < num_kv_tiles; kt++) {
                int sj_start = kt * MHA_MAX_S_SMEM;
                int sj_end   = min(sj_start + MHA_MAX_S_SMEM, (int)S);
                int tile_s    = sj_end - sj_start;

                /* Load K/V tile */
                for (int i = tid; i < tile_s * (int)d; i += num_threads) {
                    int sk = i / (int)d;
                    int dk = i % (int)d;
                    int sj_abs = sj_start + sk;
                    float k_val = 0.0f, v_val = 0.0f;
                    for (int j = 0; j < (int)D; j++) {
                        float xv = X_b[sj_abs * D + j];
                        k_val += xv * WK[j * D + head_offset + dk];
                        v_val += xv * WV[j * D + head_offset + dk];
                    }
                    K_smem[sk * d + dk] = k_val + (bK ? bK[head_offset + dk] : 0.0f);
                    V_smem[sk * d + dk] = v_val + (bV ? bV[head_offset + dk] : 0.0f);
                }
                __syncthreads();

                /* Scores for this tile */
                float tile_scores[MHA_MAX_S_SMEM];
                float tile_max = -1e38f;
                for (int sk = 0; sk < tile_s; sk++) {
                    float dot = 0.0f;
                    for (int di = 0; di < (int)d; di++) {
                        dot += Q_reg[di] * K_smem[sk * d + di];
                    }
                    tile_scores[sk] = dot * scale;
                    if (tile_scores[sk] > tile_max) tile_max = tile_scores[sk];
                }

                /* Rescale previous accumulators if new max found */
                if (tile_max > max_val) {
                    float rescale = __expf(max_val - tile_max);
                    sum_val *= rescale;
                    for (int di = 0; di < (int)d; di++) {
                        out_acc[di] *= rescale;
                    }
                    max_val = tile_max;
                }

                /* Accumulate exp + weighted V */
                for (int sk = 0; sk < tile_s; sk++) {
                    float p = __expf(tile_scores[sk] - max_val);
                    sum_val += p;
                    for (int di = 0; di < (int)d; di++) {
                        out_acc[di] += p * V_smem[sk * d + di];
                    }
                }
                __syncthreads();
            }

            /* Normalize and accumulate into output */
            if (sum_val < 1e-12f) sum_val = 1e-12f;
            float inv_sum = 1.0f / sum_val;

            for (int j = 0; j < (int)D; j++) {
                float contrib = 0.0f;
                for (int di = 0; di < (int)d; di++) {
                    contrib += out_acc[di] * inv_sum * WO[(head_offset + di) * D + j];
                }
                Y_bs[j] += contrib;
            }
        }
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

    /* One block per (batch, query_position) */
    dim3 grid((unsigned int)(B * S), 1, 1);

    /* Threads per block: enough to cover S for cooperative K/V loads */
    int threads = (int)(S < 32 ? 32 : (S > 256 ? 256 : S));
    dim3 block(threads, 1, 1);

    int has_residual = p->has_residual && R ? 1 : 0;

    return CUDA_KERNEL_LAUNCH(mha_fused_kernel, grid, block, 0, s,
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
