#include "operator.h"
#include "mha_fused_int.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Reference CPU implementation of fused Multi-Head Attention.

   inputs[0] = X       — (B, S, D) LayerNorm output
   inputs[1] = residual — (B, S, D) original input, or NULL
   inputs[2] = W_Q     — (D, D)
   inputs[3] = b_Q     — (D,)
   inputs[4] = W_K     — (D, D)
   inputs[5] = b_K     — (D,)
   inputs[6] = W_V     — (D, D)
   inputs[7] = b_V     — (D,)
   inputs[8] = W_O     — (D, D)
   inputs[9] = b_O     — (D,)

   outputs[0] = Y — (B, S, D) result

   Step-by-step:
     1. Q = X·W_Q + b_Q,  K = X·W_K + b_K,  V = X·W_V + b_V
     2. Reshape: (B,S,D) → (B,S,H,d), Transpose: → (B,H,S,d)
     3. scores = Q·K^T / sqrt(d)   (B,H,S,S)
     4. probs = softmax(scores)
     5. attn  = probs·V            (B,H,S,d)
     6. Merge: Transpose → (B,S,H,d), Reshape → (B,S,D)
     7. out = attn·W_O + b_O       (B,S,D)
     8. if has_residual: out += residual
 */
int mha_fused_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
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
    float* Y         = (float*)outputs[0];

    int64_t B = p->batch_size;
    int64_t S = p->seq_len;
    int64_t D = p->hidden_size;
    int64_t H = p->num_heads;
    int64_t d = p->head_dim;

    /* Allocate temp buffers for Q, K, V */
    float* Q = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* Kbuf = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* V = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* attn = (float*)calloc((size_t)B * H * S * d, sizeof(float));
    if (!Q || !Kbuf || !V || !attn) {
        free(Q); free(Kbuf); free(V); free(attn);
        return -1;
    }

    /* 1. QKV projections: (B, S, D) × (D, D) → (B, S, D) */
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            const float* x_bs = X + (b * S + s) * D;
            float* q_bs = Q + (b * S + s) * D;
            float* k_bs = Kbuf + (b * S + s) * D;
            float* v_bs = V + (b * S + s) * D;

            for (int64_t j = 0; j < D; j++) {
                float sq = 0.0f, sk = 0.0f, sv = 0.0f;
                for (int64_t i = 0; i < D; i++) {
                    float xv = x_bs[i];
                    sq += xv * WQ[i * D + j];
                    sk += xv * WK[i * D + j];
                    sv += xv * WV[i * D + j];
                }
                q_bs[j] = sq + (bQ ? bQ[j] : 0.0f);
                k_bs[j] = sk + (bK ? bK[j] : 0.0f);
                v_bs[j] = sv + (bV ? bV[j] : 0.0f);
            }
        }
    }

    /* 2-3. Multi-head split + attention per head per batch */
    float scale = p->scale;
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            /* Extract Q_h, K_h, V_h: (S, d) slices */
            float* attn_bh = attn + (b * H + h) * S * d;

            /* 3. scores = Q_h · K_h^T / scale, then softmax */
            for (int64_t si = 0; si < S; si++) {
                /* Compute scores for row si */
                float max_score = -1e38f;
                float* scores_row = (float*)calloc((size_t)S, sizeof(float));
                if (!scores_row) { free(Q); free(Kbuf); free(V); free(attn); return -1; }

                for (int64_t sj = 0; sj < S; sj++) {
                    float dot = 0.0f;
                    for (int64_t di = 0; di < d; di++) {
                        float qi = Q[(b * S + si) * D + h * d + di];
                        float kj = Kbuf[(b * S + sj) * D + h * d + di];
                        dot += qi * kj;
                    }
                    scores_row[sj] = dot * scale;
                    if (scores_row[sj] > max_score) max_score = scores_row[sj];
                }

                /* Softmax: exp + normalize */
                float sum = 0.0f;
                for (int64_t sj = 0; sj < S; sj++) {
                    scores_row[sj] = expf(scores_row[sj] - max_score);
                    sum += scores_row[sj];
                }
                if (sum < 1e-12f) sum = 1e-12f;
                float inv_sum = 1.0f / sum;

                /* 4. Weighted sum: attn_bh[si, :] = sum_j probs[j] * V_h[j, :] */
                for (int64_t di = 0; di < d; di++) {
                    float acc = 0.0f;
                    for (int64_t sj = 0; sj < S; sj++) {
                        acc += scores_row[sj] * inv_sum * V[(b * S + sj) * D + h * d + di];
                    }
                    attn_bh[si * d + di] = acc;
                }
                free(scores_row);
            }
        }
    }

    /* 5. Merge heads: (B, H, S, d) → (B, S, H, d) → (B, S, D) */
    float* merged = (float*)calloc((size_t)B * S * D, sizeof(float));
    if (!merged) { free(Q); free(Kbuf); free(V); free(attn); return -1; }
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t di = 0; di < d; di++) {
                    merged[(b * S + s) * D + h * d + di] =
                        attn[(b * H + h) * S * d + s * d + di];
                }
            }
        }
    }

    /* 6. Output projection: merged · W_O + b_O */
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            float* y_bs = Y + (b * S + s) * D;
            const float* m_bs = merged + (b * S + s) * D;
            for (int64_t j = 0; j < D; j++) {
                float acc = 0.0f;
                for (int64_t i = 0; i < D; i++) {
                    acc += m_bs[i] * WO[i * D + j];
                }
                y_bs[j] = acc + (bO ? bO[j] : 0.0f);
            }

            /* 7. Residual add */
            if (p->has_residual && R) {
                const float* r_bs = R + (b * S + s) * D;
                for (int64_t j = 0; j < D; j++) {
                    y_bs[j] += r_bs[j];
                }
            }
        }
    }

    free(Q); free(Kbuf); free(V); free(attn); free(merged);
    return 0;
}

static const operator_registry_t s_mha_fused_reg = {
    .name      = "mha_fused_f32",
    .data_type = "f32",
    .func      = mha_fused_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_mha_fused_f32(void) {
    return operator_register(&s_mha_fused_reg);
}

/* FP16 MHA is CUDA-only (Tensor Core WMMA). CPU stub returns error. */
int mha_fused_f16(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)inputs; (void)outputs; (void)params; (void)stream;
    return -1;  /* FP16 not supported on CPU */
}
