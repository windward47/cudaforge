#include "operator.h"
#include "mha_fused_int.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define MHA_HAS_AVX512 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* Max sequence length for stack-allocated score buffer */
#define MHA_STACK_SEQ 2048

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
    float scale = p->scale;

    float* Q = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* Kbuf = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* V = (float*)calloc((size_t)B * S * D, sizeof(float));
    float* attn = (float*)calloc((size_t)B * H * S * d, sizeof(float));
    float* merged = (float*)calloc((size_t)B * S * D, sizeof(float));
    if (!Q || !Kbuf || !V || !attn || !merged) {
        free(Q); free(Kbuf); free(V); free(attn); free(merged);
        return -1;
    }

    /* 1. QKV projections: (B, S, D) × (D, D) → (B, S, D)
       Parallelize over (b, s) — each row is independent */
#pragma omp parallel for collapse(2) schedule(static) if(B * S > 1)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            const float* x_bs = X + (b * S + s) * D;
            float* q_bs = Q + (b * S + s) * D;
            float* k_bs = Kbuf + (b * S + s) * D;
            float* v_bs = V + (b * S + s) * D;

#ifdef MHA_HAS_AVX512
            for (int64_t j = 0; j < D; j++) {
                __m512 vsum_q = _mm512_setzero_ps();
                __m512 vsum_k = _mm512_setzero_ps();
                __m512 vsum_v = _mm512_setzero_ps();
                int64_t i = 0;
                for (; i + 15 < D; i += 16) {
                    __m512 xv = _mm512_loadu_ps(x_bs + i);
                    vsum_q = _mm512_fmadd_ps(xv, _mm512_loadu_ps(WQ + i * D + j), vsum_q);
                    vsum_k = _mm512_fmadd_ps(xv, _mm512_loadu_ps(WK + i * D + j), vsum_k);
                    vsum_v = _mm512_fmadd_ps(xv, _mm512_loadu_ps(WV + i * D + j), vsum_v);
                }
                float sq = _mm512_reduce_add_ps(vsum_q);
                float sk = _mm512_reduce_add_ps(vsum_k);
                float sv = _mm512_reduce_add_ps(vsum_v);
                for (; i < D; i++) {
                    float xv = x_bs[i];
                    sq += xv * WQ[i * D + j];
                    sk += xv * WK[i * D + j];
                    sv += xv * WV[i * D + j];
                }
                q_bs[j] = sq + (bQ ? bQ[j] : 0.0f);
                k_bs[j] = sk + (bK ? bK[j] : 0.0f);
                v_bs[j] = sv + (bV ? bV[j] : 0.0f);
            }
#else
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
#endif
        }
    }

    /* 2-4. Multi-head attention: scores = Q·K^T/sqrt(d), softmax, attn = probs·V
       Parallelize over (b, h, si) — each row is independent */
#pragma omp parallel for collapse(3) schedule(static) if(B * H * S > 1)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t si = 0; si < S; si++) {
                /* Stack buffer for scores (avoids per-row calloc) */
                float scores_stack[MHA_STACK_SEQ];
                float* scores_row = (S <= MHA_STACK_SEQ)
                    ? scores_stack
                    : (float*)malloc((size_t)S * sizeof(float));

                /* Q_h · K_h^T */
                float max_score = -1e38f;
                for (int64_t sj = 0; sj < S; sj++) {
                    float dot = 0.0f;
                    const float* qi = Q + (b * S + si) * D + h * d;
                    const float* kj = Kbuf + (b * S + sj) * D + h * d;
#ifdef MHA_HAS_AVX512
                    __m512 vd = _mm512_setzero_ps();
                    int64_t di = 0;
                    for (; di + 15 < d; di += 16)
                        vd = _mm512_fmadd_ps(_mm512_loadu_ps(qi + di),
                                             _mm512_loadu_ps(kj + di), vd);
                    dot = _mm512_reduce_add_ps(vd);
                    for (; di < d; di++) dot += qi[di] * kj[di];
#else
                    for (int64_t di = 0; di < d; di++)
                        dot += qi[di] * kj[di];
#endif
                    scores_row[sj] = dot * scale;
                    if (scores_row[sj] > max_score) max_score = scores_row[sj];
                }

                /* Softmax */
                float sum = 0.0f;
                for (int64_t sj = 0; sj < S; sj++) {
                    scores_row[sj] = expf(scores_row[sj] - max_score);
                    sum += scores_row[sj];
                }
                float inv_sum = 1.0f / (sum > 1e-12f ? sum : 1e-12f);

                /* Weighted sum: attn = probs · V */
                float* attn_bh = attn + (b * H + h) * S * d;
                for (int64_t di = 0; di < d; di++) {
                    float acc = 0.0f;
                    for (int64_t sj = 0; sj < S; sj++)
                        acc += scores_row[sj] * inv_sum * V[(b * S + sj) * D + h * d + di];
                    attn_bh[si * d + di] = acc;
                }

                if (S > MHA_STACK_SEQ) free(scores_row);
            }
        }
    }

    /* 5. Merge heads: (B,H,S,d) → (B,S,H,d) → (B,S,D) */
#pragma omp parallel for collapse(2) schedule(static) if(B * S > 1)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t h = 0; h < H; h++) {
                memcpy(merged + (b * S + s) * D + h * d,
                       attn + (b * H + h) * S * d + s * d,
                       (size_t)d * sizeof(float));
            }
        }
    }

    /* 6. Output projection: merged · W_O + b_O, then residual add */
#pragma omp parallel for collapse(2) schedule(static) if(B * S > 1)
    for (int64_t b = 0; b < B; b++) {
        for (int64_t s = 0; s < S; s++) {
            float* y_bs = Y + (b * S + s) * D;
            const float* m_bs = merged + (b * S + s) * D;
            for (int64_t j = 0; j < D; j++) {
                float acc = 0.0f;
#ifdef MHA_HAS_AVX512
                __m512 va = _mm512_setzero_ps();
                int64_t i = 0;
                for (; i + 15 < D; i += 16)
                    va = _mm512_fmadd_ps(_mm512_loadu_ps(m_bs + i),
                                         _mm512_loadu_ps(WO + i * D + j), va);
                acc = _mm512_reduce_add_ps(va);
                for (; i < D; i++) acc += m_bs[i] * WO[i * D + j];
#else
                for (int64_t i = 0; i < D; i++)
                    acc += m_bs[i] * WO[i * D + j];
#endif
                y_bs[j] = acc + (bO ? bO[j] : 0.0f);
            }

            /* Residual add */
            if (p->has_residual && R) {
                const float* r_bs = R + (b * S + s) * D;
                for (int64_t j = 0; j < D; j++)
                    y_bs[j] += r_bs[j];
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
    .version   = 2,
    .flags     = OP_FLAG_NONE,
};

int register_mha_fused_f32(void) {
    return operator_register(&s_mha_fused_reg);
}

/* FP16 MHA is CUDA-only (Tensor Core WMMA). CPU stub returns error. */
int mha_fused_f16(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)inputs; (void)outputs; (void)params; (void)stream;
    return -1;
}

int register_mha_fused_f16(void) { return 0; }

