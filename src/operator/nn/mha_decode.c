/* CPU reference for MHA decode — single-token attention with KV-cache.
   Supports GQA (Grouped-Query Attention): H_kv <= H_q.
   inputs:  [X_new(B,1,D), K_cache(B,max_seq,H_kv,d), V_cache(B,max_seq,H_kv,d),
             WQ(D,D), bQ(D), WK(D,H_kv*d), bK(H_kv*d), WV(D,H_kv*d), bV(H_kv*d), WO(D,D), bO(D)]
   outputs: [Y(B,1,D), K_cache_out, V_cache_out] */
#include "operator.h"
#include "mha_decode_int.h"
#include <math.h>
#include <string.h>

int mha_decode_f32(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!params) return -1;
    for (int i = 0; i < 11; i++) if (!inputs[i]) return -1;
    for (int i = 0; i < 3; i++)  if (!outputs[i]) return -1;

    const mha_decode_params_t* p = (const mha_decode_params_t*)params;
    int64_t B = p->batch_size, D = p->hidden_size;
    int64_t H = p->num_heads, d = p->head_dim;
    int64_t H_kv = p->num_kv_heads > 0 ? p->num_kv_heads : H;
    int64_t cache_len = p->cache_len, max_seq = p->max_seq;
    int64_t kv_dim = H_kv * d;  /* WK/WV output dimension */
    int64_t group_size = H / H_kv;  /* query heads per KV head */

    const float* X_new   = (const float*)inputs[0];   /* (B, 1, D) */
    const float* K_cache = (const float*)inputs[1];    /* (B, max_seq, H_kv, d) */
    const float* V_cache = (const float*)inputs[2];    /* (B, max_seq, H_kv, d) */
    const float* WQ = (const float*)inputs[3];
    const float* bQ = (const float*)inputs[4];
    const float* WK = (const float*)inputs[5];
    const float* bK = (const float*)inputs[6];
    const float* WV = (const float*)inputs[7];
    const float* bV = (const float*)inputs[8];
    const float* WO = (const float*)inputs[9];
    const float* bO = (const float*)inputs[10];

    float* Y          = (float*)outputs[0];  /* (B, 1, D) */
    float* K_cache_out = (float*)outputs[1]; /* updated cache */
    float* V_cache_out = (float*)outputs[2]; /* updated cache */

    /* Copy caches to output (in-place update) */
    memcpy(K_cache_out, K_cache, (size_t)B * max_seq * H_kv * d * sizeof(float));
    memcpy(V_cache_out, V_cache, (size_t)B * max_seq * H_kv * d * sizeof(float));

    for (int64_t b = 0; b < B; b++) {
        const float* x = X_new + b * D;  /* (D,) */
        float* y = Y + b * D;             /* (D,) */

        /* Initialize Y with bias */
        for (int64_t j = 0; j < D; j++) y[j] = bO ? bO[j] : 0.0f;

        /* Compute K_new, V_new once per KV head (shared across group) */
        for (int64_t kv_h = 0; kv_h < H_kv; kv_h++) {
            int64_t kv_ho = kv_h * d;
            float K_new[64], V_new[64];
            for (int64_t di = 0; di < d; di++) {
                float k_acc = 0.0f, v_acc = 0.0f;
                for (int64_t j = 0; j < D; j++) {
                    k_acc += x[j] * WK[j * kv_dim + kv_ho + di];
                    v_acc += x[j] * WV[j * kv_dim + kv_ho + di];
                }
                K_new[di] = k_acc + (bK ? bK[kv_ho + di] : 0.0f);
                V_new[di] = v_acc + (bV ? bV[kv_ho + di] : 0.0f);
            }
            /* Write to cache */
            int64_t cache_idx = (b * max_seq + cache_len) * H_kv * d + kv_ho;
            for (int64_t di = 0; di < d; di++) {
                K_cache_out[cache_idx + di] = K_new[di];
                V_cache_out[cache_idx + di] = V_new[di];
            }
        }

        /* Process each query head */
        for (int64_t h = 0; h < H; h++) {
            int64_t ho = h * d;
            int64_t kv_h = h / group_size;  /* GQA: map query head → KV head */
            int64_t kv_ho = kv_h * d;

            /* Q = x · WQ + bQ  →  (d,) */
            float Q[64];
            for (int64_t di = 0; di < d; di++) {
                float acc = 0.0f;
                for (int64_t j = 0; j < D; j++) acc += x[j] * WQ[j * D + ho + di];
                Q[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
            }

            /* Attention scores: Q · K_cache^T · scale, positions 0..cache_len */
            int64_t total_len = cache_len + 1;
            float scores[512];  /* max_seq */
            float max_score = -1e38f;
            for (int64_t t = 0; t < total_len; t++) {
                float dot = 0.0f;
                int64_t k_off = (b * max_seq + t) * H_kv * d + kv_ho;
                for (int64_t di = 0; di < d; di++) dot += Q[di] * K_cache_out[k_off + di];
                scores[t] = dot * p->scale;
                if (scores[t] > max_score) max_score = scores[t];
            }

            /* Softmax */
            float sum_exp = 0.0f;
            for (int64_t t = 0; t < total_len; t++) {
                scores[t] = expf(scores[t] - max_score);
                sum_exp += scores[t];
            }
            if (sum_exp < 1e-12f) sum_exp = 1e-12f;

            /* Weighted V sum */
            float merged[64] = {0};
            for (int64_t t = 0; t < total_len; t++) {
                float w = scores[t] / sum_exp;
                int64_t v_off = (b * max_seq + t) * H_kv * d + kv_ho;
                for (int64_t di = 0; di < d; di++) merged[di] += w * V_cache_out[v_off + di];
            }

            /* Output projection: Y += merged · WO */
            for (int64_t j = 0; j < D; j++) {
                float contrib = 0.0f;
                for (int64_t di = 0; di < d; di++) contrib += merged[di] * WO[(ho + di) * D + j];
                y[j] += contrib;
            }
        }
    }
    return 0;
}

int register_mha_decode_f32(void) {
    static operator_registry_t reg = {
        .name = "mha_decode_f32", .data_type = "f32",
        .func = mha_decode_f32, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
