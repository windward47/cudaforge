#include "operator.h"
#include "reduce_int.h"
#include <math.h>

/* ============================================================
 * AVX2 优化路径
 * ============================================================ */
#if defined(USE_AVX2)
#include <immintrin.h>

static inline float hsum_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(s);
    s = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, s);
    s = _mm_add_ps(s, shuf);
    return _mm_cvtss_f32(s);
}

static inline float hmax_avx2(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(m);
    m = _mm_max_ps(m, shuf);
    shuf = _mm_movehl_ps(shuf, m);
    m = _mm_max_ps(m, shuf);
    return _mm_cvtss_f32(m);
}

#endif /* USE_AVX2 */

/* ============================================================
 * 入口
 * ============================================================ */
int reduce_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    const reduce_params_t* rp = (const reduce_params_t*)params;

    int64_t reduce_size = rp->reduce_size;
    int64_t num_blocks  = rp->num_blocks;
    int     op          = rp->op;

    if (op == REDUCE_SUM) {
        for (int64_t b = 0; b < num_blocks; b++) {
            const float* block_in = in + b * reduce_size;
#if defined(USE_AVX2)
            __m256 vsum = _mm256_setzero_ps();
            int64_t i = 0;
            for (; i + 8 <= reduce_size; i += 8) {
                vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(block_in + i));
            }
            float sum = hsum_avx2(vsum);
            for (; i < reduce_size; i++) sum += block_in[i];
            out[b] = sum;
#else
            float sum = 0.0f;
            for (int64_t i = 0; i < reduce_size; i++) {
                sum += block_in[i];
            }
            out[b] = sum;
#endif
        }
    } else { /* REDUCE_MAX */
        for (int64_t b = 0; b < num_blocks; b++) {
            const float* block_in = in + b * reduce_size;
#if defined(USE_AVX2)
            float max_val = block_in[0];
            __m256 vmax = _mm256_set1_ps(max_val);
            int64_t i = 0;
            for (; i + 8 <= reduce_size; i += 8) {
                vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(block_in + i));
            }
            max_val = hmax_avx2(vmax);
            for (; i < reduce_size; i++) {
                if (block_in[i] > max_val) max_val = block_in[i];
            }
            out[b] = max_val;
#else
            float max_val = block_in[0];
            for (int64_t i = 1; i < reduce_size; i++) {
                if (block_in[i] > max_val) max_val = block_in[i];
            }
            out[b] = max_val;
#endif
        }
    }
    return 0;
}

static const operator_registry_t s_reduce_reg = {
    .name = "reduce_f32", .data_type = "f32",
    .func = reduce_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_reduce_f32(void) {
    return operator_register(&s_reduce_reg);
}
