#include "operator.h"
#include "softmax_int.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ============================================================
 * SIMD 优化路径: AVX-512 > AVX2 > 标量
 * ============================================================ */
#if defined(USE_AVX512)
#include <immintrin.h>
#define SOFTMAX_HAS_AVX512 1
#elif defined(USE_AVX2)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- AVX2 helpers (only when AVX2 without AVX512) ---- */
#if defined(USE_AVX2) && !defined(SOFTMAX_HAS_AVX512)

static inline __m256 fast_exp_avx2(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);

    __m256 max_x = _mm256_set1_ps(88.0f);
    __m256 min_x = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    __m256 t = _mm256_mul_ps(x, inv_ln2);
    __m256i ti = _mm256_cvtps_epi32(t);
    __m256 tf = _mm256_cvtepi32_ps(ti);
    __m256 r = _mm256_sub_ps(t, tf);

    __m256 c4 = _mm256_set1_ps(0.009678f);
    __m256 c3 = _mm256_set1_ps(0.055494f);
    __m256 c2 = _mm256_set1_ps(0.240227f);
    __m256 c1 = _mm256_set1_ps(0.693147f);

    __m256 p = _mm256_fmadd_ps(c4, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, one);

    __m256i exp_int = _mm256_add_epi32(ti, _mm256_set1_epi32(127));
    exp_int = _mm256_slli_epi32(exp_int, 23);
    __m256 scale = _mm256_castsi256_ps(exp_int);

    return _mm256_mul_ps(p, scale);
}

static void softmax_row_avx2(const float* in, float* out, int64_t C) {
    /* 1. Find max */
    __m256 vmax = _mm256_loadu_ps(in);
    int64_t c = 8;
    for (; c + 8 <= C; c += 8) {
        __m256 v = _mm256_loadu_ps(in + c);
        vmax = _mm256_max_ps(vmax, v);
    }
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m128 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(m128);
    m128 = _mm_max_ps(m128, shuf);
    shuf = _mm_movehl_ps(shuf, m128);
    m128 = _mm_max_ps(m128, shuf);
    float max_val = _mm_cvtss_f32(m128);

    for (; c < C; c++) {
        if (in[c] > max_val) max_val = in[c];
    }

    /* 2. Compute exp(x - max) and sum */
    __m256 vmax8 = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    c = 0;
    for (; c + 8 <= C; c += 8) {
        __m256 v = _mm256_loadu_ps(in + c);
        __m256 shifted = _mm256_sub_ps(v, vmax8);
        __m256 exp_v = fast_exp_avx2(shifted);
        _mm256_storeu_ps(out + c, exp_v);
        vsum = _mm256_add_ps(vsum, exp_v);
    }
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    __m128 s128 = _mm_add_ps(lo, hi);
    shuf = _mm_movehdup_ps(s128);
    s128 = _mm_add_ps(s128, shuf);
    shuf = _mm_movehl_ps(shuf, s128);
    s128 = _mm_add_ps(s128, shuf);
    float sum = _mm_cvtss_f32(s128);

    for (; c < C; c++) {
        out[c] = expf(in[c] - max_val);
        sum += out[c];
    }

    /* 3. Normalize */
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
    __m256 vinv = _mm256_set1_ps(inv_sum);
    c = 0;
    for (; c + 8 <= C; c += 8) {
        __m256 v = _mm256_loadu_ps(out + c);
        _mm256_storeu_ps(out + c, _mm256_mul_ps(v, vinv));
    }
    for (; c < C; c++) {
        out[c] *= inv_sum;
    }
}

#endif /* USE_AVX2 && !SOFTMAX_HAS_AVX512 */

/* ---- AVX-512 helpers ---- */
#ifdef SOFTMAX_HAS_AVX512

static inline float avx512_max_reduce(const float* data, int64_t n) {
    if (n < 16) {
        float m = data[0];
        for (int64_t i = 1; i < n; i++)
            if (data[i] > m) m = data[i];
        return m;
    }
    __m512 vmax = _mm512_loadu_ps(data);
    int64_t i = 16;
    for (; i + 15 < n; i += 16)
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(data + i));
    float m = _mm512_reduce_max_ps(vmax);
    for (; i < n; i++)
        if (data[i] > m) m = data[i];
    return m;
}

static inline float avx512_sum_reduce(const float* data, int64_t n) {
    if (n < 16) {
        float s = 0.0f;
        for (int64_t i = 0; i < n; i++) s += data[i];
        return s;
    }
    __m512 vsum = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < n; i += 16)
        vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(data + i));
    float s = _mm512_reduce_add_ps(vsum);
    for (; i < n; i++) s += data[i];
    return s;
}

static inline void avx512_scale(float* data, int64_t n, float scale) {
    __m512 vs = _mm512_set1_ps(scale);
    int64_t i = 0;
    for (; i + 15 < n; i += 16)
        _mm512_storeu_ps(data + i, _mm512_mul_ps(_mm512_loadu_ps(data + i), vs));
    for (; i < n; i++) data[i] *= scale;
}

static inline void avx512_sub_scalar(float* dst, const float* src,
                                     int64_t n, float val) {
    __m512 vv = _mm512_set1_ps(val);
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_sub_ps(_mm512_loadu_ps(src + i), vv);
        _mm512_storeu_ps(dst + i, v);
    }
    for (; i < n; i++) dst[i] = src[i] - val;
}

#endif /* SOFTMAX_HAS_AVX512 */

/* ============================================================
 * 入口
 * ============================================================ */
int softmax_f32(const void* inputs[], void* outputs[],
                const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const softmax_params_t* p = (const softmax_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    int64_t C = p->num_classes;
    int64_t N = p->num_blocks;

#pragma omp parallel for schedule(static) if(N > 1)
    for (int64_t n = 0; n < N; n++) {
        const float* in_n = in + n * C;
        float* out_n = out + n * C;

#ifdef SOFTMAX_HAS_AVX512
        /* Find max — AVX-512 reduction */
        float max_val = avx512_max_reduce(in_n, C);

        /* Subtract max and compute exp */
        avx512_sub_scalar(out_n, in_n, C, max_val);
        for (int64_t c = 0; c < C; c++)
            out_n[c] = expf(out_n[c]);

        /* Sum and normalize — AVX-512 */
        float sum = avx512_sum_reduce(out_n, C);
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        avx512_scale(out_n, C, inv_sum);
#elif defined(USE_AVX2)
        softmax_row_avx2(in_n, out_n, C);
#else
        /* Scalar fallback */
        float max_val = in_n[0];
        for (int64_t c = 1; c < C; c++) {
            if (in_n[c] > max_val) max_val = in_n[c];
        }

        float sum = 0.0f;
        for (int64_t c = 0; c < C; c++) {
            out_n[c] = expf(in_n[c] - max_val);
            sum += out_n[c];
        }

        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int64_t c = 0; c < C; c++)
            out_n[c] *= inv_sum;
#endif
    }
    return 0;
}

static const operator_registry_t s_softmax_reg = {
    .name      = "softmax_f32",
    .data_type = "f32",
    .func      = softmax_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_softmax_f32(void) {
    return operator_register(&s_softmax_reg);
}
