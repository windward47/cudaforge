#include "operator.h"
#include "softmax_int.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* ============================================================
 * AVX2 优化路径
 * ============================================================ */
#if defined(USE_AVX2)
#include <immintrin.h>

/* 快速 exp 近似 (复用 activations.c 中的实现) */
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

/* AVX2 softmax 单行处理 */
static void softmax_row_avx2(const float* in, float* out, int64_t C) {
    /* 1. 找最大值 (AVX2 水平归约) */
    __m256 vmax = _mm256_loadu_ps(in);
    int64_t c = 8;
    for (; c + 8 <= C; c += 8) {
        __m256 v = _mm256_loadu_ps(in + c);
        vmax = _mm256_max_ps(vmax, v);
    }
    /* 水平归约 */
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 m128 = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(m128);
    m128 = _mm_max_ps(m128, shuf);
    shuf = _mm_movehl_ps(shuf, m128);
    m128 = _mm_max_ps(m128, shuf);
    float max_val = _mm_cvtss_f32(m128);

    /* 标量尾部 */
    for (; c < C; c++) {
        if (in[c] > max_val) max_val = in[c];
    }

    /* 2. 计算 exp(x - max) 和 sum */
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
    /* 水平求和 */
    hi = _mm256_extractf128_ps(vsum, 1);
    lo = _mm256_castps256_ps128(vsum);
    __m128 s128 = _mm_add_ps(lo, hi);
    shuf = _mm_movehdup_ps(s128);
    s128 = _mm_add_ps(s128, shuf);
    shuf = _mm_movehl_ps(shuf, s128);
    s128 = _mm_add_ps(s128, shuf);
    float sum = _mm_cvtss_f32(s128);

    /* 标量尾部 */
    for (; c < C; c++) {
        out[c] = expf(in[c] - max_val);
        sum += out[c];
    }

    /* 3. 归一化 */
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

#endif /* USE_AVX2 */

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

    for (int64_t n = 0; n < N; n++) {
        const float* in_n = in + n * C;
        float* out_n = out + n * C;

#if defined(USE_AVX2)
        softmax_row_avx2(in_n, out_n, C);
#else
        /* Find max for numerical stability */
        float max_val = in_n[0];
        for (int64_t c = 1; c < C; c++) {
            if (in_n[c] > max_val) max_val = in_n[c];
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (int64_t c = 0; c < C; c++) {
            out_n[c] = expf(in_n[c] - max_val);
            sum += out_n[c];
        }

        /* Normalize */
        float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int64_t c = 0; c < C; c++) {
            out_n[c] *= inv_sum;
        }
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
