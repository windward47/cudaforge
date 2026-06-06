#include "operator.h"
#include <math.h>

/* ============================================================
 * AVX2 优化路径
 * ============================================================ */
#if defined(USE_AVX2)
#include <immintrin.h>

/* 快速 exp 近似 (Schraudolph + 修正项，误差 < 1e-4) */
static inline __m256 fast_exp_avx2(__m256 x) {
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);

    /* 限制范围避免溢出 */
    __m256 max_x = _mm256_set1_ps(88.0f);
    __m256 min_x = _mm256_set1_ps(-88.0f);
    x = _mm256_min_ps(x, max_x);
    x = _mm256_max_ps(x, min_x);

    /* t = x / ln(2) */
    __m256 t = _mm256_mul_ps(x, inv_ln2);

    /* floor(t) */
    __m256i ti = _mm256_cvtps_epi32(t);
    __m256 tf = _mm256_cvtepi32_ps(ti);

    /* r = t - floor(t), r in [0, 1) */
    __m256 r = _mm256_sub_ps(t, tf);

    /* 多项式近似: 2^r ≈ 1 + r*(0.693147 + r*(0.240227 + r*(0.055494 + r*0.009678))) */
    __m256 c4 = _mm256_set1_ps(0.009678f);
    __m256 c3 = _mm256_set1_ps(0.055494f);
    __m256 c2 = _mm256_set1_ps(0.240227f);
    __m256 c1 = _mm256_set1_ps(0.693147f);

    __m256 p = _mm256_fmadd_ps(c4, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);
    p = _mm256_fmadd_ps(p, r, one);

    /* 乘以 2^floor(t) */
    __m256i exp_int = _mm256_add_epi32(ti, _mm256_set1_epi32(127));
    exp_int = _mm256_slli_epi32(exp_int, 23);
    __m256 scale = _mm256_castsi256_ps(exp_int);

    return _mm256_mul_ps(p, scale);
}

/* Sigmoid AVX2: 1 / (1 + exp(-x)) */
static void sigmoid_avx2(const float* in, float* out, int64_t n) {
    __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 exp_neg = fast_exp_avx2(neg_x);
        __m256 denom = _mm256_add_ps(one, exp_neg);
        _mm256_storeu_ps(out + i, _mm256_div_ps(one, denom));
    }
    for (; i < n; i++) out[i] = 1.0f / (1.0f + expf(-in[i]));
}

/* GELU AVX2 (tanh 近似) */
static void gelu_avx2(const float* in, float* out, int64_t n) {
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 c = _mm256_set1_ps(0.7978845608f);   /* sqrt(2/pi) */
    __m256 k = _mm256_set1_ps(0.044715f);

    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        __m256 inner = _mm256_mul_ps(c, _mm256_fmadd_ps(k, x3, x));
        /* tanh 近似: tanh(x) ≈ x * (27 + x^2) / (27 + 9*x^2) */
        __m256 inner2 = _mm256_mul_ps(inner, inner);
        __m256 s27 = _mm256_set1_ps(27.0f);
        __m256 s9 = _mm256_set1_ps(9.0f);
        __m256 tanh_approx = _mm256_mul_ps(inner,
            _mm256_div_ps(_mm256_add_ps(s27, inner2),
                          _mm256_fmadd_ps(s9, inner2, s27)));
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, _mm256_add_ps(one, tanh_approx)));
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < n; i++) {
        const float sqrt_2_over_pi = 0.7978845608f;
        float x = in[i];
        float x3 = x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
    }
}

/* SiLU AVX2: x * sigmoid(x) */
static void silu_avx2(const float* in, float* out, int64_t n) {
    __m256 one = _mm256_set1_ps(1.0f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        __m256 exp_neg = fast_exp_avx2(neg_x);
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(x, sigmoid));
    }
    for (; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

/* Exp AVX2 */
static void exp_avx2(const float* in, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, fast_exp_avx2(x));
    }
    for (; i < n; i++) out[i] = expf(in[i]);
}

#endif /* USE_AVX2 */

/* ============================================================
 * 标量 fallback
 * ============================================================ */

/* Sigmoid: 1 / (1 + exp(-x)) */
int sigmoid_f32(const void* inputs[], void* outputs[],
                const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

#if defined(USE_AVX2)
    sigmoid_avx2(in, out, n);
#else
    for (int64_t i = 0; i < n; i++) {
        out[i] = 1.0f / (1.0f + expf(-in[i]));
    }
#endif
    return 0;
}

/* GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
static inline float gelu_tanh(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x3)));
}

int gelu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

#if defined(USE_AVX2)
    gelu_avx2(in, out, n);
#else
    for (int64_t i = 0; i < n; i++) {
        out[i] = gelu_tanh(in[i]);
    }
#endif
    return 0;
}

/* SiLU (Sigmoid Linear Unit): x * sigmoid(x) */
int silu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

#if defined(USE_AVX2)
    silu_avx2(in, out, n);
#else
    for (int64_t i = 0; i < n; i++) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
#endif
    return 0;
}

static const operator_registry_t s_sigmoid_reg = {
    .name = "sigmoid_f32", .data_type = "f32",
    .func = sigmoid_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

static const operator_registry_t s_gelu_reg = {
    .name = "gelu_f32", .data_type = "f32",
    .func = gelu_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

/* Exp: y = exp(x) */
int exp_f32(const void* inputs[], void* outputs[],
            const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];

#if defined(USE_AVX2)
    exp_avx2(in, out, n);
#else
    for (int64_t i = 0; i < n; i++) {
        out[i] = expf(in[i]);
    }
#endif
    return 0;
}

static const operator_registry_t s_silu_reg = {
    .name = "silu_f32", .data_type = "f32",
    .func = silu_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

static const operator_registry_t s_exp_reg = {
    .name = "exp_f32", .data_type = "f32",
    .func = exp_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

/* Tanh */
int tanh_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params; (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    int64_t n       = *(const int64_t*)inputs[1];
    for (int64_t i = 0; i < n; i++) out[i] = tanhf(in[i]);
    return 0;
}

static const operator_registry_t s_tanh_reg = {
    .name = "tanh_f32", .data_type = "f32",
    .func = tanh_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

int register_activations(void) {
    int ret = 0;
    ret += operator_register(&s_sigmoid_reg);
    ret += operator_register(&s_gelu_reg);
    ret += operator_register(&s_silu_reg);
    ret += operator_register(&s_exp_reg);
    ret += operator_register(&s_tanh_reg);
    return ret;
}
