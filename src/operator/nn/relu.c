#include "operator.h"
#include <math.h>
#include <stddef.h>

/* ============================================================
 * SIMD 优化路径: AVX-512 > AVX2 > 标量
 * ============================================================ */
#if defined(USE_AVX512)
#include <immintrin.h>
#define RELU_HAS_AVX512 1
#elif defined(USE_AVX2)
#include <immintrin.h>
#endif

/* ============================================================
 * 标量 fallback
 * ============================================================ */
static int relu_f32_scalar(const float* in, float* out, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        out[i] = fmaxf(in[i], 0.0f);
    }
    return 0;
}

#if defined(USE_AVX2) && !defined(RELU_HAS_AVX512)
static int relu_f32_avx2(const float* in, float* out, int64_t n) {
    __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; i++) {
        out[i] = fmaxf(in[i], 0.0f);
    }
    return 0;
}
#endif

/* ============================================================
 * 入口：ReLU y = max(0, x)
 * ============================================================ */
int relu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params;
    (void)stream;

    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];
    int64_t n        = *(const int64_t*)inputs[1];

#ifdef RELU_HAS_AVX512
    __m512 vzero = _mm512_setzero_ps();
    int64_t i = 0;
    for (; i + 15 < n; i += 16) {
        __m512 v = _mm512_loadu_ps(in + i);
        _mm512_storeu_ps(out + i, _mm512_max_ps(v, vzero));
    }
    for (; i < n; i++)
        out[i] = fmaxf(in[i], 0.0f);
#elif defined(USE_AVX2)
    return relu_f32_avx2(in, out, n);
#else
    return relu_f32_scalar(in, out, n);
#endif
    return 0;
}

static const operator_registry_t s_relu_reg = {
    .name      = "relu_f32",
    .data_type = "f32",
    .func      = relu_f32,

    .version   = 1,
    .flags     = OP_FLAG_IN_PLACE,
};

int register_relu_f32(void) {
    return operator_register(&s_relu_reg);
}
