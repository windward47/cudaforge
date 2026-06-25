#include "operator.h"
#include "layernorm_int.h"
#include <math.h>
#include <stddef.h>

/* ============================================================
 * SIMD 优化路径: AVX-512 > AVX2 > 标量
 * ============================================================ */
#if defined(USE_AVX512)
#include <immintrin.h>
#define LN_HAS_AVX512 1
#elif defined(USE_AVX2)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- AVX2 helpers (only when AVX2 without AVX512) ---- */
#if defined(USE_AVX2) && !defined(LN_HAS_AVX512)

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

static void layernorm_row_avx2(const float* x, float* y,
                                const float* gamma, const float* beta,
                                int64_t D, float eps) {
    /* 1. Mean */
    __m256 vsum = _mm256_setzero_ps();
    int64_t d = 0;
    for (; d + 8 <= D; d += 8) {
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(x + d));
    }
    float mean = hsum_avx2(vsum);
    for (; d < D; d++) mean += x[d];
    mean /= (float)D;

    /* 2. Variance */
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 vx = _mm256_loadu_ps(x + d);
        __m256 diff = _mm256_sub_ps(vx, vmean);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    float var = hsum_avx2(vvar);
    for (; d < D; d++) {
        float diff = x[d] - mean;
        var += diff * diff;
    }
    var = var / (float)D + eps;
    float inv_std = 1.0f / sqrtf(var);

    /* 3. Normalize + affine */
    __m256 vinv = _mm256_set1_ps(inv_std);
    d = 0;
    for (; d + 8 <= D; d += 8) {
        __m256 vx = _mm256_loadu_ps(x + d);
        __m256 vg = _mm256_loadu_ps(gamma + d);
        __m256 vb = _mm256_loadu_ps(beta + d);
        __m256 norm = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), vinv);
        _mm256_storeu_ps(y + d, _mm256_fmadd_ps(norm, vg, vb));
    }
    for (; d < D; d++) {
        y[d] = (x[d] - mean) * inv_std * gamma[d] + beta[d];
    }
}

#endif /* USE_AVX2 && !LN_HAS_AVX512 */

/* ============================================================
 * 入口
 * ============================================================ */
int layernorm_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const layernorm_params_t* p = (const layernorm_params_t*)params;
    const float* x     = (const float*)inputs[0];
    const float* gamma = (const float*)inputs[1];
    const float* beta  = (const float*)inputs[2];
    float* y           = (float*)outputs[0];

    int64_t N  = p->N;
    int64_t D  = p->normalized_size;
    float eps  = p->epsilon;

#pragma omp parallel for schedule(static) if(N > 1)
    for (int64_t n = 0; n < N; n++) {
        const float* xn = x + n * D;
        float*       yn = y + n * D;

#ifdef LN_HAS_AVX512
        /* Mean — AVX-512 */
        __m512 vsum = _mm512_setzero_ps();
        int64_t d = 0;
        for (; d + 15 < D; d += 16)
            vsum = _mm512_add_ps(vsum, _mm512_loadu_ps(xn + d));
        float mean = _mm512_reduce_add_ps(vsum);
        for (; d < D; d++) mean += xn[d];
        mean /= (float)D;

        /* Variance — AVX-512 */
        __m512 vmean = _mm512_set1_ps(mean);
        __m512 vvar = _mm512_setzero_ps();
        d = 0;
        for (; d + 15 < D; d += 16) {
            __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(xn + d), vmean);
            vvar = _mm512_fmadd_ps(diff, diff, vvar);
        }
        float var = _mm512_reduce_add_ps(vvar);
        for (; d < D; d++) { float diff = xn[d] - mean; var += diff * diff; }
        var = var / (float)D + eps;
        float inv_std = 1.0f / sqrtf(var);

        /* Normalize: y = (x - mean) * inv_std * gamma + beta — AVX-512 */
        __m512 vinv = _mm512_set1_ps(inv_std);
        d = 0;
        for (; d + 15 < D; d += 16) {
            __m512 vx = _mm512_loadu_ps(xn + d);
            __m512 vg = _mm512_loadu_ps(gamma + d);
            __m512 vb = _mm512_loadu_ps(beta + d);
            __m512 norm = _mm512_mul_ps(_mm512_sub_ps(vx, vmean), vinv);
            _mm512_storeu_ps(yn + d, _mm512_add_ps(_mm512_mul_ps(norm, vg), vb));
        }
        for (; d < D; d++)
            yn[d] = (xn[d] - mean) * inv_std * gamma[d] + beta[d];
#elif defined(USE_AVX2)
        layernorm_row_avx2(xn, yn, gamma, beta, D, eps);
#else
        /* Scalar fallback */
        float mean = 0.0f;
        for (int64_t d = 0; d < D; d++) mean += xn[d];
        mean /= (float)D;

        float var = 0.0f;
        for (int64_t d = 0; d < D; d++) {
            float diff = xn[d] - mean;
            var += diff * diff;
        }
        var = var / (float)D + eps;
        float inv_std = 1.0f / sqrtf(var);

        for (int64_t d = 0; d < D; d++)
            yn[d] = (xn[d] - mean) * inv_std * gamma[d] + beta[d];
#endif
    }
    return 0;
}

static const operator_registry_t s_layernorm_reg = {
    .name      = "layernorm_f32",
    .data_type = "f32",
    .func      = layernorm_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_layernorm_f32(void) {
    return operator_register(&s_layernorm_reg);
}
