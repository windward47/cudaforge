#include "operator.h"
#include "add_int.h"
#include <stddef.h>

/* ============================================================
 * AVX2 优化路径
 * ============================================================ */
#if defined(USE_AVX2)
#include <immintrin.h>

/* 标量 broadcast: out[i] = a[i] + b_scalar */
static void add_avx2_scalar(const float* a, float b_scalar, float* out, int64_t n) {
    __m256 vb = _mm256_set1_ps(b_scalar);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b_scalar;
}

/* 逐元素: out[i] = a[i] + b[i] */
static void add_avx2_vec(const float* a, const float* b, float* out, int64_t n) {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}

#endif /* USE_AVX2 */

/* ============================================================
 * 入口
 * ============================================================ */
int add_f32(const void* inputs[], void* outputs[],
            const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const add_params_t* p = (const add_params_t*)params;
    const float* a = (const float*)inputs[0];
    const float* b = (const float*)inputs[1];
    float* out = (float*)outputs[0];

    int64_t N = p->numel;
    int64_t BN = p->B_numel;

    if (BN == 1) {
        float bv = b[0];
#if defined(USE_AVX2)
        add_avx2_scalar(a, bv, out, N);
#else
        for (int64_t i = 0; i < N; i++) out[i] = a[i] + bv;
#endif
    } else {
        int64_t blocks = N / BN;
        for (int64_t blk = 0; blk < blocks; blk++) {
            const float* a_blk = a + blk * BN;
            float* out_blk = out + blk * BN;
#if defined(USE_AVX2)
            add_avx2_vec(a_blk, b, out_blk, BN);
#else
            for (int64_t i = 0; i < BN; i++) out_blk[i] = a_blk[i] + b[i];
#endif
        }
    }
    return 0;
}

static const operator_registry_t s_add_reg = {
    .name      = "add_f32",
    .data_type = "f32",
    .func      = add_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_add_f32(void) {
    return operator_register(&s_add_reg);
}
