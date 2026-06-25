#include "operator.h"
#include "matmul_int.h"
#include <stddef.h>

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#include <immintrin.h>
#define MATMUL_HAS_AVX512 1
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ==========================================================================
 *  AVX-512 + OpenMP optimized matmul (non-transposed path)
 *  - Inner loop over K is vectorized: 4×ZMM accumulators = 64 floats/iter
 *  - Outer loop over M rows is parallelized with OpenMP
 * ========================================================================== */
#ifdef MATMUL_HAS_AVX512
static void matmul_avx512(const float* A, const float* B, const float* bias,
                          float* C, int64_t M, int64_t N, int64_t K) {
#pragma omp parallel for schedule(static) if(M > 1)
    for (int64_t i = 0; i < M; i++) {
        const float* Ai = A + i * K;
        float* Ci = C + i * N;

        int64_t j = 0;
        /* Main loop: 64 columns at a time (4 × 16-wide AVX-512 registers) */
        for (; j + 63 < N; j += 64) {
            __m512 c0 = _mm512_setzero_ps();
            __m512 c1 = _mm512_setzero_ps();
            __m512 c2 = _mm512_setzero_ps();
            __m512 c3 = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; k++) {
                __m512 a = _mm512_set1_ps(Ai[k]);
                const float* Bk = B + k * N + j;
                c0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(Bk),      c0);
                c1 = _mm512_fmadd_ps(a, _mm512_loadu_ps(Bk + 16), c1);
                c2 = _mm512_fmadd_ps(a, _mm512_loadu_ps(Bk + 32), c2);
                c3 = _mm512_fmadd_ps(a, _mm512_loadu_ps(Bk + 48), c3);
            }
            if (bias) {
                const float* bj = bias + j;
                __m512 b0 = _mm512_loadu_ps(bj);
                __m512 b1 = _mm512_loadu_ps(bj + 16);
                __m512 b2 = _mm512_loadu_ps(bj + 32);
                __m512 b3 = _mm512_loadu_ps(bj + 48);
                c0 = _mm512_add_ps(c0, b0);
                c1 = _mm512_add_ps(c1, b1);
                c2 = _mm512_add_ps(c2, b2);
                c3 = _mm512_add_ps(c3, b3);
            }
            _mm512_storeu_ps(Ci + j,      c0);
            _mm512_storeu_ps(Ci + j + 16, c1);
            _mm512_storeu_ps(Ci + j + 32, c2);
            _mm512_storeu_ps(Ci + j + 48, c3);
        }
        /* Remainder: 16 columns at a time */
        for (; j + 15 < N; j += 16) {
            __m512 c0 = _mm512_setzero_ps();
            for (int64_t k = 0; k < K; k++) {
                __m512 a = _mm512_set1_ps(Ai[k]);
                c0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(B + k * N + j), c0);
            }
            if (bias) c0 = _mm512_add_ps(c0, _mm512_loadu_ps(bias + j));
            _mm512_storeu_ps(Ci + j, c0);
        }
        /* Scalar tail: remaining columns */
        for (; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++)
                sum += Ai[k] * B[k * N + j];
            Ci[j] = sum + (bias ? bias[j] : 0.0f);
        }
    }
}
#endif /* MATMUL_HAS_AVX512 */

/* ==========================================================================
 *  matmul_f32 — entry point with dispatch
 * ========================================================================== */
/* C = A * B (or transposed variants)
   A: M×K, B: K×N, C: M×N
   inputs[0] = A (float*), inputs[1] = B (float*)
   outputs[0] = C (float*)
   params = matmul_params_t*

   Supports batched operation: when batch_size > 1, each batch slice
   is an independent M×K @ K×N → M×N matmul, with stride_a/b/c
   giving the element offset between consecutive slices.
*/
int matmul_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;

    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const matmul_params_t* p = (const matmul_params_t*)params;

    const float* A    = (const float*)inputs[0];
    const float* B    = (const float*)inputs[1];
    const float* bias = (const float*)inputs[2];
    float* C          = (float*)outputs[0];

    int64_t M = p->M, N = p->N, K = p->K;
    int64_t batch_size = p->batch_size > 0 ? p->batch_size : 1;
    int64_t stride_a = p->stride_a;
    int64_t stride_b = p->stride_b;
    int64_t stride_c = p->stride_c;

    for (int64_t bi = 0; bi < batch_size; bi++) {
        const float* Ab = A + bi * stride_a;
        const float* Bb = B + bi * stride_b;
        float*       Cb = C + bi * stride_c;

#ifdef MATMUL_HAS_AVX512
        /* Fast path: non-transposed, AVX-512 + OpenMP */
        if (!p->transpose_a && !p->transpose_b) {
            matmul_avx512(Ab, Bb, bias, Cb, M, N, K);
            continue;
        }
#endif

        /* Scalar fallback: transposed variants */
        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    float av = p->transpose_a ? Ab[k * M + i] : Ab[i * K + k];
                    float bv = p->transpose_b ? Bb[j * K + k] : Bb[k * N + j];
                    sum += av * bv;
                }
                Cb[i * N + j] = sum + (bias ? bias[j] : 0.0f);
            }
        }
    }
    return 0;
}

static const operator_registry_t s_matmul_reg = {
    .name      = "matmul_f32",
    .data_type = "f32",
    .func      = matmul_f32,

    .version   = 2,
    .flags     = OP_FLAG_NONE,
};

int register_matmul_f32(void) {
    return operator_register(&s_matmul_reg);
}
