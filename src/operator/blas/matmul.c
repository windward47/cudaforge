#include "operator.h"
#include "matmul_int.h"
#include <stddef.h>

/* C = A * B (or transposed variants)
   A: M×K, B: K×N, C: M×N
   inputs[0] = A (float*), inputs[1] = B (float*)
   outputs[0] = C (float*)
   params = matmul_params_t*
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

    /* Naive triple loop */
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                float a = p->transpose_a ? A[k * M + i] : A[i * K + k];
                float b = p->transpose_b ? B[j * K + k] : B[k * N + j];
                sum += a * b;
            }
            C[i * N + j] = sum + (bias ? bias[j] : 0.0f);
        }
    }
    return 0;
}

static const operator_registry_t s_matmul_reg = {
    .name      = "matmul_f32",
    .data_type = "f32",
    .func      = matmul_f32,

    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_matmul_f32(void) {
    return operator_register(&s_matmul_reg);
}
