#include "operator.h"
#include "layernorm_int.h"
#include <math.h>
#include <stddef.h>

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

    for (int64_t n = 0; n < N; n++) {
        const float* xn = x + n * D;
        float*       yn = y + n * D;

        /* Mean */
        float mean = 0.0f;
        for (int64_t d = 0; d < D; d++) mean += xn[d];
        mean /= (float)D;

        /* Variance */
        float var = 0.0f;
        for (int64_t d = 0; d < D; d++) {
            float diff = xn[d] - mean;
            var += diff * diff;
        }
        var = var / (float)D + eps;

        float inv_std = 1.0f / sqrtf(var);

        /* Normalize */
        for (int64_t d = 0; d < D; d++) {
            yn[d] = (xn[d] - mean) * inv_std * gamma[d] + beta[d];
        }
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
