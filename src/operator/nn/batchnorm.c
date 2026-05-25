#include "operator.h"
#include "batchnorm_int.h"
#include <math.h>

/* BatchNorm inference: y = gamma * (x - mean) / sqrt(var + eps) + beta
   inputs[0] = x (NCHW), inputs[1] = gamma (C), inputs[2] = beta (C),
   inputs[3] = mean (C), inputs[4] = var (C)
   outputs[0] = y (NCHW)
*/
int batchnorm_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] ||
        !inputs[3] || !inputs[4] || !inputs[5] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const batchnorm_params_t* p = (const batchnorm_params_t*)params;
    const float* x      = (const float*)inputs[0];
    const float* gamma  = (const float*)inputs[1];
    const float* beta   = (const float*)inputs[2];
    const float* mean   = (const float*)inputs[3];
    const float* var    = (const float*)inputs[4];
    float* y            = (float*)outputs[0];

    int64_t hw = *(const int64_t*)inputs[5];

    for (int64_t c = 0; c < p->C; c++) {
        float scale = gamma[c] / sqrtf(var[c] + p->epsilon);
        float bias  = beta[c] - scale * mean[c];
        for (int64_t i = 0; i < hw; i++) {
            y[c * hw + i] = x[c * hw + i] * scale + bias;
        }
    }
    return 0;
}

static const operator_registry_t s_batchnorm_reg = {
    .name = "batchnorm_f32", .data_type = "f32",
    .func = batchnorm_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_batchnorm(void) {
    return operator_register(&s_batchnorm_reg);
}
