#include "operator.h"
#include <math.h>
#include <stddef.h>

/* ReLU: y = max(0, x) */
int relu_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)params;
    (void)stream;

    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];
    int64_t n        = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        out[i] = fmaxf(in[i], 0.0f);
    }
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
