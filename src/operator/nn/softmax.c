#include "operator.h"
#include "softmax_int.h"
#include <math.h>
#include <string.h>
#include <float.h>

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
