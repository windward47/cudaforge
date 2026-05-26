#include "operator.h"
#include "concat_int.h"
#include <string.h>

int concat_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const concat_params_t* p = (const concat_params_t*)params;
    float* out = (float*)outputs[0];

    for (int64_t oi = 0; oi < p->outer; oi++) {
        for (int ii = 0; ii < p->num_inputs; ii++) {
            if (!inputs[ii]) continue;
            const float* in_i = (const float*)inputs[ii];
            int64_t copy_len = p->C_per_input[ii] * p->inner;
            int64_t src_off = oi * p->C_per_input[ii] * p->inner;
            int64_t dst_off = (oi * p->C_total + p->C_offset[ii]) * p->inner;
            memcpy(out + dst_off, in_i + src_off, (size_t)copy_len * sizeof(float));
        }
    }
    return 0;
}

static const operator_registry_t s_concat_reg = {
    .name      = "concat_f32",
    .data_type = "f32",
    .func      = concat_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_concat_f32(void) {
    return operator_register(&s_concat_reg);
}
