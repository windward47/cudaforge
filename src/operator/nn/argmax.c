#include "operator.h"
#include "argmax_int.h"
#include <math.h>

int argmax_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    const argmax_params_t* ap = (const argmax_params_t*)params;

    int64_t reduce_size = ap->reduce_size;
    int64_t num_blocks  = ap->num_blocks;

    for (int64_t b = 0; b < num_blocks; b++) {
        const float* block_in = in + b * reduce_size;
        float max_val = block_in[0];
        int64_t max_idx = 0;
        for (int64_t i = 1; i < reduce_size; i++) {
            if (block_in[i] > max_val) {
                max_val = block_in[i];
                max_idx = i;
            }
        }
        out[b] = (float)max_idx;  /* store index as float */
    }
    return 0;
}

static const operator_registry_t s_argmax_reg = {
    .name = "argmax_f32", .data_type = "f32",
    .func = argmax_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_argmax_f32(void) {
    return operator_register(&s_argmax_reg);
}
