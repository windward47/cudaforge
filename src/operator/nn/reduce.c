#include "operator.h"
#include "reduce_int.h"
#include <math.h>

int reduce_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const float* in = (const float*)inputs[0];
    float* out      = (float*)outputs[0];
    const reduce_params_t* rp = (const reduce_params_t*)params;

    int64_t reduce_size = rp->reduce_size;
    int64_t num_blocks  = rp->num_blocks;
    int     op          = rp->op;

    if (op == REDUCE_SUM) {
        for (int64_t b = 0; b < num_blocks; b++) {
            const float* block_in = in + b * reduce_size;
            float sum = 0.0f;
            for (int64_t i = 0; i < reduce_size; i++) {
                sum += block_in[i];
            }
            out[b] = sum;
        }
    } else { /* REDUCE_MAX */
        for (int64_t b = 0; b < num_blocks; b++) {
            const float* block_in = in + b * reduce_size;
            float max_val = block_in[0];
            for (int64_t i = 1; i < reduce_size; i++) {
                if (block_in[i] > max_val) max_val = block_in[i];
            }
            out[b] = max_val;
        }
    }
    return 0;
}

static const operator_registry_t s_reduce_reg = {
    .name = "reduce_f32", .data_type = "f32",
    .func = reduce_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_reduce_f32(void) {
    return operator_register(&s_reduce_reg);
}
