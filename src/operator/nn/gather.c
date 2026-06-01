#include "operator.h"
#include "gather_int.h"
#include <string.h>

int gather_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const gather_params_t* p = (const gather_params_t*)params;
    const float*   data    = (const float*)inputs[0];
    const void*    indices = inputs[1];
    float* out              = (float*)outputs[0];

    int64_t ni = p->num_indices;
    int64_t bs = p->block_size;
    int64_t os = p->outer_size;
    int indices_are_i64 = p->out_axis_dim;  /* reuse field as flag */

    for (int64_t o = 0; o < os; o++) {
        for (int64_t i = 0; i < ni; i++) {
            int64_t src_idx;
            if (indices_are_i64)
                src_idx = ((const int64_t*)indices)[i];
            else
                src_idx = (int64_t)((const float*)indices)[i];
            int64_t axis_dim = p->inner_size / bs;
            if (src_idx < 0) src_idx += axis_dim;
            int64_t in_start  = o * p->inner_size + src_idx * bs;
            int64_t out_start = (o * ni + i) * bs;
            memcpy(&out[out_start], &data[in_start], (size_t)bs * sizeof(float));
        }
    }
    return 0;
}

static const operator_registry_t s_gather_reg = {
    .name      = "gather_f32",
    .data_type = "f32",
    .func      = gather_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_gather_f32(void) {
    return operator_register(&s_gather_reg);
}
