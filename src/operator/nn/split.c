#include "operator.h"
#include "split_int.h"
#include <string.h>

int split_f32(const void* inputs[], void* outputs[],
              const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs) return -1;
    if (!params) return -1;

    const split_params_t* p = (const split_params_t*)params;
    const float* in = (const float*)inputs[0];

    int ndim = p->ndim;
    int axis = p->axis;
    if (axis < 0) axis = ndim + axis;

    /* Compute elements per unit (all dims before axis combined) */
    int64_t outer = 1;
    for (int d = 0; d < axis; d++) outer *= p->in_shape[d];

    /* Compute inner stride (all dims after axis combined) */
    int64_t inner = 1;
    for (int d = axis + 1; d < ndim; d++) inner *= p->in_shape[d];
    int64_t axis_size = p->in_shape[axis];

    for (int o = 0; o < p->num_outputs; o++) {
        if (!outputs[o]) continue;
        float* out = (float*)outputs[o];
        int64_t split_size = p->splits[o];
        int64_t offset = p->offsets[o];
        int64_t out_offset = 0;
        for (int64_t b = 0; b < outer; b++) {
            int64_t src_base = b * axis_size * inner + offset;
            memcpy(out + out_offset, in + src_base,
                   (size_t)(split_size * inner) * sizeof(float));
            out_offset += split_size * inner;
        }
    }
    return 0;
}

static const operator_registry_t s_split_reg = {
    .name      = "split_f32",
    .data_type = "f32",
    .func      = split_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_split_f32(void) {
    return operator_register(&s_split_reg);
}
