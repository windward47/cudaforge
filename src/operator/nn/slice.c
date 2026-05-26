#include "operator.h"
#include "slice_int.h"
#include <stddef.h>

int slice_f32(const void* inputs[], void* outputs[],
              const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const slice_params_t* p = (const slice_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    int ndim = p->ndim;
    if (ndim < 1 || ndim > 8) return -1;

    for (int64_t oi = 0; oi < p->numel; oi++) {
        int64_t remaining = oi;
        int64_t in_idx = 0;
        for (int d = ndim - 1; d >= 0; d--) {
            int64_t coord = remaining % p->out_shape[d];
            remaining /= p->out_shape[d];
            in_idx += (p->starts[d] + coord * p->steps[d]) * p->in_strides[d];
        }
        out[oi] = in[in_idx];
    }
    return 0;
}

static const operator_registry_t s_slice_reg = {
    .name      = "slice_f32",
    .data_type = "f32",
    .func      = slice_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_slice_f32(void) {
    return operator_register(&s_slice_reg);
}
