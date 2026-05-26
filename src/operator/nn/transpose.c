#include "operator.h"
#include "transpose_int.h"

int transpose_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const transpose_params_t* p = (const transpose_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    int64_t numel = 1;
    for (int d = 0; d < p->ndim; d++) numel *= p->shape[d];

    /* Precompute input strides */
    int64_t in_stride[8];
    in_stride[p->ndim - 1] = 1;
    for (int d = p->ndim - 2; d >= 0; d--)
        in_stride[d] = in_stride[d + 1] * p->shape[d + 1];

    /* Precompute output strides */
    int64_t out_stride[8];
    out_stride[p->ndim - 1] = 1;
    for (int d = p->ndim - 2; d >= 0; d--)
        out_stride[d] = out_stride[d + 1] * p->out_shape[d + 1];

    for (int64_t out_idx = 0; out_idx < numel; out_idx++) {
        /* Compute output multi-index */
        int64_t tmp = out_idx;
        int64_t out_coord[8] = {0};
        for (int d = p->ndim - 1; d >= 0; d--) {
            out_coord[d] = tmp % p->out_shape[d];
            tmp /= p->out_shape[d];
        }

        /* Compute input multi-index: in_coord[perm[d]] = out_coord[d] */
        int64_t in_coord[8] = {0};
        for (int d = 0; d < p->ndim; d++)
            in_coord[p->perm[d]] = out_coord[d];

        /* Compute input flat index via strides */
        int64_t in_idx = 0;
        for (int d = 0; d < p->ndim; d++)
            in_idx += in_coord[d] * in_stride[d];

        out[out_idx] = in[in_idx];
    }
    return 0;
}

static const operator_registry_t s_transpose_reg = {
    .name      = "transpose_f32",
    .data_type = "f32",
    .func      = transpose_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_transpose_f32(void) {
    return operator_register(&s_transpose_reg);
}
