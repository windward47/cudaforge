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

    int64_t N = (p->H > 0 && p->W > 0 && p->C_total > 0)
        ? p->total_numel / (p->C_total * p->H * p->W) : 1;
    int64_t HW = p->H * p->W;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t h = 0; h < p->H; h++) {
            for (int64_t w = 0; w < p->W; w++) {
                int64_t out_spatial = (n * p->C_total * p->H + h) * p->W + w;
                for (int ii = 0; ii < p->num_inputs; ii++) {
                    if (!inputs[ii]) continue;
                    const float* in_i = (const float*)inputs[ii];
                    int64_t Ci = p->C_per_input[ii];
                    int64_t in_spatial = (n * Ci * p->H + h) * p->W + w;
                    for (int64_t c = 0; c < Ci; c++) {
                        out[out_spatial + (p->C_offset[ii] + c) * HW] =
                            in_i[in_spatial + c * HW];
                    }
                }
            }
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
