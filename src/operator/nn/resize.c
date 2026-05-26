#include "operator.h"
#include "resize_int.h"

int resize_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const resize_params_t* p = (const resize_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    float scale_h = p->scale_h;
    float scale_w = p->scale_w;

    for (int64_t n = 0; n < p->N; n++) {
        for (int64_t c = 0; c < p->C; c++) {
            for (int64_t h = 0; h < p->H_out; h++) {
                int64_t h_in = (int64_t)(h / scale_h);
                if (h_in >= p->H_in) h_in = p->H_in - 1;
                for (int64_t w = 0; w < p->W_out; w++) {
                    int64_t w_in = (int64_t)(w / scale_w);
                    if (w_in >= p->W_in) w_in = p->W_in - 1;
                    int64_t out_idx = ((n * p->C + c) * p->H_out + h) * p->W_out + w;
                    int64_t in_idx  = ((n * p->C + c) * p->H_in  + h_in) * p->W_in + w_in;
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
    return 0;
}

static const operator_registry_t s_resize_reg = {
    .name      = "resize_f32",
    .data_type = "f32",
    .func      = resize_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_resize_f32(void) {
    return operator_register(&s_resize_reg);
}
