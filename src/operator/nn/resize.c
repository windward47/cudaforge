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
                for (int64_t w = 0; w < p->W_out; w++) {
                    int64_t out_idx = ((n * p->C + c) * p->H_out + h) * p->W_out + w;
                    int64_t in_base = (n * p->C + c) * p->H_in * p->W_in;

                    if (p->mode == 1) {
                        /* Bilinear interpolation */
                        float src_y = ((float)h + 0.5f) / scale_h - 0.5f;
                        float src_x = ((float)w + 0.5f) / scale_w - 0.5f;
                        int64_t y0 = (int64_t)src_y;
                        int64_t x0 = (int64_t)src_x;
                        int64_t y1 = y0 + 1;
                        int64_t x1 = x0 + 1;
                        if (y0 < 0) y0 = 0;
                        if (x0 < 0) x0 = 0;
                        if (y1 >= p->H_in) y1 = p->H_in - 1;
                        if (x1 >= p->W_in) x1 = p->W_in - 1;
                        float dy = src_y - (float)y0;
                        float dx = src_x - (float)x0;
                        float tl = in[in_base + y0 * p->W_in + x0];
                        float tr = in[in_base + y0 * p->W_in + x1];
                        float bl = in[in_base + y1 * p->W_in + x0];
                        float br = in[in_base + y1 * p->W_in + x1];
                        out[out_idx] = (1.0f - dy) * (1.0f - dx) * tl
                                     + (1.0f - dy) *        dx  * tr
                                     +        dy  * (1.0f - dx) * bl
                                     +        dy  *        dx  * br;
                    } else {
                        /* Nearest neighbor */
                        int64_t h_in = (int64_t)((float)h / scale_h);
                        if (h_in >= p->H_in) h_in = p->H_in - 1;
                        int64_t w_in = (int64_t)((float)w / scale_w);
                        if (w_in >= p->W_in) w_in = p->W_in - 1;
                        out[out_idx] = in[in_base + h_in * p->W_in + w_in];
                    }
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
