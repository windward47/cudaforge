/* CPU Pad operator — constant/edge/reflect padding for 4D tensors.
   inputs[0] = input (N, C, H, W)
   outputs[0] = output (N, C, H+pad_h, W+pad_w) */
#include "operator.h"
#include "pad_int.h"
#include <string.h>

int pad_f32(const void* inputs[], void* outputs[],
            const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const pad_params_t* p = (const pad_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    int64_t H_out = p->H + p->pad_h_begin + p->pad_h_end;
    int64_t W_out = p->W + p->pad_w_begin + p->pad_w_end;

    /* Initialize output with constant value */
    if (p->mode == PAD_MODE_CONSTANT) {
        int64_t total = p->N * p->C * H_out * W_out;
        for (int64_t i = 0; i < total; i++) out[i] = p->value;
    }

    for (int64_t n = 0; n < p->N; n++) {
        for (int64_t c = 0; c < p->C; c++) {
            for (int64_t oh = 0; oh < H_out; oh++) {
                for (int64_t ow = 0; ow < W_out; ow++) {
                    int64_t ih = oh - p->pad_h_begin;
                    int64_t iw = ow - p->pad_w_begin;

                    int64_t out_idx = ((n * p->C + c) * H_out + oh) * W_out + ow;

                    if (ih >= 0 && ih < p->H && iw >= 0 && iw < p->W) {
                        /* Inside input — copy */
                        int64_t in_idx = ((n * p->C + c) * p->H + ih) * p->W + iw;
                        out[out_idx] = in[in_idx];
                    } else if (p->mode == PAD_MODE_EDGE) {
                        /* Clamp to nearest edge */
                        if (ih < 0) ih = 0;
                        if (ih >= p->H) ih = p->H - 1;
                        if (iw < 0) iw = 0;
                        if (iw >= p->W) iw = p->W - 1;
                        int64_t in_idx = ((n * p->C + c) * p->H + ih) * p->W + iw;
                        out[out_idx] = in[in_idx];
                    } else if (p->mode == PAD_MODE_REFLECT) {
                        /* Reflect around edges */
                        while (ih < 0 || ih >= p->H) {
                            if (ih < 0) ih = -ih;
                            if (ih >= p->H) ih = 2 * p->H - ih - 2;
                        }
                        while (iw < 0 || iw >= p->W) {
                            if (iw < 0) iw = -iw;
                            if (iw >= p->W) iw = 2 * p->W - iw - 2;
                        }
                        int64_t in_idx = ((n * p->C + c) * p->H + ih) * p->W + iw;
                        out[out_idx] = in[in_idx];
                    }
                    /* else: constant mode already initialized */
                }
            }
        }
    }
    return 0;
}

static const operator_registry_t s_pad_reg = {
    .name = "pad_f32", .data_type = "f32",
    .func = pad_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_pad_f32(void) {
    return operator_register(&s_pad_reg);
}
