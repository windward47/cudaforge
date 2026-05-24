#include "operator.h"
#include "pooling_int.h"
#include <math.h>
#include <stddef.h>
#include <float.h>

/* MaxPool2D */
int maxpool2d_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const pool_params_t* p = (const pool_params_t*)params;
    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];

    int64_t OH = (p->H + 2 * p->pad_h - p->kernel_h) / p->stride_h + 1;
    int64_t OW = (p->W + 2 * p->pad_w - p->kernel_w) / p->stride_w + 1;

    for (int64_t n = 0; n < p->N; n++) {
        for (int64_t c = 0; c < p->C; c++) {
            for (int64_t oh = 0; oh < OH; oh++) {
                for (int64_t ow = 0; ow < OW; ow++) {
                    float max_val = -FLT_MAX;
                    for (int64_t kh = 0; kh < p->kernel_h; kh++) {
                        for (int64_t kw = 0; kw < p->kernel_w; kw++) {
                            int64_t ih = oh * p->stride_h + kh - p->pad_h;
                            int64_t iw = ow * p->stride_w + kw - p->pad_w;
                            if (ih >= 0 && ih < p->H && iw >= 0 && iw < p->W) {
                                float v = in[((n * p->C + c) * p->H + ih) * p->W + iw];
                                if (v > max_val) max_val = v;
                            }
                        }
                    }
                    out[((n * p->C + c) * OH + oh) * OW + ow] = max_val;
                }
            }
        }
    }
    return 0;
}

/* AvgPool2D */
int avgpool2d_f32(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const pool_params_t* p = (const pool_params_t*)params;
    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];

    int64_t OH = (p->H + 2 * p->pad_h - p->kernel_h) / p->stride_h + 1;
    int64_t OW = (p->W + 2 * p->pad_w - p->kernel_w) / p->stride_w + 1;
    float inv = 1.0f / (float)(p->kernel_h * p->kernel_w);

    for (int64_t n = 0; n < p->N; n++) {
        for (int64_t c = 0; c < p->C; c++) {
            for (int64_t oh = 0; oh < OH; oh++) {
                for (int64_t ow = 0; ow < OW; ow++) {
                    float sum = 0.0f;
                    for (int64_t kh = 0; kh < p->kernel_h; kh++) {
                        for (int64_t kw = 0; kw < p->kernel_w; kw++) {
                            int64_t ih = oh * p->stride_h + kh - p->pad_h;
                            int64_t iw = ow * p->stride_w + kw - p->pad_w;
                            if (ih >= 0 && ih < p->H && iw >= 0 && iw < p->W) {
                                sum += in[((n * p->C + c) * p->H + ih) * p->W + iw];
                            }
                        }
                    }
                    out[((n * p->C + c) * OH + oh) * OW + ow] = sum * inv;
                }
            }
        }
    }
    return 0;
}

static const operator_registry_t s_maxpool_reg = {
    .name = "maxpool2d_f32", .data_type = "f32",
    .func = maxpool2d_f32, .version = 1, .flags = OP_FLAG_NONE,
};

static const operator_registry_t s_avgpool_reg = {
    .name = "avgpool2d_f32", .data_type = "f32",
    .func = avgpool2d_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_pooling(void) {
    int ret = 0;
    ret += operator_register(&s_maxpool_reg);
    ret += operator_register(&s_avgpool_reg);
    return ret;
}
