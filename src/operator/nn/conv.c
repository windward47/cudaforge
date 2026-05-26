#include "operator.h"
#include "conv_int.h"
#include "matmul_int.h"
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

/* matmul dispatched via operator registry */

/* ============================================================
 * Naive 6-loop reference convolution
 * ============================================================ */
static int conv2d_f32_ref(const float* input, const float* weight, float* output,
                           const conv_params_t* p) {
    int64_t N = p->N, C = p->C, H = p->H, W = p->W;
    int64_t K = p->K;
    int64_t KH = p->kernel_h, KW = p->kernel_w;
    int64_t pad_h = p->pad_h, pad_w = p->pad_w;
    int64_t stride_h = p->stride_h, stride_w = p->stride_w;
    int64_t dil_h = p->dilation_h, dil_w = p->dilation_w;

    int64_t OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) / stride_h + 1;
    int64_t OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) / stride_w + 1;

    for (int64_t n = 0; n < N; n++) {
        for (int64_t k = 0; k < K; k++) {
            for (int64_t oh = 0; oh < OH; oh++) {
                for (int64_t ow = 0; ow < OW; ow++) {
                    float sum = 0.0f;
                    for (int64_t c = 0; c < C; c++) {
                        for (int64_t kh = 0; kh < KH; kh++) {
                            for (int64_t kw = 0; kw < KW; kw++) {
                                int64_t ih = oh * stride_h + kh * dil_h - pad_h;
                                int64_t iw = ow * stride_w + kw * dil_w - pad_w;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int64_t in_idx = ((n * C + c) * H + ih) * W + iw;
                                    int64_t w_idx  = ((k * C + c) * KH + kh) * KW + kw;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    int64_t out_idx = ((n * K + k) * OH + oh) * OW + ow;
                    output[out_idx] = sum;
                }
            }
        }
    }
    return 0;
}

/* ============================================================
 * im2col + gemm implementation
 * ============================================================ */

/* im2col: extract patches from input into a (C*KH*KW) × (OH*OW) matrix */
static void im2col_f32(const float* input, float* col_buf,
                       int64_t C, int64_t H, int64_t W,
                       int64_t KH, int64_t KW,
                       int64_t pad_h, int64_t pad_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t dil_h, int64_t dil_w,
                       int64_t OH, int64_t OW) {
    int64_t col_rows = C * KH * KW;
    int64_t col_cols = OH * OW;

    for (int64_t ci = 0; ci < col_rows; ci++) {
        int64_t c     = ci / (KH * KW);
        int64_t kh    = (ci / KW) % KH;
        int64_t kw    = ci % KW;

        for (int64_t oj = 0; oj < col_cols; oj++) {
            int64_t oh = oj / OW;
            int64_t ow = oj % OW;

            int64_t ih = oh * stride_h + kh * dil_h - pad_h;
            int64_t iw = ow * stride_w + kw * dil_w - pad_w;

            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                col_buf[ci * col_cols + oj] = input[c * H * W + ih * W + iw];
            } else {
                col_buf[ci * col_cols + oj] = 0.0f;
            }
        }
    }
}

/* conv2d via im2col + matmul */
static int conv2d_f32_im2col(const float* input, const float* weight, float* output,
                              const conv_params_t* p) {
    int64_t N = p->N, C = p->C, H = p->H, W = p->W, K = p->K;
    int64_t KH = p->kernel_h, KW = p->kernel_w;
    int64_t OH = (H + 2 * p->pad_h - p->dilation_h * (KH - 1) - 1) / p->stride_h + 1;
    int64_t OW = (W + 2 * p->pad_w - p->dilation_w * (KW - 1) - 1) / p->stride_w + 1;

    int64_t col_rows = C * KH * KW;
    int64_t col_cols = OH * OW;

    float* col_buf = (float*)malloc((size_t)col_rows * col_cols * sizeof(float));
    if (!col_buf) return -1;

    for (int64_t n = 0; n < N; n++) {
        const float* in_n = input + n * C * H * W;
        float* out_n      = output + n * K * OH * OW;

        im2col_f32(in_n, col_buf, C, H, W, KH, KW,
                   p->pad_h, p->pad_w, p->stride_h, p->stride_w,
                   p->dilation_h, p->dilation_w, OH, OW);

        /* matmul: weight (K × col_rows) * col_buf (col_rows × col_cols) = output (K × col_cols) */
        matmul_params_t mp = {.M = K, .N = col_cols, .K = col_rows};
        const void* mat_inputs[]  = {weight, col_buf, NULL};
        void*       mat_outputs[] = {out_n};
        const operator_registry_t* mm = operator_find("matmul_f32");
        if (mm) mm->func(mat_inputs, mat_outputs, (const operator_params_t*)&mp, NULL);
    }

    free(col_buf);
    return 0;
}

/* ============================================================
 * Public API
 * ============================================================ */
int conv2d_f32(const void* inputs[], void* outputs[],
               const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const conv_params_t* p = (const conv_params_t*)params;
    const float* in   = (const float*)inputs[0];
    const float* w    = (const float*)inputs[1];
    const float* bias = (const float*)inputs[2];
    float* out        = (float*)outputs[0];

    int ret;
    if (p->kernel_h * p->kernel_w <= 9) {
        ret = conv2d_f32_ref(in, w, out, p);
    } else {
        ret = conv2d_f32_im2col(in, w, out, p);
    }
    if (ret != 0) return ret;

    /* Add bias + fused activation if enabled */
    {
        int64_t OH = (p->H + 2 * p->pad_h - p->dilation_h * (p->kernel_h - 1) - 1)
                     / p->stride_h + 1;
        int64_t OW = (p->W + 2 * p->pad_w - p->dilation_w * (p->kernel_w - 1) - 1)
                     / p->stride_w + 1;
        for (int64_t n = 0; n < p->N; n++) {
            for (int64_t k = 0; k < p->K; k++) {
                float bv = bias ? bias[k] : 0.0f;
                for (int64_t oh = 0; oh < OH; oh++) {
                    for (int64_t ow = 0; ow < OW; ow++) {
                        int64_t idx = ((n * p->K + k) * OH + oh) * OW + ow;
                        float val = out[idx] + bv;
                        if (p->fuse_activation == 1) {
                            val = val > 0.0f ? val : 0.0f;
                        } else if (p->fuse_activation == 2) {
                            val = 1.0f / (1.0f + expf(-val));
                        } else if (p->fuse_activation == 3) {
                            float t = tanhf(0.79788456f * (val + 0.044715f * val * val * val));
                            val = 0.5f * val * (1.0f + t);
                        } else if (p->fuse_activation == 13) {
                            val = val / (1.0f + expf(-val));
                        }
                        out[idx] = val;
                    }
                }
            }
        }
    }
    return 0;
}

static const operator_registry_t s_conv_reg = {
    .name      = "conv2d_f32",
    .data_type = "f32",
    .func      = conv2d_f32,

    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_conv2d_f32(void) {
    return operator_register(&s_conv_reg);
}
