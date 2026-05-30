/* FP16 Conv2D CUDA kernel — direct convolution (no im2col).
   Reads FP16 inputs/weights, accumulates in FP32, writes FP16 output.
   Registered as "conv2d_f16_cuda". */
#include "operator.h"
#include "cuda_ops.h"
#include "conv_int.h"
#include <cuda_fp16.h>

__global__ void conv2d_f16_direct_kernel(
    const __half* __restrict__ input,   /* (N, C, H, W) */
    const __half* __restrict__ weight,  /* (K, C, KH, KW) */
    const __half* __restrict__ bias,    /* (K,) or NULL */
    __half* __restrict__ output,        /* (N, K, OH, OW) */
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t KH, int64_t KW,
    int64_t OH, int64_t OW,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dil_h, int64_t dil_w,
    int fuse_act)
{
    /* Each thread computes one output element */
    int64_t total = N * K * OH * OW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t ow = idx % OW;
    int64_t tmp = idx / OW;
    int64_t oh = tmp % OH;
    tmp = tmp / OH;
    int64_t k  = tmp % K;
    int64_t n  = tmp / K;

    float acc = 0.0f;

    for (int64_t c = 0; c < C; c++) {
        for (int64_t kh = 0; kh < KH; kh++) {
            for (int64_t kw = 0; kw < KW; kw++) {
                int64_t ih = oh * stride_h + kh * dil_h - pad_h;
                int64_t iw = ow * stride_w + kw * dil_w - pad_w;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = __half2float(input[((n * C + c) * H + ih) * W + iw]);
                    float w_val  = __half2float(weight[((k * C + c) * KH + kh) * KW + kw]);
                    acc += in_val * w_val;
                }
            }
        }
    }

    if (bias) acc += __half2float(bias[k]);

    /* Fuse activation */
    if (fuse_act == 1) {        /* ReLU */
        if (acc < 0.0f) acc = 0.0f;
    } else if (fuse_act == 2) { /* Sigmoid */
        acc = 1.0f / (1.0f + expf(-acc));
    } else if (fuse_act == 3) { /* GELU */
        float c = 0.707106781186547524f;
        acc = 0.5f * acc * (1.0f + tanhf(0.7978845608028654f * (acc + 0.044715f * acc * acc * acc)));
    }

    output[idx] = __float2half(acc);
}

int conv2d_f16_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;

    const conv_params_t* p = (const conv_params_t*)params;
    const __half* in = (const __half*)inputs[0];
    const __half* w  = (const __half*)inputs[1];
    const __half* b  = (const __half*)inputs[2];  /* may be NULL */
    __half* out = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t N = p->N, C = p->C, H = p->H, W = p->W, K = p->K;
    int64_t KH = p->kernel_h, KW = p->kernel_w;
    int64_t OH = (H + 2 * p->pad_h - p->dilation_h * (KH - 1) - 1) / p->stride_h + 1;
    int64_t OW = (W + 2 * p->pad_w - p->dilation_w * (KW - 1) - 1) / p->stride_w + 1;

    int64_t total = N * K * OH * OW;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(conv2d_f16_direct_kernel, grid, block, 0, s,
                              in, w, b, out,
                              N, C, H, W, K, KH, KW, OH, OW,
                              p->stride_h, p->stride_w,
                              p->pad_h, p->pad_w,
                              p->dilation_h, p->dilation_w,
                              p->fuse_activation);
}

extern "C" int register_conv2d_f16_cuda(void) {
    static operator_registry_t reg = {
        .name = "conv2d_f16_cuda", .data_type = "f16",
        .func = conv2d_f16_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
