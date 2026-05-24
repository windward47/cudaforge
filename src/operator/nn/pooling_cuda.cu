#include "operator.h"
#include "cuda_ops.h"
#include "pooling_int.h"
#include <float.h>

/* Each block handles one (n, c, oh) — threads handle ow */
__global__ void maxpool2d_f32_kernel(const float* input, float* output,
                                      int64_t C, int64_t H, int64_t W,
                                      int64_t OH, int64_t OW,
                                      int64_t KH, int64_t KW,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t pad_h, int64_t pad_w) {
    int64_t n  = blockIdx.z;
    int64_t c  = blockIdx.y;
    int64_t oh = blockIdx.x;

    int64_t ow = threadIdx.x;
    if (ow >= OW) return;

    float max_val = -FLT_MAX;
    for (int64_t kh = 0; kh < KH; kh++) {
        for (int64_t kw = 0; kw < KW; kw++) {
            int64_t ih = oh * stride_h + kh - pad_h;
            int64_t iw = ow * stride_w + kw - pad_w;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                float v = input[((n * C + c) * H + ih) * W + iw];
                if (v > max_val) max_val = v;
            }
        }
    }
    output[((n * C + c) * OH + oh) * OW + ow] = max_val;
}

__global__ void avgpool2d_f32_kernel(const float* input, float* output,
                                      int64_t C, int64_t H, int64_t W,
                                      int64_t OH, int64_t OW,
                                      int64_t KH, int64_t KW,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t pad_h, int64_t pad_w) {
    int64_t n  = blockIdx.z;
    int64_t c  = blockIdx.y;
    int64_t oh = blockIdx.x;

    int64_t ow = threadIdx.x;
    if (ow >= OW) return;

    float sum = 0.0f;
    for (int64_t kh = 0; kh < KH; kh++) {
        for (int64_t kw = 0; kw < KW; kw++) {
            int64_t ih = oh * stride_h + kh - pad_h;
            int64_t iw = ow * stride_w + kw - pad_w;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                sum += input[((n * C + c) * H + ih) * W + iw];
            }
        }
    }
    float inv = 1.0f / (float)(KH * KW);
    output[((n * C + c) * OH + oh) * OW + ow] = sum * inv;
}

static int launch_pool(cudaStream_t s, const void* kernel,
                        const float* in, float* out,
                        int64_t N, int64_t C, int64_t H, int64_t W,
                        int64_t KH, int64_t KW, int64_t OH, int64_t OW,
                        int64_t stride_h, int64_t stride_w,
                        int64_t pad_h, int64_t pad_w) {
    dim3 block(OW > 256 ? 256 : (OW > 0 ? (unsigned)OW : 1), 1, 1);
    dim3 grid(OH, C, N);
    CUDA_KERNEL_LAUNCH(kernel, grid, block, 0, s,
                       in, out, C, H, W, OH, OW, KH, KW,
                       stride_h, stride_w, pad_h, pad_w);
    return 0;
}

int maxpool2d_f32_cuda(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const pool_params_t* p = (const pool_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t OH = (p->H + 2 * p->pad_h - p->kernel_h) / p->stride_h + 1;
    int64_t OW = (p->W + 2 * p->pad_w - p->kernel_w) / p->stride_w + 1;

    return launch_pool(s, (const void*)maxpool2d_f32_kernel,
                       (const float*)inputs[0], (float*)outputs[0],
                       p->N, p->C, p->H, p->W,
                       p->kernel_h, p->kernel_w, OH, OW,
                       p->stride_h, p->stride_w, p->pad_h, p->pad_w);
}

int avgpool2d_f32_cuda(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const pool_params_t* p = (const pool_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t OH = (p->H + 2 * p->pad_h - p->kernel_h) / p->stride_h + 1;
    int64_t OW = (p->W + 2 * p->pad_w - p->kernel_w) / p->stride_w + 1;

    return launch_pool(s, (const void*)avgpool2d_f32_kernel,
                       (const float*)inputs[0], (float*)outputs[0],
                       p->N, p->C, p->H, p->W,
                       p->kernel_h, p->kernel_w, OH, OW,
                       p->stride_h, p->stride_w, p->pad_h, p->pad_w);
}

extern "C" int register_pooling_cuda(void) {
    static operator_registry_t maxpool_reg = {
        .name = "maxpool2d_f32_cuda", .data_type = "f32",
        .func = maxpool2d_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    static operator_registry_t avgpool_reg = {
        .name = "avgpool2d_f32_cuda", .data_type = "f32",
        .func = avgpool2d_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    int ret = 0;
    ret += operator_register(&maxpool_reg);
    ret += operator_register(&avgpool_reg);
    return ret;
}
