/* CUDA Pad kernel — constant/edge/reflect padding for 4D tensors. */
#include "operator.h"
#include "cuda_ops.h"
#include "pad_int.h"
#include <cuda_fp16.h>

__global__ void pad_f32_kernel(const float* in, float* out,
                                int64_t N, int64_t C,
                                int64_t H, int64_t W,
                                int64_t H_out, int64_t W_out,
                                int64_t pad_hb, int64_t pad_wb,
                                int mode, float value) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t ow = tmp % W_out; tmp /= W_out;
    int64_t oh = tmp % H_out; tmp /= H_out;
    int64_t c  = tmp % C;     tmp /= C;
    int64_t n  = tmp;

    int64_t ih = oh - pad_hb;
    int64_t iw = ow - pad_wb;

    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        out[idx] = in[in_idx];
    } else if (mode == 1) { /* EDGE */
        if (ih < 0) ih = 0;
        if (ih >= H) ih = H - 1;
        if (iw < 0) iw = 0;
        if (iw >= W) iw = W - 1;
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        out[idx] = in[in_idx];
    } else if (mode == 2) { /* REFLECT */
        while (ih < 0 || ih >= H) {
            if (ih < 0) ih = -ih;
            if (ih >= H) ih = 2 * H - ih - 2;
        }
        while (iw < 0 || iw >= W) {
            if (iw < 0) iw = -iw;
            if (iw >= W) iw = 2 * W - iw - 2;
        }
        int in_idx = ((n * C + c) * H + ih) * W + iw;
        out[idx] = in[in_idx];
    } else { /* CONSTANT */
        out[idx] = value;
    }
}

int pad_f32_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const pad_params_t* p = (const pad_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t H_out = p->H + p->pad_h_begin + p->pad_h_end;
    int64_t W_out = p->W + p->pad_w_begin + p->pad_w_end;
    int64_t total = p->N * p->C * H_out * W_out;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(pad_f32_kernel, grid, block, 0, s,
                              in, out, p->N, p->C, p->H, p->W,
                              H_out, W_out, p->pad_h_begin, p->pad_w_begin,
                              p->mode, p->value);
}

extern "C" int register_pad_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "pad_f32_cuda", .data_type = "f32",
        .func = pad_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
