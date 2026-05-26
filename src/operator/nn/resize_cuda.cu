#include "operator.h"
#include "cuda_ops.h"
#include "resize_int.h"

__global__ void resize_nearest_f32_kernel(const float* in, float* out,
                                           int64_t N, int64_t C,
                                           int64_t H_in, int64_t W_in,
                                           int64_t H_out, int64_t W_out,
                                           float scale_h, float scale_w) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    /* Decompose flat index: n, c, h, w */
    int64_t tmp = idx;
    int64_t w = tmp % W_out; tmp /= W_out;
    int64_t h = tmp % H_out; tmp /= H_out;
    int64_t c = tmp % C; tmp /= C;
    int64_t n = tmp;

    int64_t h_in = (int64_t)((float)h / scale_h);
    if (h_in >= H_in) h_in = H_in - 1;
    int64_t w_in = (int64_t)((float)w / scale_w);
    if (w_in >= W_in) w_in = W_in - 1;

    int64_t in_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
    out[idx] = in[in_idx];
}

int resize_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const resize_params_t* p = (const resize_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t total = p->N * p->C * p->H_out * p->W_out;
    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)((total + 255) / 256), 1, 1);

    CUDA_KERNEL_LAUNCH(resize_nearest_f32_kernel, grid, block, 0, s,
                       in, out, p->N, p->C,
                       p->H_in, p->W_in, p->H_out, p->W_out,
                       p->scale_h, p->scale_w);
    return 0;
}

extern "C" int register_resize_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "resize_f32_cuda", .data_type = "f32",
        .func = resize_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
