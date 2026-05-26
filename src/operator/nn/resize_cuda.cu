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

__global__ void resize_bilinear_f32_kernel(const float* in, float* out,
                                            int64_t N, int64_t C,
                                            int64_t H_in, int64_t W_in,
                                            int64_t H_out, int64_t W_out,
                                            float scale_h, float scale_w) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t w = tmp % W_out; tmp /= W_out;
    int64_t h = tmp % H_out; tmp /= H_out;
    int64_t c = tmp % C; tmp /= C;
    int64_t n = tmp;

    float src_y = ((float)h + 0.5f) / scale_h - 0.5f;
    float src_x = ((float)w + 0.5f) / scale_w - 0.5f;
    int64_t y0 = (int64_t)src_y;
    int64_t x0 = (int64_t)src_x;
    int64_t y1 = y0 + 1;
    int64_t x1 = x0 + 1;
    if (y0 < 0) y0 = 0;
    if (x0 < 0) x0 = 0;
    if (y1 >= H_in) y1 = H_in - 1;
    if (x1 >= W_in) x1 = W_in - 1;
    float dy = src_y - (float)y0;
    float dx = src_x - (float)x0;

    int64_t in_base = (n * C + c) * H_in * W_in;
    float tl = in[in_base + y0 * W_in + x0];
    float tr = in[in_base + y0 * W_in + x1];
    float bl = in[in_base + y1 * W_in + x0];
    float br = in[in_base + y1 * W_in + x1];
    out[idx] = (1.0f - dy) * (1.0f - dx) * tl
             + (1.0f - dy) *        dx  * tr
             +        dy  * (1.0f - dx) * bl
             +        dy  *        dx  * br;
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
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->mode == 1) {
        return CUDA_KERNEL_LAUNCH(resize_bilinear_f32_kernel, grid, block, 0, s,
                           in, out, p->N, p->C,
                           p->H_in, p->W_in, p->H_out, p->W_out,
                           p->scale_h, p->scale_w);
    }
    return CUDA_KERNEL_LAUNCH(resize_nearest_f32_kernel, grid, block, 0, s,
                       in, out, p->N, p->C,
                       p->H_in, p->W_in, p->H_out, p->W_out,
                       p->scale_h, p->scale_w);
}

extern "C" int register_resize_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "resize_f32_cuda", .data_type = "f32",
        .func = resize_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
