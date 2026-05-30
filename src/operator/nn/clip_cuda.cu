/* CUDA Clip kernel — clamp values to [min, max]. */
#include "operator.h"
#include "cuda_ops.h"
#include "clip_int.h"

__global__ void clip_f32_kernel(const float* in, float* out,
                                 int64_t n, float min_val, float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = in[idx];
    if (v < min_val) v = min_val;
    if (v > max_val) v = max_val;
    out[idx] = v;
}

int clip_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const clip_params_t* p = (const clip_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((p->numel + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(clip_f32_kernel, grid, block, 0, s,
                              in, out, p->numel, p->min_val, p->max_val);
}

extern "C" int register_clip_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "clip_f32_cuda", .data_type = "f32",
        .func = clip_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
