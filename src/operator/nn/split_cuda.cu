#include "operator.h"
#include "cuda_ops.h"
#include "split_int.h"

__global__ void split_copy_kernel(const float* in, float* out,
                                   int64_t offset, int64_t numel) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) out[i] = in[offset + i];
}

int split_f32_cuda(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs) return -1;
    if (!params) return -1;

    const split_params_t* p = (const split_params_t*)params;
    const float* in = (const float*)inputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t cumulative = 0;
    for (int o = 0; o < p->num_outputs; o++) {
        if (!outputs[o]) continue;
        float* out = (float*)outputs[o];

        dim3 block(256, 1, 1);
        dim3 grid((unsigned int)((p->out_numel[o] + 255) / 256), 1, 1);
        CUDA_KERNEL_LAUNCH(split_copy_kernel, grid, block, 0, s,
                           in, out, cumulative, p->out_numel[o]);
        cumulative += p->out_numel[o];
    }
    return 0;
}

extern "C" int register_split_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "split_f32_cuda", .data_type = "f32",
        .func = split_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
