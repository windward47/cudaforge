#include "operator.h"
#include "cuda_ops.h"
#include "reshape_int.h"

__global__ void reshape_copy_kernel(const float* src, float* dst, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = src[i];
}

int reshape_f32_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const reshape_params_t* p = (const reshape_params_t*)params;
    const float* src = (const float*)inputs[0];
    float* dst = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t N = p->numel;
    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)((N + 255) / 256), 1, 1);

    CUDA_KERNEL_LAUNCH(reshape_copy_kernel, grid, block, 0, s, src, dst, N);
    return 0;
}

extern "C" int register_reshape_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "reshape_f32_cuda", .data_type = "f32",
        .func = reshape_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
