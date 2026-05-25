#include "operator.h"
#include "cuda_ops.h"
#include <cuda_runtime.h>

/* CUDA ReLU kernel */
__global__ void relu_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

/* CUDA wrapper */
int relu_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)params;

    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;

    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];
    int64_t n        = *(const int64_t*)inputs[1];

    dim3 block(256, 1, 1);
    dim3 grid((n + 255) / 256, 1, 1);
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    CUDA_KERNEL_LAUNCH(relu_f32_kernel, grid, block, 0, s, in, out, n);
    return 0;
}

extern "C" int register_relu_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "relu_f32_cuda", .data_type = "f32",
        .func = relu_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    return operator_register(&reg);
}
