#include "operator.h"
#include "cuda_ops.h"

__global__ void sigmoid_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void gelu_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        float x3 = x * x * x;
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x3)));
    }
}

static int launch_1d(cudaStream_t s, const void* kernel,
                     const float* in, float* out, int64_t n) {
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);
    return CUDA_KERNEL_LAUNCH(kernel, grid, block, 0, s, in, out, n);
}

int sigmoid_f32_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    (void)params;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    return launch_1d(s, (const void*)sigmoid_f32_kernel,
                     (const float*)inputs[0], (float*)outputs[0],
                     *(const int64_t*)inputs[1]);
}

__global__ void exp_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = expf(input[idx]);
    }
}

int exp_f32_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    (void)params;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    return launch_1d(s, (const void*)exp_f32_kernel,
                     (const float*)inputs[0], (float*)outputs[0],
                     *(const int64_t*)inputs[1]);
}

__global__ void silu_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

int silu_f32_cuda(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    (void)params;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    return launch_1d(s, (const void*)silu_f32_kernel,
                     (const float*)inputs[0], (float*)outputs[0],
                     *(const int64_t*)inputs[1]);
}

int gelu_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    (void)params;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    return launch_1d(s, (const void*)gelu_f32_kernel,
                     (const float*)inputs[0], (float*)outputs[0],
                     *(const int64_t*)inputs[1]);
}

__global__ void tanh_f32_kernel(const float* in, float* out, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = tanhf(in[idx]);
}

int tanh_f32_cuda(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    (void)params;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    return launch_1d(s, (const void*)tanh_f32_kernel,
                     (const float*)inputs[0], (float*)outputs[0],
                     *(const int64_t*)inputs[1]);
}

extern "C" int register_activations_cuda(void) {
    static operator_registry_t sigmoid_reg = {
        .name = "sigmoid_f32_cuda", .data_type = "f32",
        .func = sigmoid_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    static operator_registry_t gelu_reg = {
        .name = "gelu_f32_cuda", .data_type = "f32",
        .func = gelu_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    static operator_registry_t silu_reg = {
        .name = "silu_f32_cuda", .data_type = "f32",
        .func = silu_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    static operator_registry_t exp_reg = {
        .name = "exp_f32_cuda", .data_type = "f32",
        .func = exp_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    int ret = 0;
    ret += operator_register(&sigmoid_reg);
    ret += operator_register(&gelu_reg);
    ret += operator_register(&silu_reg);
    ret += operator_register(&exp_reg);
    static operator_registry_t tanh_reg = {
        .name = "tanh_f32_cuda", .data_type = "f32",
        .func = tanh_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    ret += operator_register(&tanh_reg);
    return ret;
}
