/* FP16 elementwise CUDA kernels.
   Registered as relu_f16_cuda, add_f16_cuda, mul_f16_cuda, sub_f16_cuda, div_f16_cuda.
   Same interface as FP32 variants but operates on __half data. */
#include "operator.h"
#include "cuda_ops.h"
#include "add_int.h"
#include "mul_int.h"
#include "sub_int.h"
#include "div_int.h"
#include <cuda_fp16.h>

/* ============================================================
 * Unary: ReLU, Sigmoid, GELU, SiLU, Exp
 * ============================================================ */
__global__ void relu_f16_kernel(const __half* input, __half* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        __half zero = __float2half(0.0f);
        output[idx] = __hgt(input[idx], zero) ? input[idx] : zero;
    }
}

__global__ void sigmoid_f16_kernel(const __half* input, __half* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(1.0f / (1.0f + expf(-x)));
    }
}

__global__ void gelu_f16_kernel(const __half* input, __half* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        float c = 0.707106781186547524f;  /* 1/sqrt(2) */
        float t = tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
        output[idx] = __float2half(0.5f * x * (1.0f + t));
    }
}

__global__ void silu_f16_kernel(const __half* input, __half* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __half2float(input[idx]);
        output[idx] = __float2half(x / (1.0f + expf(-x)));
    }
}

__global__ void exp_f16_kernel(const __half* input, __half* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(expf(__half2float(input[idx])));
    }
}

/* ============================================================
 * Binary: Add, Mul, Sub, Div (no broadcast)
 * ============================================================ */
__global__ void add_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(a[i], b[i]);
}

__global__ void mul_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hmul(a[i], b[i]);
}

__global__ void sub_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hsub(a[i], b[i]);
}

__global__ void div_f16_kernel(const __half* a, const __half* b, __half* out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hdiv(a[i], b[i]);
}

/* ============================================================
 * Broadcast Add/Mul (channel broadcast: b is smaller)
 * ============================================================ */
__global__ void add_f16_broadcast_kernel(const __half* a, const __half* b,
                                          __half* out, int64_t N, int64_t C) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __hadd(a[i], b[i % C]);
}

__global__ void mul_f16_broadcast_kernel(const __half* a, const __half* b,
                                          __half* out, int64_t N, int64_t C) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __hmul(a[i], b[i % C]);
}

__global__ void sub_f16_broadcast_kernel(const __half* a, const __half* b,
                                          __half* out, int64_t N, int64_t C) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __hsub(a[i], b[i % C]);
}

__global__ void div_f16_broadcast_kernel(const __half* a, const __half* b,
                                          __half* out, int64_t N, int64_t C) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = __hdiv(a[i], b[i % C]);
}

/* ============================================================
 * Host dispatch functions
 * ============================================================ */
#define FP16_UNARY_DISPATCH(name) \
    int name##_f16_cuda(const void* inputs[], void* outputs[], \
                        const operator_params_t* params, stream_t* stream) { \
        (void)params; \
        if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0]) return -1; \
        const __half* in = (const __half*)inputs[0]; \
        __half* out = (__half*)outputs[0]; \
        int64_t n = *(const int64_t*)inputs[1]; \
        dim3 block(OPS_THREADS_PER_BLOCK, 1, 1); \
        dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1); \
        cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0; \
        return CUDA_KERNEL_LAUNCH(name##_f16_kernel, grid, block, 0, s, in, out, n); \
    }

FP16_UNARY_DISPATCH(relu)
FP16_UNARY_DISPATCH(sigmoid)
FP16_UNARY_DISPATCH(gelu)
FP16_UNARY_DISPATCH(silu)
FP16_UNARY_DISPATCH(exp)

/* Add FP16 */
int add_f16_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;
    const add_params_t* p = (const add_params_t*)params;
    const __half* a = (const __half*)inputs[0];
    const __half* b = (const __half*)inputs[1];
    __half* out = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t n = p->numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->B_numel == n) {
        return CUDA_KERNEL_LAUNCH(add_f16_kernel, grid, block, 0, s, a, b, out, n);
    } else if (p->B_numel > 0 && p->B_numel < n) {
        return CUDA_KERNEL_LAUNCH(add_f16_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, n, p->B_numel);
    }
    return CUDA_KERNEL_LAUNCH(add_f16_kernel, grid, block, 0, s, a, b, out, n);
}

/* Mul FP16 */
int mul_f16_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;
    const mul_params_t* p = (const mul_params_t*)params;
    const __half* a = (const __half*)inputs[0];
    const __half* b = (const __half*)inputs[1];
    __half* out = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t n = p->numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->B_numel == n) {
        return CUDA_KERNEL_LAUNCH(mul_f16_kernel, grid, block, 0, s, a, b, out, n);
    } else if (p->B_numel > 0 && p->B_numel < n) {
        return CUDA_KERNEL_LAUNCH(mul_f16_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, n, p->B_numel);
    }
    return CUDA_KERNEL_LAUNCH(mul_f16_kernel, grid, block, 0, s, a, b, out, n);
}

/* Sub FP16 */
int sub_f16_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;
    const sub_params_t* p = (const sub_params_t*)params;
    const __half* a = (const __half*)inputs[0];
    const __half* b = (const __half*)inputs[1];
    __half* out = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t n = p->numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->B_numel == n) {
        return CUDA_KERNEL_LAUNCH(sub_f16_kernel, grid, block, 0, s, a, b, out, n);
    } else if (p->B_numel > 0 && p->B_numel < n) {
        return CUDA_KERNEL_LAUNCH(sub_f16_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, n, p->B_numel);
    }
    return CUDA_KERNEL_LAUNCH(sub_f16_kernel, grid, block, 0, s, a, b, out, n);
}

/* Div FP16 */
int div_f16_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0] || !params) return -1;
    const div_params_t* p = (const div_params_t*)params;
    const __half* a = (const __half*)inputs[0];
    const __half* b = (const __half*)inputs[1];
    __half* out = (__half*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    int64_t n = p->numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->B_numel == n) {
        return CUDA_KERNEL_LAUNCH(div_f16_kernel, grid, block, 0, s, a, b, out, n);
    } else if (p->B_numel > 0 && p->B_numel < n) {
        return CUDA_KERNEL_LAUNCH(div_f16_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, n, p->B_numel);
    }
    return CUDA_KERNEL_LAUNCH(div_f16_kernel, grid, block, 0, s, a, b, out, n);
}

/* ============================================================ */
/* Registration — avoid name collision with C stdlib mul/sub/div */
/* ============================================================ */
extern "C" int register_relu_f16_cuda(void) {
    static operator_registry_t reg = { "relu_f16_cuda", "f16", relu_f16_cuda, 1, OP_FLAG_IN_PLACE };
    return operator_register(&reg);
}
extern "C" int register_sigmoid_f16_cuda(void) {
    static operator_registry_t reg = { "sigmoid_f16_cuda", "f16", sigmoid_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_gelu_f16_cuda(void) {
    static operator_registry_t reg = { "gelu_f16_cuda", "f16", gelu_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_silu_f16_cuda(void) {
    static operator_registry_t reg = { "silu_f16_cuda", "f16", silu_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_exp_f16_cuda(void) {
    static operator_registry_t reg = { "exp_f16_cuda", "f16", exp_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_add_f16_cuda(void) {
    static operator_registry_t reg = { "add_f16_cuda", "f16", add_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_mul_f16_cuda(void) {
    static operator_registry_t reg = { "mul_f16_cuda", "f16", mul_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_sub_f16_cuda(void) {
    static operator_registry_t reg = { "sub_f16_cuda", "f16", sub_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
extern "C" int register_div_f16_cuda(void) {
    static operator_registry_t reg = { "div_f16_cuda", "f16", div_f16_cuda, 1, OP_FLAG_NONE };
    return operator_register(&reg);
}
