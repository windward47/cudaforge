#include "operator.h"
#include "relu_int.h"
#include "cuda_ops.h"
#include <cuda_runtime.h>

/* ---- Templated kernel for different block sizes ---- */
template<int BLOCK_SIZE>
__global__ void relu_f32_kernel_t(const float* input, float* output, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

/* ---- Config-aware launcher ---- */
static int relu_launch(cudaStream_t s, const float* in, float* out, int64_t n, int config) {
    switch (config) {
    case 1: { /* small: 128 threads */
        dim3 block(128, 1, 1);
        dim3 grid((unsigned int)((n + 127) / 128), 1, 1);
        return CUDA_KERNEL_LAUNCH(relu_f32_kernel_t<128>, grid, block, 0, s, in, out, n);
    }
    case 2: { /* large: 512 threads */
        dim3 block(512, 1, 1);
        dim3 grid((unsigned int)((n + 511) / 512), 1, 1);
        return CUDA_KERNEL_LAUNCH(relu_f32_kernel_t<512>, grid, block, 0, s, in, out, n);
    }
    default: { /* default: 256 threads */
        dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
        dim3 grid((unsigned int)((n + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);
        return CUDA_KERNEL_LAUNCH(relu_f32_kernel_t<256>, grid, block, 0, s, in, out, n);
    }
    }
}

/* ---- Public entry point ---- */
int relu_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;

    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];
    int64_t n        = *(const int64_t*)inputs[1];
    cudaStream_t s   = stream ? (cudaStream_t)stream->cuda_stream : 0;

    /* Read tuning config from params if available, else auto-select from input size */
    int config = 0;
    if (params) {
        const relu_params_t* p = (const relu_params_t*)params;
        config = p->tuning_config;
    } else {
        /* Auto-select: small(1K) → 128 threads, large(1M) → 512 threads */
        if (n < 1024) config = 1;
        else if (n > 1048576) config = 2;
    }

    return relu_launch(s, in, out, n, config);
}

extern "C" int register_relu_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "relu_f32_cuda", .data_type = "f32",
        .func = relu_f32_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    return operator_register(&reg);
}
