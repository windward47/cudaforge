#include "operator.h"
#include "cuda_ops.h"
#include "add_int.h"

__global__ void add_f32_no_broadcast_kernel(const float* a, const float* b,
                                              float* out, int64_t N) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

__global__ void add_f32_broadcast_c_kernel(const float* a, const float* b,
                                             float* out, int64_t N, int64_t C) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int64_t c = i % C;
        out[i] = a[i] + b[c];
    }
}

int add_f32_cuda(const void* inputs[], void* outputs[],
                 const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const add_params_t* p = (const add_params_t*)params;
    const float* a = (const float*)inputs[0];
    const float* b = (const float*)inputs[1];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t N = p->numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((N + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (p->B_numel == 1) {
        float bv;
        g_cuda.memcpy_d2h(&bv, (void*)b, sizeof(float), 0);
        g_cuda.stream_synchronize(0);
        return CUDA_KERNEL_LAUNCH(add_f32_no_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, N);
    } else if (p->B_numel == N) {
        return CUDA_KERNEL_LAUNCH(add_f32_no_broadcast_kernel, grid, block, 0, s,
                                  a, b, out, N);
    }
    return CUDA_KERNEL_LAUNCH(add_f32_broadcast_c_kernel, grid, block, 0, s,
                      a, b, out, N, p->B_numel);
}

extern "C" int register_add_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "add_f32_cuda", .data_type = "f32",
        .func = add_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
