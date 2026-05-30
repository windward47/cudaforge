/* CUDA Where kernel — element-wise ternary select with broadcasting. */
#include "operator.h"
#include "cuda_ops.h"
#include "where_int.h"

__global__ void where_f32_kernel(const float* cond, const float* x,
                                  const float* y, float* out, int64_t n,
                                  int64_t cond_n, int64_t x_n, int64_t y_n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (cond[idx % cond_n] != 0.0f) ? x[idx % x_n] : y[idx % y_n];
    }
}

int where_f32_cuda(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] || !outputs || !outputs[0] || !params)
        return -1;

    const where_params_t* p = (const where_params_t*)params;
    const float* cond = (const float*)inputs[0];
    const float* x    = (const float*)inputs[1];
    const float* y    = (const float*)inputs[2];
    float* out        = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t cond_n = p->cond_numel > 0 ? p->cond_numel : p->numel;
    int64_t x_n = p->x_numel > 0 ? p->x_numel : p->numel;
    int64_t y_n = p->y_numel > 0 ? p->y_numel : p->numel;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((p->numel + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(where_f32_kernel, grid, block, 0, s,
                              cond, x, y, out, p->numel, cond_n, x_n, y_n);
}

extern "C" int register_where_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "where_f32_cuda", .data_type = "f32",
        .func = where_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
