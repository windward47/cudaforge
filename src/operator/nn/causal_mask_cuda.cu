#include "operator.h"
#include "cuda_ops.h"
#include "causal_mask_int.h"
#include <math.h>

__global__ void causal_mask_f32_kernel(float* mask, int64_t S) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = S * S;
    if (idx < total) {
        int64_t i = idx / S;
        int64_t j = idx % S;
        mask[idx] = (j <= i) ? 0.0f : __int_as_float(0xFF800000);  /* -inf */
    }
}

int causal_mask_f32_cuda(const void* inputs[], void* outputs[],
                         const operator_params_t* params, stream_t* stream) {
    if (!outputs || !outputs[0] || !params) return -1;

    const causal_mask_params_t* p = (const causal_mask_params_t*)params;
    float* mask = (float*)outputs[0];
    int64_t S = p->seq_len;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((S * S + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(causal_mask_f32_kernel, grid, block, 0, s, mask, S);
}

extern "C" int register_causal_mask_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "causal_mask_f32_cuda", .data_type = "f32",
        .func = causal_mask_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
