/* FP16 Softmax CUDA kernel.
   FP16 input/output, FP32 accumulation for numerical stability.
   Registered as "softmax_f16_cuda". */
#include "operator.h"
#include "cuda_ops.h"
#include "softmax_int.h"
#include <cuda_fp16.h>

__global__ void softmax_f16_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int64_t num_blocks, int64_t block_size)
{
    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    const __half* in_row = input + block_idx * block_size;
    __half* out_row = output + block_idx * block_size;

    /* Find max for numerical stability */
    float local_max = -1e38f;
    for (int i = tid; i < block_size; i += nthreads) {
        float v = __half2float(in_row[i]);
        if (v > local_max) local_max = v;
    }

    /* Reduce max across threads (simple sequential for small block_size) */
    if (tid == 0) {
        float row_max = -1e38f;
        for (int64_t i = 0; i < block_size; i++) {
            float v = __half2float(in_row[i]);
            if (v > row_max) row_max = v;
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (int64_t i = 0; i < block_size; i++) {
            float v = __expf(__half2float(in_row[i]) - row_max);
            sum += v;
            out_row[i] = __float2half(v);  /* temp: store exp values */
        }

        /* Normalize */
        if (sum < 1e-12f) sum = 1e-12f;
        float inv = 1.0f / sum;
        for (int64_t i = 0; i < block_size; i++) {
            out_row[i] = __float2half(__half2float(out_row[i]) * inv);
        }
    }
}

int softmax_f16_cuda(const void* inputs[], void* outputs[],
                      const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const softmax_params_t* p = (const softmax_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 grid((unsigned int)p->num_blocks, 1, 1);
    dim3 block(256, 1, 1);

    return CUDA_KERNEL_LAUNCH(softmax_f16_kernel, grid, block, 0, s,
                              (const __half*)inputs[0],
                              (__half*)outputs[0],
                              p->num_blocks, p->num_classes);
}

extern "C" int register_softmax_f16_cuda(void) {
    static operator_registry_t reg = {
        .name = "softmax_f16_cuda", .data_type = "f16",
        .func = softmax_f16_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
