#include "operator.h"
#include "cuda_ops.h"
#include "cast_int.h"

__global__ void cast_i64_to_f32_kernel(const int64_t* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (float)input[idx];
    }
}

__global__ void cast_f32_to_i64_kernel(const float* input, int64_t* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = (int64_t)input[idx];
    }
}

int cast_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const cast_params_t* cp = (const cast_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((cp->numel + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    if (cp->src_dtype == ONNX_DTYPE_INT64 && cp->dst_dtype == ONNX_DTYPE_FLOAT) {
        CUDA_KERNEL_LAUNCH(cast_i64_to_f32_kernel, grid, block, 0, s,
                           (const int64_t*)inputs[0], (float*)outputs[0], cp->numel);
    } else if (cp->src_dtype == ONNX_DTYPE_FLOAT && cp->dst_dtype == ONNX_DTYPE_INT64) {
        CUDA_KERNEL_LAUNCH(cast_f32_to_i64_kernel, grid, block, 0, s,
                           (const float*)inputs[0], (int64_t*)outputs[0], cp->numel);
    } else {
        /* Unsupported casts (e.g. F32→F32): fallback to memcpy (matches CPU behavior) */
        cudaMemcpyAsync(outputs[0], inputs[0],
                        (size_t)cp->numel * sizeof(float),
                        cudaMemcpyDeviceToDevice, s);
    }
    return 0;
}

extern "C" int register_cast_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "cast_f32_cuda", .data_type = "f32",
        .func = cast_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
