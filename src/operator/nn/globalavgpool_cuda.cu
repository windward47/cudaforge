#include "operator.h"
#include "cuda_ops.h"
#include "globalavgpool_int.h"

__global__ void globalavgpool_f32_kernel(const float* input, float* output,
                                           int64_t C, int64_t H, int64_t W) {
    int n = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;

    int64_t spatial = H * W;

    __shared__ float smem[256];
    float sum = 0.0f;

    for (int64_t i = tid; i < spatial; i += blockDim.x) {
        int h = (int)(i / W);
        int w = (int)(i % W);
        sum += input[((n * C + c) * H + h) * W + w];
    }

    smem[tid] = sum;
    __syncthreads();

    /* Parallel reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        output[n * C + c] = smem[0] / (float)spatial;
    }
}

int globalavgpool_f32_cuda(const void* inputs[], void* outputs[],
                           const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const globalavgpool_params_t* p = (const globalavgpool_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)p->C, (unsigned int)p->N, 1);

    return CUDA_KERNEL_LAUNCH(globalavgpool_f32_kernel, grid, block, 0, s,
                       in, out, p->C, p->H, p->W);
}

extern "C" int register_globalavgpool_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "globalavgpool_f32_cuda", .data_type = "f32",
        .func = globalavgpool_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
