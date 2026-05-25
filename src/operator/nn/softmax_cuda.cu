#include "operator.h"
#include "cuda_ops.h"
#include "softmax_int.h"

__global__ void softmax_f32_kernel(const float* input, float* output,
                                     int64_t C, int64_t N) {
    int n = blockIdx.x;
    if (n >= N) return;

    const float* in_n = input + n * C;
    float* out_n = output + n * C;

    /* Find max (parallel reduction in shared memory) */
    __shared__ float smem[256];
    int tid = threadIdx.x;
    float max_val = -3.4028235e38f;
    for (int i = tid; i < C; i += blockDim.x) {
        if (in_n[i] > max_val) max_val = in_n[i];
    }
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    /* Compute exp and sum */
    float sum = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float v = expf(in_n[i] - max_val);
        out_n[i] = v;
        sum += v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    sum = smem[0];

    /* Normalize */
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (int i = tid; i < C; i += blockDim.x) {
            out_n[i] *= inv;
        }
    }
}

int softmax_f32_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const softmax_params_t* p = (const softmax_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)p->num_blocks, 1, 1);

    CUDA_KERNEL_LAUNCH(softmax_f32_kernel, grid, block, 0, s,
                       in, out, p->num_classes, p->num_blocks);
    return 0;
}

extern "C" int register_softmax_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "softmax_f32_cuda", .data_type = "f32",
        .func = softmax_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
