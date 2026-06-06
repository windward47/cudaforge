#include "operator.h"
#include "cuda_ops.h"
#include "softmax_int.h"

/* ---- Templated kernel for different block sizes ---- */
template<int BLOCK_SIZE>
__global__ void softmax_f32_kernel_t(const float* input, float* output,
                                      int64_t C, int64_t N) {
    int n = blockIdx.x;
    if (n >= N) return;

    const float* in_n = input + n * C;
    float* out_n = output + n * C;

    /* Find max (parallel reduction in shared memory) */
    __shared__ float smem[BLOCK_SIZE];
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

/* ---- Config-aware launcher ---- */
static int softmax_launch(cudaStream_t s, const float* in, float* out,
                          int64_t num_classes, int64_t num_blocks, int config) {
    switch (config) {
    case 1: { /* small: 128 threads */
        dim3 block(128, 1, 1);
        dim3 grid((unsigned int)num_blocks, 1, 1);
        return CUDA_KERNEL_LAUNCH(softmax_f32_kernel_t<128>, grid, block, 0, s,
                                  in, out, num_classes, num_blocks);
    }
    case 2: { /* large: 512 threads */
        dim3 block(512, 1, 1);
        dim3 grid((unsigned int)num_blocks, 1, 1);
        return CUDA_KERNEL_LAUNCH(softmax_f32_kernel_t<512>, grid, block, 0, s,
                                  in, out, num_classes, num_blocks);
    }
    default: { /* default: 256 threads */
        dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
        dim3 grid((unsigned int)num_blocks, 1, 1);
        return CUDA_KERNEL_LAUNCH(softmax_f32_kernel_t<256>, grid, block, 0, s,
                                  in, out, num_classes, num_blocks);
    }
    }
}

/* ---- Public entry point ---- */
int softmax_f32_cuda(const void* inputs[], void* outputs[],
                     const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const softmax_params_t* p = (const softmax_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    return softmax_launch(s, in, out, p->num_classes, p->num_blocks, p->tuning_config);
}

extern "C" int register_softmax_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "softmax_f32_cuda", .data_type = "f32",
        .func = softmax_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
