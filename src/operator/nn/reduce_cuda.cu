#include "operator.h"
#include "cuda_ops.h"
#include "reduce_int.h"

__global__ void reduce_sum_f32_kernel(const float* input, float* output,
                                        int64_t reduce_size, int64_t num_blocks) {
    extern __shared__ float s_buf[];
    int64_t block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    const float* in_block = input + block_id * reduce_size;

    /* Load into shared memory */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += in_block[i];
    }
    s_buf[threadIdx.x] = sum;
    __syncthreads();

    /* Tree reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_buf[threadIdx.x] += s_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[block_id] = s_buf[0];
    }
}

__global__ void reduce_max_f32_kernel(const float* input, float* output,
                                        int64_t reduce_size, int64_t num_blocks) {
    extern __shared__ float s_buf[];
    int64_t block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    const float* in_block = input + block_id * reduce_size;

    /* Init with first element covered by this thread, or -inf */
    float best = -1.0f / 0.0f;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        if (in_block[i] > best) best = in_block[i];
    }
    s_buf[threadIdx.x] = best;
    __syncthreads();

    /* Tree reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = s_buf[threadIdx.x + stride];
            if (other > s_buf[threadIdx.x]) s_buf[threadIdx.x] = other;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[block_id] = s_buf[0];
    }
}

int reduce_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const reduce_params_t* rp = (const reduce_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int threads = 256;
    if (rp->reduce_size < 256) {
        threads = (int)rp->reduce_size;
        /* Round up to next power of 2 */
        threads--;
        threads |= threads >> 1;
        threads |= threads >> 2;
        threads |= threads >> 4;
        threads |= threads >> 8;
        threads |= threads >> 16;
        threads++;
    }
    dim3 block(threads, 1, 1);
    dim3 grid((unsigned int)rp->num_blocks, 1, 1);
    size_t smem = (size_t)threads * sizeof(float);

    if (rp->op == REDUCE_SUM) {
        CUDA_KERNEL_LAUNCH(reduce_sum_f32_kernel, grid, block, smem, s,
                           (const float*)inputs[0], (float*)outputs[0],
                           rp->reduce_size, rp->num_blocks);
    } else {
        CUDA_KERNEL_LAUNCH(reduce_max_f32_kernel, grid, block, smem, s,
                           (const float*)inputs[0], (float*)outputs[0],
                           rp->reduce_size, rp->num_blocks);
    }
    return 0;
}

extern "C" int register_reduce_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "reduce_f32_cuda", .data_type = "f32",
        .func = reduce_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
