#include "operator.h"
#include "cuda_ops.h"
#include "argmax_int.h"

__global__ void argmax_f32_kernel(const float* input, float* output,
                                    int64_t reduce_size, int64_t num_blocks) {
    extern __shared__ float s_buf[];
    float* s_val = s_buf;
    float* s_idx = s_buf + blockDim.x;  /* reuse shared mem for (val, idx) pairs */

    int64_t block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    const float* in_block = input + block_id * reduce_size;

    float best_val = -1.0f / 0.0f;
    float best_idx = 0.0f;
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        if (in_block[i] > best_val) {
            best_val = in_block[i];
            best_idx = (float)i;
        }
    }
    s_val[threadIdx.x] = best_val;
    s_idx[threadIdx.x] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_val[threadIdx.x + stride] > s_val[threadIdx.x]) {
                s_val[threadIdx.x] = s_val[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[block_id] = s_idx[0];
    }
}

int argmax_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const argmax_params_t* ap = (const argmax_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int threads = 256;
    if (ap->reduce_size < 256) {
        threads = (int)ap->reduce_size;
        threads--;
        threads |= threads >> 1;
        threads |= threads >> 2;
        threads |= threads >> 4;
        threads |= threads >> 8;
        threads |= threads >> 16;
        threads++;
    }
    dim3 block(threads, 1, 1);
    dim3 grid((unsigned int)ap->num_blocks, 1, 1);
    size_t smem = (size_t)threads * 2 * sizeof(float);

    CUDA_KERNEL_LAUNCH(argmax_f32_kernel, grid, block, smem, s,
                       (const float*)inputs[0], (float*)outputs[0],
                       ap->reduce_size, ap->num_blocks);
    return 0;
}

extern "C" int register_argmax_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "argmax_f32_cuda", .data_type = "f32",
        .func = argmax_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
