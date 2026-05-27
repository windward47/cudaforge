#include "operator.h"
#include "cuda_ops.h"
#include "gather_int.h"

/* One thread per output element.  Indices must be on device. */
__global__ void gather_f32_kernel(const float* data, const float* indices,
                                   float* out,
                                   int64_t num_indices, int64_t block_size,
                                   int64_t outer_size, int64_t inner_size) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = outer_size * num_indices * block_size;
    if (idx >= total) return;

    /* Flat output index → (outer, index_pos, block_elem) */
    int64_t tmp = idx;
    int64_t be  = tmp % block_size;  tmp /= block_size;
    int64_t ip  = tmp % num_indices; tmp /= num_indices;
    int64_t o   = tmp;

    int64_t axis_dim = inner_size / block_size;
    int64_t src_idx = (int64_t)indices[ip];
    if (src_idx < 0) src_idx += axis_dim;

    int64_t in_idx = o * inner_size + src_idx * block_size + be;
    out[idx] = data[in_idx];
}

int gather_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const gather_params_t* p = (const gather_params_t*)params;
    const float* data         = (const float*)inputs[0];
    const float* indices       = (const float*)inputs[1];
    float* out                 = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t total = p->outer_size * p->num_indices * p->block_size;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(gather_f32_kernel, grid, block, 0, s,
                               data, indices, out,
                               p->num_indices, p->block_size,
                               p->outer_size, p->inner_size);
}

extern "C" int register_gather_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "gather_f32_cuda", .data_type = "f32",
        .func = gather_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
