#include "operator.h"
#include "cuda_ops.h"
#include "concat_int.h"

__global__ void concat_f32_kernel(const float* const* inputs, float* out,
                                   int64_t total_numel,
                                   int64_t inner, int64_t C_total,
                                   int num_inputs,
                                   const int64_t* C_per_input,
                                   const int64_t* C_offset) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_numel) return;

    int64_t tmp = out_idx;
    int64_t inner_pos = tmp % inner;
    tmp /= inner;
    int64_t c = tmp % C_total;
    int64_t outer = tmp / C_total;

    /* Find which input this axis position belongs to */
    int ii = 0;
    for (; ii < num_inputs; ii++) {
        if (c < C_offset[ii] + C_per_input[ii]) break;
    }
    int64_t c_local = c - C_offset[ii];
    int64_t Ci = C_per_input[ii];
    int64_t in_idx = outer * Ci * inner + c_local * inner + inner_pos;

    out[out_idx] = inputs[ii][in_idx];
}

int concat_f32_cuda(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    if (!inputs || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const concat_params_t* p = (const concat_params_t*)params;
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    /* Build device array of input pointers */
    const float* d_inputs[8] = {NULL};
    for (int ii = 0; ii < p->num_inputs && ii < 8; ii++)
        d_inputs[ii] = (const float*)inputs[ii];

    /* Copy input pointer array and metadata to device */
    const float** d_input_ptrs =
        (const float**)g_cuda.device_alloc((size_t)p->num_inputs * sizeof(float*));
    g_cuda.memcpy_h2d(d_input_ptrs, d_inputs, (size_t)p->num_inputs * sizeof(float*), s);

    int64_t* d_C_per_input =
        (int64_t*)g_cuda.device_alloc((size_t)p->num_inputs * sizeof(int64_t));
    g_cuda.memcpy_h2d(d_C_per_input, p->C_per_input, (size_t)p->num_inputs * sizeof(int64_t), s);

    int64_t* d_C_offset =
        (int64_t*)g_cuda.device_alloc((size_t)p->num_inputs * sizeof(int64_t));
    g_cuda.memcpy_h2d(d_C_offset, p->C_offset, (size_t)p->num_inputs * sizeof(int64_t), s);

    int64_t total = p->total_numel;
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    int ret = CUDA_KERNEL_LAUNCH(concat_f32_kernel, grid, block, 0, s,
                       d_input_ptrs, out, total,
                       p->inner, p->C_total, p->num_inputs,
                       d_C_per_input, d_C_offset);

    g_cuda.stream_synchronize(s);
    g_cuda.device_free((void*)d_input_ptrs);
    g_cuda.device_free((void*)d_C_per_input);
    g_cuda.device_free((void*)d_C_offset);
    return ret;
}

extern "C" int register_concat_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "concat_f32_cuda", .data_type = "f32",
        .func = concat_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
