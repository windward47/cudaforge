#include "operator.h"
#include "cuda_ops.h"
#include "transpose_int.h"

__global__ void transpose_f32_kernel(const float* in, float* out,
                                      int64_t numel,
                                      int64_t in_stride0, int64_t in_stride1,
                                      int64_t in_stride2, int64_t in_stride3,
                                      int64_t out_shape0, int64_t out_shape1,
                                      int64_t out_shape2, int64_t out_shape3,
                                      int64_t perm0, int64_t perm1,
                                      int64_t perm2, int64_t perm3,
                                      int ndim) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= numel) return;

    int64_t out_shape[4] = {out_shape0, out_shape1, out_shape2, out_shape3};
    int64_t perm[4] = {perm0, perm1, perm2, perm3};

    /* Compute output multi-index */
    int64_t tmp = out_idx;
    int64_t out_coord[4] = {0, 0, 0, 0};
    for (int d = ndim - 1; d >= 0; d--) {
        out_coord[d] = tmp % out_shape[d];
        tmp /= out_shape[d];
    }

    /* Map to input coordinates: in_coord[perm[d]] = out_coord[d] */
    int64_t in_coord[4] = {0, 0, 0, 0};
    for (int d = 0; d < ndim; d++)
        in_coord[perm[d]] = out_coord[d];

    int64_t in_stride[4] = {in_stride0, in_stride1, in_stride2, in_stride3};
    int64_t in_idx = 0;
    for (int d = 0; d < ndim; d++)
        in_idx += in_coord[d] * in_stride[d];

    out[out_idx] = in[in_idx];
}

int transpose_f32_cuda(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const transpose_params_t* p = (const transpose_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t numel = 1;
    for (int d = 0; d < p->ndim; d++) numel *= p->shape[d];

    int64_t in_stride[8];
    in_stride[p->ndim - 1] = 1;
    for (int d = p->ndim - 2; d >= 0; d--)
        in_stride[d] = in_stride[d + 1] * p->shape[d + 1];

    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)((numel + 255) / 256), 1, 1);

    CUDA_KERNEL_LAUNCH(transpose_f32_kernel, grid, block, 0, s,
                       in, out, numel,
                       in_stride[0], in_stride[1], in_stride[2], in_stride[3],
                       p->out_shape[0], p->out_shape[1], p->out_shape[2], p->out_shape[3],
                       p->perm[0], p->perm[1], p->perm[2], p->perm[3],
                       (int64_t)p->ndim);
    return 0;
}

extern "C" int register_transpose_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "transpose_f32_cuda", .data_type = "f32",
        .func = transpose_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
