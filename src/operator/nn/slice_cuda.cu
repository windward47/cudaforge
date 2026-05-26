#include "operator.h"
#include "cuda_ops.h"
#include "slice_int.h"

__global__ void slice_f32_kernel(const float* in, float* out,
                                  int64_t numel, int ndim,
                                  int64_t out_shape0, int64_t out_shape1,
                                  int64_t out_shape2, int64_t out_shape3,
                                  int64_t out_shape4, int64_t out_shape5,
                                  int64_t out_shape6, int64_t out_shape7,
                                  int64_t start0, int64_t start1,
                                  int64_t start2, int64_t start3,
                                  int64_t start4, int64_t start5,
                                  int64_t start6, int64_t start7,
                                  int64_t step0, int64_t step1,
                                  int64_t step2, int64_t step3,
                                  int64_t step4, int64_t step5,
                                  int64_t step6, int64_t step7,
                                  int64_t stride0, int64_t stride1,
                                  int64_t stride2, int64_t stride3,
                                  int64_t stride4, int64_t stride5,
                                  int64_t stride6, int64_t stride7) {
    int64_t oi = blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= numel) return;

    int64_t out_shape[8] = {out_shape0, out_shape1, out_shape2, out_shape3,
                            out_shape4, out_shape5, out_shape6, out_shape7};
    int64_t starts[8] = {start0, start1, start2, start3,
                         start4, start5, start6, start7};
    int64_t steps[8] = {step0, step1, step2, step3,
                        step4, step5, step6, step7};
    int64_t strides[8] = {stride0, stride1, stride2, stride3,
                          stride4, stride5, stride6, stride7};

    int64_t remaining = oi;
    int64_t in_idx = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int64_t coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        in_idx += (starts[d] + coord * steps[d]) * strides[d];
    }
    out[oi] = in[in_idx];
}

int slice_f32_cuda(const void* inputs[], void* outputs[],
                   const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const slice_params_t* p = (const slice_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((p->numel + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(slice_f32_kernel, grid, block, 0, s,
        in, out, p->numel, p->ndim,
        p->out_shape[0], p->out_shape[1], p->out_shape[2], p->out_shape[3],
        p->out_shape[4], p->out_shape[5], p->out_shape[6], p->out_shape[7],
        p->starts[0], p->starts[1], p->starts[2], p->starts[3],
        p->starts[4], p->starts[5], p->starts[6], p->starts[7],
        p->steps[0], p->steps[1], p->steps[2], p->steps[3],
        p->steps[4], p->steps[5], p->steps[6], p->steps[7],
        p->in_strides[0], p->in_strides[1], p->in_strides[2], p->in_strides[3],
        p->in_strides[4], p->in_strides[5], p->in_strides[6], p->in_strides[7]);
}

extern "C" int register_slice_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "slice_f32_cuda", .data_type = "f32",
        .func = slice_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
