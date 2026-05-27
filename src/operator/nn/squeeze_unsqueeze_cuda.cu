#include "operator.h"
#include "cuda_ops.h"
#include "squeeze_unsqueeze_int.h"

int squeeze_unsqueeze_f32_cuda(const void* inputs[], void* outputs[],
                                const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const squeeze_unsqueeze_params_t* p = (const squeeze_unsqueeze_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;
    cudaError_t err = cudaMemcpyAsync(outputs[0], inputs[0],
                                       (size_t)p->numel * sizeof(float),
                                       cudaMemcpyDeviceToDevice, s);
    return err != cudaSuccess ? (int)err : 0;
}

extern "C" int register_squeeze_unsqueeze_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "squeeze_unsqueeze_f32_cuda", .data_type = "f32",
        .func = squeeze_unsqueeze_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
