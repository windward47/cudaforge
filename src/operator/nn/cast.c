#include "operator.h"
#include "cast_int.h"
#include <stdint.h>
#include <string.h>

int cast_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const cast_params_t* cp = (const cast_params_t*)params;
    int64_t n = cp->numel;

    if (cp->src_dtype == ONNX_DTYPE_INT64 && cp->dst_dtype == ONNX_DTYPE_FLOAT) {
        const int64_t* in = (const int64_t*)inputs[0];
        float* out         = (float*)outputs[0];
        for (int64_t i = 0; i < n; i++) {
            out[i] = (float)in[i];
        }
    } else if (cp->src_dtype == ONNX_DTYPE_FLOAT && cp->dst_dtype == ONNX_DTYPE_INT64) {
        const float* in = (const float*)inputs[0];
        int64_t* out    = (int64_t*)outputs[0];
        for (int64_t i = 0; i < n; i++) {
            out[i] = (int64_t)in[i];
        }
    } else {
        /* Unsupported cast: fallback to memcpy (e.g. F32→F32) */
        memcpy(outputs[0], inputs[0], (size_t)n * sizeof(float));
    }
    return 0;
}

static const operator_registry_t s_cast_reg = {
    .name = "cast_f32", .data_type = "f32",
    .func = cast_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_cast_f32(void) {
    return operator_register(&s_cast_reg);
}
