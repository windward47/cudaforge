/* CPU Clip operator — clamp values to [min, max].
   Equivalent to ONNX Clip op. */
#include "operator.h"
#include "clip_int.h"

int clip_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const clip_params_t* p = (const clip_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    for (int64_t i = 0; i < p->numel; i++) {
        float v = in[i];
        if (v < p->min_val) v = p->min_val;
        if (v > p->max_val) v = p->max_val;
        out[i] = v;
    }
    return 0;
}

static const operator_registry_t s_clip_reg = {
    .name = "clip_f32", .data_type = "f32",
    .func = clip_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_clip_f32(void) {
    return operator_register(&s_clip_reg);
}
