#include "operator.h"
#include "causal_mask_int.h"
#include <math.h>

/* CPU causal mask: output (S, S) with 0 on/below diagonal, -inf above.
   inputs[0] = unused (placeholder for graph compatibility)
   outputs[0] = mask (S, S) float32 */
int causal_mask_f32(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream) {
    (void)inputs; (void)stream;
    if (!outputs || !outputs[0] || !params) return -1;

    const causal_mask_params_t* p = (const causal_mask_params_t*)params;
    int64_t S = p->seq_len;
    float* mask = (float*)outputs[0];

    for (int64_t i = 0; i < S; i++) {
        for (int64_t j = 0; j < S; j++) {
            mask[i * S + j] = (j <= i) ? 0.0f : -INFINITY;
        }
    }
    return 0;
}

static const operator_registry_t s_causal_mask_reg = {
    .name = "causal_mask_f32", .data_type = "f32",
    .func = causal_mask_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_causal_mask_f32(void) {
    return operator_register(&s_causal_mask_reg);
}
