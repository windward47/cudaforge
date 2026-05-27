#include "operator.h"
#include "squeeze_unsqueeze_int.h"
#include <string.h>

int squeeze_unsqueeze_f32(const void* inputs[], void* outputs[],
                           const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const squeeze_unsqueeze_params_t* p = (const squeeze_unsqueeze_params_t*)params;
    memcpy(outputs[0], inputs[0], (size_t)p->numel * sizeof(float));
    return 0;
}

static const operator_registry_t s_reg = {
    .name = "squeeze_unsqueeze_f32", .data_type = "f32",
    .func = squeeze_unsqueeze_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_squeeze_unsqueeze_f32(void) {
    return operator_register(&s_reg);
}
