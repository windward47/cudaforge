/* CPU Where operator — element-wise ternary select with broadcasting.
   output[i] = condition[i % cond_numel] ? X[i % x_numel] : Y[i % y_numel]
   condition is float (0.0 = false, non-zero = true). */
#include "operator.h"
#include "where_int.h"

int where_f32(const void* inputs[], void* outputs[],
              const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !inputs[2] || !outputs || !outputs[0] || !params)
        return -1;

    const where_params_t* p = (const where_params_t*)params;
    const float* cond = (const float*)inputs[0];  /* condition (bool as float) */
    const float* x    = (const float*)inputs[1];  /* values where true */
    const float* y    = (const float*)inputs[2];  /* values where false */
    float* out         = (float*)outputs[0];

    int64_t cond_numel = p->cond_numel > 0 ? p->cond_numel : p->numel;
    int64_t x_numel = p->x_numel > 0 ? p->x_numel : p->numel;
    int64_t y_numel = p->y_numel > 0 ? p->y_numel : p->numel;

    for (int64_t i = 0; i < p->numel; i++) {
        float c = cond[i % cond_numel];
        float tv = x[i % x_numel];
        float fv = y[i % y_numel];
        out[i] = (c != 0.0f) ? tv : fv;
    }
    return 0;
}

static const operator_registry_t s_where_reg = {
    .name = "where_f32", .data_type = "f32",
    .func = where_f32, .version = 1, .flags = OP_FLAG_NONE,
};

int register_where_f32(void) {
    return operator_register(&s_where_reg);
}
