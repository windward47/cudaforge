#include "operator.h"
#include "mul_int.h"
#include <stddef.h>

int mul_f32(const void* inputs[], void* outputs[],
            const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const mul_params_t* p = (const mul_params_t*)params;
    const float* a = (const float*)inputs[0];
    const float* b = (const float*)inputs[1];
    float* out = (float*)outputs[0];

    int64_t N = p->numel;
    int64_t BN = p->B_numel;

    if (BN == 1) {
        float bv = b[0];
        for (int64_t i = 0; i < N; i++) out[i] = a[i] * bv;
    } else {
        int64_t blocks = N / BN;
        for (int64_t blk = 0; blk < blocks; blk++) {
            for (int64_t i = 0; i < BN; i++) {
                out[blk * BN + i] = a[blk * BN + i] * b[i];
            }
        }
    }
    return 0;
}

static const operator_registry_t s_mul_reg = {
    .name      = "mul_f32",
    .data_type = "f32",
    .func      = mul_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_mul_f32(void) {
    return operator_register(&s_mul_reg);
}
