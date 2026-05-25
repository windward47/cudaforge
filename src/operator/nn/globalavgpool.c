#include "operator.h"
#include "globalavgpool_int.h"
#include <stddef.h>
#include <string.h>

int globalavgpool_f32(const void* inputs[], void* outputs[],
                      const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const globalavgpool_params_t* p = (const globalavgpool_params_t*)params;
    const float* in = (const float*)inputs[0];
    float* out = (float*)outputs[0];

    int64_t N = p->N, C = p->C, H = p->H, W = p->W;
    float scale = 1.0f / (float)(H * W);

    memset(out, 0, (size_t)N * C * sizeof(float));
    for (int64_t n = 0; n < N; n++) {
        for (int64_t c = 0; c < C; c++) {
            float sum = 0.0f;
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    sum += in[((n * C + c) * H + h) * W + w];
                }
            }
            out[n * C + c] = sum * scale;
        }
    }
    return 0;
}

static const operator_registry_t s_globalavgpool_reg = {
    .name      = "globalavgpool_f32",
    .data_type = "f32",
    .func      = globalavgpool_f32,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

int register_globalavgpool_f32(void) {
    return operator_register(&s_globalavgpool_reg);
}
