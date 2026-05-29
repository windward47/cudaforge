/* CPU RoPE — Rotary Position Encoding.
   Applies rotation to Q and K tensors in-place.
   inputs[0] = Q or K tensor (B, S, H, d) float32
   outputs[0] = same tensor with RoPE applied (in-place)

   For each position pos and dimension pair (2i, 2i+1):
     angle = pos / (base ^ (2i / d))
     q'[2i]   = q[2i] * cos(angle) - q[2i+1] * sin(angle)
     q'[2i+1] = q[2i] * sin(angle) + q[2i+1] * cos(angle)
*/
#include "operator.h"
#include "rope_int.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int rope_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const rope_params_t* p = (const rope_params_t*)params;
    int64_t S = p->seq_len, d = p->head_dim, H = p->num_heads;
    float base = p->base;

    /* Copy input to output if not in-place */
    if (inputs[0] != outputs[0]) {
        /* Assume B=1 for simplicity; caller handles batching */
        memcpy(outputs[0], inputs[0], S * H * d * sizeof(float));
    }

    float* data = (float*)outputs[0];

    /* Precompute sin/cos table: sin_table[pos * (d/2)], cos_table[pos * (d/2)] */
    int64_t half_d = d / 2;
    float* cos_table = (float*)malloc((size_t)S * half_d * sizeof(float));
    float* sin_table = (float*)malloc((size_t)S * half_d * sizeof(float));
    if (!cos_table || !sin_table) { free(cos_table); free(sin_table); return -1; }

    for (int64_t pos = 0; pos < S; pos++) {
        for (int64_t i = 0; i < half_d; i++) {
            float angle = (float)pos / powf(base, (float)(2 * i) / (float)d);
            cos_table[pos * half_d + i] = cosf(angle);
            sin_table[pos * half_d + i] = sinf(angle);
        }
    }

    /* Apply rotation: for each (pos, h, pair) */
    for (int64_t pos = 0; pos < S; pos++) {
        for (int64_t h = 0; h < H; h++) {
            float* q = data + (pos * H + h) * d;
            for (int64_t i = 0; i < half_d; i++) {
                float c = cos_table[pos * half_d + i];
                float s = sin_table[pos * half_d + i];
                float x0 = q[2 * i];
                float x1 = q[2 * i + 1];
                q[2 * i]     = x0 * c - x1 * s;
                q[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }

    free(cos_table);
    free(sin_table);
    return 0;
}

static const operator_registry_t s_rope_reg = {
    .name = "rope_f32", .data_type = "f32",
    .func = rope_f32, .version = 1, .flags = OP_FLAG_IN_PLACE,
};

int register_rope_f32(void) {
    return operator_register(&s_rope_reg);
}
