#include "operator.h"
#include "cast_int.h"
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Software FP16 encode/decode (IEEE 754 half-precision) */
static uint16_t f32_to_f16_bits(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)(sign);  /* flush to zero */
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);  /* inf/nan */
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    int32_t exp = ((h >> 10) & 0x1F);
    uint32_t mant = ((uint32_t)(h & 0x3FF)) << 13;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }  /* zero */
        else {  /* denormalized */
            while (!(mant & 0x800000)) { mant <<= 1; exp--; }
            exp++; mant &= ~0x800000;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | mant;
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | mant;  /* inf/nan */
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | mant;
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

int cast_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const cast_params_t* cp = (const cast_params_t*)params;
    int64_t n = cp->numel;

    if (cp->src_dtype == ONNX_DTYPE_INT64 && cp->dst_dtype == ONNX_DTYPE_FLOAT) {
        const int64_t* in = (const int64_t*)inputs[0];
        float* out         = (float*)outputs[0];
        for (int64_t i = 0; i < n; i++) out[i] = (float)in[i];
    } else if (cp->src_dtype == ONNX_DTYPE_FLOAT && cp->dst_dtype == ONNX_DTYPE_INT64) {
        const float* in = (const float*)inputs[0];
        int64_t* out    = (int64_t*)outputs[0];
        for (int64_t i = 0; i < n; i++) out[i] = (int64_t)in[i];
    } else if (cp->src_dtype == ONNX_DTYPE_FLOAT && cp->dst_dtype == ONNX_DTYPE_FLOAT16) {
        const float* in = (const float*)inputs[0];
        uint16_t* out   = (uint16_t*)outputs[0];
        for (int64_t i = 0; i < n; i++) out[i] = f32_to_f16_bits(in[i]);
    } else if (cp->src_dtype == ONNX_DTYPE_FLOAT16 && cp->dst_dtype == ONNX_DTYPE_FLOAT) {
        const uint16_t* in = (const uint16_t*)inputs[0];
        float* out         = (float*)outputs[0];
        for (int64_t i = 0; i < n; i++) out[i] = f16_bits_to_f32(in[i]);
    } else {
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
