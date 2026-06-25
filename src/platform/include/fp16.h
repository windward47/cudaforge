#ifndef FP16_H_
#define FP16_H_

/* FP16 conversion utilities for CPU.
 * Uses _Float16 (GCC 9+, Clang 15+) for direct conversion.
 * Falls back to bit-manipulation if _Float16 is unavailable.
 */

#include <stdint.h>
#include <string.h>

#if defined(__FLT16_TYPE__)
/* Native _Float16 support — compiler generates efficient conversions */
typedef _Float16 fp16_t;

static inline fp16_t fp32_to_fp16(float v) { return (fp16_t)v; }
static inline float  fp16_to_fp32(fp16_t v) { return (float)v; }

static inline void fp32_to_fp16_buf(fp16_t* dst, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = (fp16_t)src[i];
}

static inline void fp16_to_fp32_buf(float* dst, const fp16_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = (float)src[i];
}

#else
/* Fallback: IEEE 754 half-precision bit manipulation */
typedef uint16_t fp16_t;

static inline fp16_t fp32_to_fp16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t  exp  = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp <= 0) return (fp16_t)(sign);
    if (exp >= 31) return (fp16_t)(sign | 0x7C00);
    return (fp16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static inline float fp16_to_fp32(fp16_t v) {
    uint32_t sign = ((uint32_t)(v & 0x8000)) << 16;
    uint32_t exp  = ((uint32_t)(v & 0x7C00)) >> 10;
    uint32_t mant = ((uint32_t)(v & 0x03FF)) << 13;
    uint32_t bits;
    if (exp == 0)      bits = sign;
    else if (exp == 31) bits = sign | 0x7F800000 | mant;
    else               bits = sign | ((exp + 112) << 23) | mant;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline void fp32_to_fp16_buf(fp16_t* dst, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = fp32_to_fp16(src[i]);
}

static inline void fp16_to_fp32_buf(float* dst, const fp16_t* src, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_fp32(src[i]);
}

#endif /* __FLT16_TYPE__ */

#endif /* FP16_H_ */
