/**
 * @file quantize.c
 * @brief INT8 block quantization — CPU reference implementation.
 *
 * Symmetric quantization: scale = max(abs(block)) / 127
 *   quantized[i] = clamp(round(val[i] / scale), -128, 127)
 *   dequantized[i] = quantized[i] * scale
 */
#include "quantize_int.h"
#include <math.h>
#include <string.h>

/**
 * Quantize FP32 array to INT8 block format.
 *
 * @param src      Input FP32 array (n elements)
 * @param dst      Output block_q8_t array (Q8_NUM_BLOCKS(n) blocks)
 * @param n        Number of input elements
 */
void quantize_f32_to_q8(const float* src, block_q8_t* dst, int64_t n) {
    int64_t num_blocks = Q8_NUM_BLOCKS(n);
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t offset = b * Q8_BLOCK_SIZE;
        int64_t count = n - offset;
        if (count > Q8_BLOCK_SIZE) count = Q8_BLOCK_SIZE;

        /* Find max absolute value in this block */
        float max_abs = 0.0f;
        for (int64_t i = 0; i < count; i++) {
            float v = src[offset + i];
            float a = v < 0.0f ? -v : v;
            if (a > max_abs) max_abs = a;
        }

        /* Compute scale */
        float scale = (max_abs > 1e-12f) ? (max_abs / 127.0f) : 1e-12f;
        dst[b].scale = scale;

        /* Quantize */
        float inv_scale = 1.0f / scale;
        for (int64_t i = 0; i < count; i++) {
            float v = src[offset + i] * inv_scale;
            int q = (int)roundf(v);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            dst[b].values[i] = (int8_t)q;
        }
        /* Zero-pad remainder */
        for (int64_t i = count; i < Q8_BLOCK_SIZE; i++)
            dst[b].values[i] = 0;
    }
}

/**
 * Dequantize INT8 block format back to FP32.
 *
 * @param src      Input block_q8_t array
 * @param dst      Output FP32 array (n elements)
 * @param n        Number of output elements
 */
void dequantize_q8_to_f32(const block_q8_t* src, float* dst, int64_t n) {
    int64_t num_blocks = Q8_NUM_BLOCKS(n);
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t offset = b * Q8_BLOCK_SIZE;
        int64_t count = n - offset;
        if (count > Q8_BLOCK_SIZE) count = Q8_BLOCK_SIZE;

        float scale = src[b].scale;
        for (int64_t i = 0; i < count; i++)
            dst[offset + i] = (float)src[b].values[i] * scale;
    }
}
