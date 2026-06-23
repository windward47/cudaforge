/**
 * @file quantize_int.h
 * @brief INT8 block quantization structures.
 *
 * Block quantization: every BLOCK_SIZE elements share one FP32 scale.
 *   quantized[i] = round(float_val[i] / scale)
 *   float_val[i] = quantized[i] * scale
 *
 * scale = max(abs(block)) / 127  (symmetric quantization)
 */
#ifndef QUANTIZE_INT_H_
#define QUANTIZE_INT_H_

#include <stdint.h>

/* Block size for INT8 quantization.
 * 64 elements per block: good balance between precision and overhead.
 * For a 768-dim weight row: 12 blocks. */
#define Q8_BLOCK_SIZE 64

/**
 * Quantized block: one scale + BLOCK_SIZE int8 values.
 * Total size: 4 + 64 = 68 bytes per block.
 * Effective compression: 68 / (64 * 4) = 0.266x vs FP32 (~3.75× compression).
 */
typedef struct {
    float   scale;                    /* dequantization scale */
    int8_t  values[Q8_BLOCK_SIZE];    /* quantized values */
} block_q8_t;

/* Utility: number of blocks for a given element count */
#define Q8_NUM_BLOCKS(n) (((n) + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE)

/* Utility: total quantized size in bytes for n elements */
#define Q8_PACKED_BYTES(n) (Q8_NUM_BLOCKS(n) * sizeof(block_q8_t))

#endif /* QUANTIZE_INT_H_ */
