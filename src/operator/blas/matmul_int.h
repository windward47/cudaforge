#ifndef MATMUL_INT_H_
#define MATMUL_INT_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t M, N, K;
    bool transpose_a;
    bool transpose_b;
} matmul_params_t;

/* Tile size for CUDA shared memory tiling */
#define MATMUL_TILE_SIZE 16

#endif /* MATMUL_INT_H_ */
