#ifndef MATMUL_INT_H_
#define MATMUL_INT_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t M, N, K;
    bool transpose_a;
    bool transpose_b;
    int64_t batch_size;   /* number of independent matmuls (1 for 2D) */
    int64_t stride_a;     /* elements between batch slices of A */
    int64_t stride_b;     /* elements between batch slices of B */
    int64_t stride_c;     /* elements between batch slices of C */
    int     tuning_config; /* 0=heuristic(default), 1=naive, 2=tiled, 3=warp, 4=tensor_core */
} matmul_params_t;

/* Tile size for CUDA shared memory tiling */
#define MATMUL_TILE_SIZE 16

#endif /* MATMUL_INT_H_ */
