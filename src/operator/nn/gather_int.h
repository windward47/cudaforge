#ifndef GATHER_INT_H_
#define GATHER_INT_H_

#include <stdint.h>

typedef struct {
    int64_t axis;        /* axis to gather along */
    int64_t num_indices; /* number of indices to gather */
    int64_t block_size;  /* num elements per gather slice (product of dims after axis) */
    int64_t outer_size;  /* num slices before axis */
    int64_t inner_size;  /* num elements per axis slice (= block_size * input_axis_dim) */
    int64_t out_axis_dim;/* output size along axis (= num_indices per outer) */
} gather_params_t;

#endif /* GATHER_INT_H_ */
