#ifndef CONCAT_INT_H_
#define CONCAT_INT_H_

#include <stdint.h>

typedef struct {
    int axis;
    int ndim;
    int num_inputs;
    int64_t total_numel;
    int64_t outer;            /* product of dims before axis */
    int64_t inner;            /* product of dims after axis (= stride of one element along axis) */
    int64_t C_total;          /* total size along concat axis */
    int64_t C_per_input[8];   /* size of each input along concat axis */
    int64_t C_offset[8];      /* cumulative offset along concat axis */
} concat_params_t;

#endif /* CONCAT_INT_H_ */
