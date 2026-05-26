#ifndef CONCAT_INT_H_
#define CONCAT_INT_H_

#include <stdint.h>

typedef struct {
    int axis;
    int num_inputs;
    int64_t total_numel;
    int64_t H, W;
    int64_t C_total;
    int64_t C_per_input[8];
    int64_t C_offset[8];
} concat_params_t;

#endif /* CONCAT_INT_H_ */
