#ifndef ARGMAX_INT_H_
#define ARGMAX_INT_H_

#include <stdint.h>

typedef struct {
    int64_t reduce_size;   /* size of the axis to argmax over */
    int64_t num_blocks;    /* product of all other dims */
} argmax_params_t;

#endif /* ARGMAX_INT_H_ */
