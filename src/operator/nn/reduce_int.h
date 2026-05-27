#ifndef REDUCE_INT_H_
#define REDUCE_INT_H_

#include <stdint.h>

typedef enum { REDUCE_SUM = 0, REDUCE_MAX = 1 } reduce_op_t;

typedef struct {
    int64_t reduce_size;   /* product of axes to reduce */
    int64_t num_blocks;    /* product of all other dims */
    int64_t total_elems;   /* input numel */
    int     op;            /* REDUCE_SUM or REDUCE_MAX */
} reduce_params_t;

#endif /* REDUCE_INT_H_ */
