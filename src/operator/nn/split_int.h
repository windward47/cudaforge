#ifndef SPLIT_INT_H_
#define SPLIT_INT_H_

#include <stdint.h>

typedef struct {
    int     axis;            /* axis to split along */
    int     num_outputs;     /* number of output tensors */
    int     ndim;            /* number of dimensions */
    int64_t in_shape[8];     /* input shape */
    int64_t splits[8];       /* size of each split along axis */
    int64_t offsets[8];      /* cumulative element offset in input for each split */
    int64_t out_numel[8];    /* total elements per output */
} split_params_t;

#endif /* SPLIT_INT_H_ */
