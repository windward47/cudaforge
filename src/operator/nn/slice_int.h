#ifndef SLICE_INT_H_
#define SLICE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;           /* total output elements */
    int64_t in_numel;        /* total input elements */
    int     ndim;            /* number of dimensions */
    int64_t in_shape[8];     /* input shape */
    int64_t out_shape[8];    /* output shape */
    int64_t starts[8];       /* start index per dim */
    int64_t steps[8];        /* step per dim */
    int64_t in_strides[8];   /* precomputed input strides */
} slice_params_t;

#endif /* SLICE_INT_H_ */
