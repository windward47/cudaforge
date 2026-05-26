#ifndef TRANSPOSE_INT_H_
#define TRANSPOSE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t perm[8];
    int ndim;
    int64_t shape[8];
    int64_t out_shape[8];
} transpose_params_t;

#endif /* TRANSPOSE_INT_H_ */
