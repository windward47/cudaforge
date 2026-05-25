#ifndef RESHAPE_INT_H_
#define RESHAPE_INT_H_

#include <stdint.h>
#include "platform.h"

typedef struct {
    int64_t numel;          /* total elements (must match input) */
    int     ndim;           /* target number of dimensions */
    int64_t shape[MAX_TENSOR_DIMS];  /* target shape */
} reshape_params_t;

#endif /* RESHAPE_INT_H_ */
