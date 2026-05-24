#ifndef BATCHNORM_INT_H_
#define BATCHNORM_INT_H_

#include <stdint.h>

typedef struct {
    int64_t C;
    float epsilon;
} batchnorm_params_t;

#endif /* BATCHNORM_INT_H_ */
