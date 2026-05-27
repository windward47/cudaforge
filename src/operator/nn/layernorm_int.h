#ifndef LAYERNORM_INT_H_
#define LAYERNORM_INT_H_

#include <stdint.h>

typedef struct {
    int64_t N;               /* batch dims (product of dims before normalized dim) */
    int64_t normalized_size; /* size of the normalized dimension (e.g. 768) */
    float   epsilon;
} layernorm_params_t;

#endif /* LAYERNORM_INT_H_ */
