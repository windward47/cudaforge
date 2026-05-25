#ifndef SOFTMAX_INT_H_
#define SOFTMAX_INT_H_

#include <stdint.h>

typedef struct {
    int64_t num_classes;   /* number of classes (softmax axis dim) */
    int64_t num_blocks;    /* number of independent softmax blocks (N * H * W) */
} softmax_params_t;

#endif /* SOFTMAX_INT_H_ */
