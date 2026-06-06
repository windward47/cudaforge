#ifndef SOFTMAX_INT_H_
#define SOFTMAX_INT_H_

#include <stdint.h>

typedef struct {
    int64_t num_classes;   /* number of classes (softmax axis dim) */
    int64_t num_blocks;    /* number of independent softmax blocks (N * H * W) */
    int     tuning_config; /* 0=default(256), 1=small(128), 2=large(512) */
} softmax_params_t;

#endif /* SOFTMAX_INT_H_ */
