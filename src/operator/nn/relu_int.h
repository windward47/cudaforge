#ifndef RELU_INT_H_
#define RELU_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;         /* number of elements */
    int     tuning_config; /* 0=default(256), 1=small(128), 2=large(512) */
} relu_params_t;

#endif /* RELU_INT_H_ */
