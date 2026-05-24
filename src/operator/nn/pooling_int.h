#ifndef POOLING_INT_H_
#define POOLING_INT_H_

#include <stdint.h>

typedef struct {
    int64_t N, C, H, W;
    int64_t kernel_h, kernel_w;
    int64_t stride_h, stride_w;
    int64_t pad_h, pad_w;
} pool_params_t;

#endif /* POOLING_INT_H_ */
