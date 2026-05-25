#ifndef CONV_INT_H_
#define CONV_INT_H_

#include <stdint.h>

typedef struct {
    int64_t N, C, H, W;        /* input: batch, channels, height, width */
    int64_t K;                  /* output channels (number of filters) */
    int64_t kernel_h, kernel_w;
    int64_t pad_h, pad_w;
    int64_t stride_h, stride_w;
    int64_t dilation_h, dilation_w;
    int64_t groups;
    /* Kernel fusion: if non-zero, apply activation inline after Conv.
       1=ReLU, 2=Sigmoid, 3=GELU (mirrors op_type_t values OP_RELU..OP_GELU). */
    int64_t fuse_activation;
} conv_params_t;

#endif /* CONV_INT_H_ */
