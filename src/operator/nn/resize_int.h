#ifndef RESIZE_INT_H_
#define RESIZE_INT_H_

#include <stdint.h>

typedef struct {
    float scale_h;
    float scale_w;
    int64_t N, C, H_in, W_in, H_out, W_out;
} resize_params_t;

#endif /* RESIZE_INT_H_ */
