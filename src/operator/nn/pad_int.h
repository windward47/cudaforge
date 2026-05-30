#ifndef PAD_INT_H_
#define PAD_INT_H_

#include <stdint.h>

/* ONNX Pad modes */
#define PAD_MODE_CONSTANT 0
#define PAD_MODE_EDGE     1
#define PAD_MODE_REFLECT  2

typedef struct {
    int64_t N;          /* batch size */
    int64_t C;          /* channels */
    int64_t H;          /* input height */
    int64_t W;          /* input width */
    int64_t pad_h_begin;
    int64_t pad_h_end;
    int64_t pad_w_begin;
    int64_t pad_w_end;
    int     mode;       /* PAD_MODE_* */
    float   value;      /* constant fill value (mode=0 only) */
} pad_params_t;

#endif /* PAD_INT_H_ */
