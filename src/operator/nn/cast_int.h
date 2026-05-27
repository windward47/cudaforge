#ifndef CAST_INT_H_
#define CAST_INT_H_

#include <stdint.h>

/* ONNX TensorProto::DataType */
#define ONNX_DTYPE_FLOAT  1
#define ONNX_DTYPE_INT64  7

typedef struct {
    int     src_dtype;   /* ONNX data type of input */
    int     dst_dtype;   /* ONNX data type of output */
    int64_t numel;
} cast_params_t;

#endif /* CAST_INT_H_ */
