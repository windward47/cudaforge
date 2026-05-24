#include "operator.h"

int register_relu_f32(void);
int register_matmul_f32(void);
int register_conv2d_f32(void);
int register_activations(void);
int register_pooling(void);
int register_batchnorm(void);

/* CUDA operator registration (extern "C" from .cu files) */
int register_relu_f32_cuda(void);
int register_matmul_f32_cuda(void);
int register_conv2d_f32_cuda(void);
int register_activations_cuda(void);
int register_pooling_cuda(void);
int register_batchnorm_f32_cuda(void);

int operator_init_all(void) {
    int ret = 0;
    ret += register_relu_f32();
    ret += register_matmul_f32();
    ret += register_conv2d_f32();
    ret += register_activations();
    ret += register_pooling();
    ret += register_batchnorm();
    ret += register_relu_f32_cuda();
    ret += register_matmul_f32_cuda();
    ret += register_conv2d_f32_cuda();
    ret += register_activations_cuda();
    ret += register_pooling_cuda();
    ret += register_batchnorm_f32_cuda();
    return ret;
}
