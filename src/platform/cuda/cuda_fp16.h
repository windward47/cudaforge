/* cuda_fp16.h — Shim for HIP compilation */
#ifndef CUDA_FP16_H_SHIM_
#define CUDA_FP16_H_SHIM_
#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#else
#error "This shim should only be used with hipcc"
#endif
#endif
