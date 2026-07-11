/* cuda_runtime.h - Shim for HIP compilation
 * When compiled with hipcc, redirects to HIP runtime header.
 * When compiled with nvcc or host compiler, this file is not used
 * (the real cuda_runtime.h from CUDA Toolkit is found instead, because
 *  src/platform/cuda is only on the include path under ENABLE_HIP).
 *
 * NOTE: src/platform/cuda must NOT be on the include path for nvcc builds,
 * otherwise this shim shadows the real cuda_runtime.h. See CMakeLists.txt
 * (the include dir is gated behind ENABLE_HIP).
 */
#ifndef CUDA_RUNTIME_H_SHIM_
#define CUDA_RUNTIME_H_SHIM_

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#else
/* Host-only stubs for .c files compiled with GCC/Clang */
typedef void* cudaEvent_t;
static inline int cudaEventCreate(cudaEvent_t* e) { *e = (void*)0; return 0; }
static inline int cudaEventRecord(cudaEvent_t e, void* s) { (void)e; (void)s; return 0; }
static inline int cudaEventSynchronize(cudaEvent_t e) { (void)e; return 0; }
static inline int cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) { (void)a; (void)b; *ms = 0.0f; return 0; }
static inline int cudaEventDestroy(cudaEvent_t e) { (void)e; return 0; }
#endif

#endif /* CUDA_RUNTIME_H_SHIM_ */
