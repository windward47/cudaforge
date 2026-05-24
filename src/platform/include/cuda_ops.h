#ifndef CUDA_OPS_H_
#define CUDA_OPS_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 *  CUDA error check — only meaningful when compiled by nvcc
 * -------------------------------------------------------------------------- */
#ifdef __CUDACC__
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) (void)(call)
#endif

/* --------------------------------------------------------------------------
 *  CUDA types — when compiled by the host C compiler, use forward decls
 * -------------------------------------------------------------------------- */
#ifndef __CUDACC__
typedef void* cudaStream_t;
struct cudaDeviceProp;
#endif

/* Grid/block dimension helper */
#ifdef __CUDACC__
typedef struct {
    dim3 grid_dim;
    dim3 block_dim;
} kernel_config_t;

static inline kernel_config_t cuda_configure_1d(int64_t n, int threads) {
    kernel_config_t cfg;
    cfg.block_dim = dim3(threads, 1, 1);
    cfg.grid_dim  = dim3((unsigned int)((n + threads - 1) / threads), 1, 1);
    return cfg;
}
#else
typedef struct {
    struct { unsigned int x, y, z; } block_dim;
    struct { unsigned int x, y, z; } grid_dim;
} kernel_config_t;

static inline kernel_config_t cuda_configure_1d(int64_t n, int threads) {
    kernel_config_t cfg;
    cfg.block_dim.x = (unsigned int)threads; cfg.block_dim.y = 1; cfg.block_dim.z = 1;
    cfg.grid_dim.x  = (unsigned int)((n + threads - 1) / threads); cfg.grid_dim.y = 1; cfg.grid_dim.z = 1;
    return cfg;
}
#endif

/* CUDA platform ops — function pointer table */
typedef struct {
    int   (*init)(int device_id);
    void  (*finalize)(void);
    void* (*device_alloc)(size_t size);
    void  (*device_free)(void* ptr);
    void* (*host_alloc_pinned)(size_t size);
    void  (*host_free_pinned)(void* ptr);
    int   (*memcpy_h2d)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*memcpy_d2h)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*memcpy_d2d)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*stream_create)(cudaStream_t* stream);
    int   (*stream_synchronize)(cudaStream_t stream);
    int   (*stream_destroy)(cudaStream_t stream);
    int   (*get_device_count)(int* count);
    int   (*get_device_props)(struct cudaDeviceProp* props, int device_id);
} cuda_ops_t;

extern cuda_ops_t g_cuda;

#ifdef __cplusplus
}
#endif

/* ==========================================================================
 *  CUDA kernel launch — C++ variadic template (nvcc only)
 *  Must be outside extern "C" because templates require C++ linkage.
 * ========================================================================== */
#ifdef __CUDACC__
#ifdef __cplusplus

template<typename... Args>
static inline void _cuda_kernel_call(const void* kernel, dim3 grid, dim3 block,
                                      size_t shared_mem, cudaStream_t stream,
                                      Args&&... args) {
    void* params[] = { (void*)&args... };
    CUDA_CHECK(cudaLaunchKernel(kernel, grid, block, params, shared_mem, stream));
}

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    _cuda_kernel_call((const void*)(kernel), grid, block, shared_mem, stream, __VA_ARGS__)

#else /* C linkage (unlikely for .cu, but keep a stub) */
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    do { (void)(kernel); (void)(grid); (void)(block); (void)(shared_mem); (void)(stream); } while(0)
#endif
#else /* !__CUDACC__ — host-only C compilation */
#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    do { (void)(kernel); (void)(grid); (void)(block); (void)(shared_mem); (void)(stream); } while(0)
#endif

#endif /* CUDA_OPS_H_ */
