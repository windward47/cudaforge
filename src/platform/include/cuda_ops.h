#ifndef CUDA_OPS_H_
#define CUDA_OPS_H_

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__CUDACC__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Default threads per block for 1D element-wise kernels */
#define OPS_THREADS_PER_BLOCK 256

/* --------------------------------------------------------------------------
 *  CUDA error check — meaningful when CUDA headers are available
 * -------------------------------------------------------------------------- */
#if defined(__CUDACC__) || defined(USE_CUDA)
/* CUDA error check — logs error and returns error code (non-fatal) */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = call; \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(_err)); \
            return (int)_err; \
        } \
    } while(0)
#else
#define CUDA_CHECK(call) (void)(call)
#endif

/* --------------------------------------------------------------------------
 *  CUDA types — when compiled by the host C compiler, use forward decls
 * -------------------------------------------------------------------------- */
#if !defined(__CUDACC__) && !defined(USE_CUDA)
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

/* --------------------------------------------------------------------------
 *  GPU hardware capabilities — runtime snapshot
 *  参考 CUDAForge 的 gpu_specs.py 硬件规格注入思路
 * -------------------------------------------------------------------------- */
typedef struct {
    /* 设备基本信息 */
    int   device_id;
    char  name[256];                /* 设备名称 */
    int   compute_major;            /* 计算能力主版本 */
    int   compute_minor;            /* 计算能力次版本 */

    /* SM 与线程 */
    int   sm_count;                 /* SM 数量 */
    int   max_threads_per_sm;       /* 每 SM 最大线程数 */
    int   max_threads_per_block;    /* 每 block 最大线程数 */
    int   warp_size;                /* warp 大小（通常 32） */

    /* 寄存器与共享内存 */
    int   regs_per_sm;              /* 每 SM 寄存器数 */
    int   regs_per_block;           /* 每 block 最大寄存器数 */
    int   shared_mem_per_sm;        /* 每 SM 共享内存（字节） */
    int   shared_mem_per_block;     /* 每 block 最大共享内存（字节） */

    /* 显存 */
    size_t total_memory;            /* 总显存（字节） */

    /* 理论峰值吞吐 (GFLOPS) */
    float tflops_fp32;              /* FP32 理论峰值 */
    float tflops_fp16;              /* FP16 理论峰值 */
    float tflops_tensor;            /* Tensor Core 理论峰值 */
} gpu_caps_t;

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
    int   (*get_gpu_caps)(gpu_caps_t* caps, int device_id);
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
static inline int _cuda_kernel_call(const void* kernel, dim3 grid, dim3 block,
                                     size_t shared_mem, cudaStream_t stream,
                                     Args&&... args) {
    void* params[] = { (void*)&args... };
    cudaError_t err = cudaLaunchKernel(kernel, grid, block, params, shared_mem, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n",
                __FILE__, __LINE__, cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
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
