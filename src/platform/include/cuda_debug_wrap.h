/**
 * cuda_debug_wrap.h — Debug 模式下拦截裸 CUDA API 调用
 *
 * 在 Debug 构建时通过 `-include cuda_debug_wrap.h` 包含此头文件，
 * 将直接 CUDA Runtime API 调用替换为断言失败，强制使用 g_cuda 接口。
 *
 * 用法（CMakeLists.txt）：
 *   if(CMAKE_BUILD_TYPE STREQUAL "Debug")
 *       target_compile_options(operator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-include${CMAKE_CURRENT_SOURCE_DIR}/src/platform/include/cuda_debug_wrap.h>)
 *   endif()
 *
 * 检测脚本：scripts/check_raw_cuda.sh
 */
#ifndef CUDA_DEBUG_WRAP_H_
#define CUDA_DEBUG_WRAP_H_

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>

/* 断言宏：打印文件/行号并终止 */
#define CUDA_RAW_API_BLOCKED(api) \
    do { \
        fprintf(stderr, \
            "ERROR: Direct CUDA API call '%s' detected at %s:%d\n" \
            "  Use g_cuda.* interface instead (see cuda_ops.h)\n", \
            #api, __FILE__, __LINE__); \
        abort(); \
    } while(0)

/*
 * 重定义常用 CUDA Runtime API 为断言失败。
 * 只在算子代码中生效（平台层 cuda_ops.h 先 include，不受影响）。
 *
 * 注意：这些宏会在预处理阶段替换所有匹配的标识符，
 * 因此 cuda_ops.h 中的合法调用也会被替换。
 * 实际使用时应确保 cuda_ops.h 在此头文件之前被 include。
 */

/* 内存管理 */
#define cudaMalloc(...)           CUDA_RAW_API_BLOCKED(cudaMalloc)
#define cudaFree(...)             CUDA_RAW_API_BLOCKED(cudaFree)
#define cudaMemcpy(...)           CUDA_RAW_API_BLOCKED(cudaMemcpy)
#define cudaMemcpyAsync(...)      CUDA_RAW_API_BLOCKED(cudaMemcpyAsync)
#define cudaMemset(...)           CUDA_RAW_API_BLOCKED(cudaMemset)

/* Stream 管理 */
#define cudaStreamCreate(...)     CUDA_RAW_API_BLOCKED(cudaStreamCreate)
#define cudaStreamDestroy(...)    CUDA_RAW_API_BLOCKED(cudaStreamDestroy)
#define cudaStreamSynchronize(...) CUDA_RAW_API_BLOCKED(cudaStreamSynchronize)

/* Event 管理 */
#define cudaEventCreate(...)      CUDA_RAW_API_BLOCKED(cudaEventCreate)
#define cudaEventDestroy(...)     CUDA_RAW_API_BLOCKED(cudaEventDestroy)
#define cudaEventRecord(...)      CUDA_RAW_API_BLOCKED(cudaEventRecord)
#define cudaEventSynchronize(...) CUDA_RAW_API_BLOCKED(cudaEventSynchronize)
#define cudaEventElapsedTime(...) CUDA_RAW_API_BLOCKED(cudaEventElapsedTime)

/* 设备同步 */
#define cudaDeviceSynchronize(...) CUDA_RAW_API_BLOCKED(cudaDeviceSynchronize)

/* 错误处理（应使用 CUDA_CHECK 宏） */
#define cudaGetErrorString(...)   CUDA_RAW_API_BLOCKED(cudaGetErrorString)

#endif /* __CUDACC__ */
#endif /* CUDA_DEBUG_WRAP_H_ */
