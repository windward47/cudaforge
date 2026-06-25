/* cuda2hip.h — CUDA-to-HIP translation layer
 *
 * Two modes:
 * 1. __HIPCC__ (hipcc compiler): full macro remapping of all CUDA symbols
 * 2. USE_CUDA && !__HIPCC__ (host GCC linking against HIP): type/function
 *    remapping for symbols used in .c files (events, graphs, streams)
 */
#ifndef CUDA2HIP_H_
#define CUDA2HIP_H_

#ifdef __HIPCC__
/* ================================================================
 * Mode 1: hipcc — full CUDA-to-HIP macro translation
 * ================================================================ */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ---- Error types ---- */
#define cudaError_t          hipError_t
#define cudaSuccess          hipSuccess
#define cudaGetLastError     hipGetLastError

/* ---- Device & stream types ---- */
#define cudaStream_t         hipStream_t
#define cudaDeviceProp       hipDeviceProp_t

/* ---- Memory copy direction ---- */
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

/* ---- Host allocation flags ---- */
#define cudaHostAllocDefault     hipHostMallocDefault

/* ---- Error handling ---- */
#define cudaGetErrorString       hipGetErrorString

/* ---- Memory management ---- */
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaHostAlloc            hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaMemsetAsync          hipMemsetAsync

/* ---- Device management ---- */
#define cudaSetDevice            hipSetDevice
#define cudaGetDevice            hipGetDevice
#define cudaDeviceReset          hipDeviceReset
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaDeviceGetAttribute   hipDeviceGetAttribute
#define cudaDevAttrClockRate     hipDeviceAttributeClockRate

/* ---- Event management ---- */
#define cudaEvent_t              hipEvent_t
#define cudaEventCreate          hipEventCreate
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDestroy         hipEventDestroy

/* ---- Function attributes ---- */
#define cudaFuncSetAttribute     hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

/* ---- Stream management ---- */
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamSynchronize    hipStreamSynchronize
#define cudaStreamDestroy        hipStreamDestroy

/* ---- Graph management ---- */
#define cudaGraph_t              hipGraph_t
#define cudaGraphExec_t          hipGraphExec_t
#define cudaGraphDestroy         hipGraphDestroy
#define cudaGraphExecDestroy     hipGraphExecDestroy
#define cudaGraphLaunch          hipGraphLaunch
#define cudaStreamBeginCapture   hipStreamBeginCapture
#define cudaStreamEndCapture     hipStreamEndCapture
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal

/* ---- Kernel launch ---- */
#define cudaLaunchKernel         hipModuleLaunchKernel
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel

#elif defined(USE_CUDA) && !defined(__CUDACC__)
/* ================================================================
 * Mode 2: Host GCC compiling .c files that link against HIP.
 * Provides type/function remapping for symbols used in host code.
 * .cu files are compiled by hipcc (Mode 1), .c files by GCC.
 * ================================================================ */
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#include <hip/hip_runtime_api.h>

/* ---- Types ---- */
typedef hipStream_t      cudaStream_t;
typedef hipDeviceProp_t  cudaDeviceProp;
typedef hipEvent_t       cudaEvent_t;
typedef hipGraph_t       cudaGraph_t;
typedef hipGraphExec_t   cudaGraphExec_t;

/* ---- Constants ---- */
#define cudaSuccess          hipSuccess
#define cudaMemcpyHostToDevice   hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaHostAllocDefault     hipHostMallocDefault
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal

/* ---- Functions (via macros) ---- */
#define cudaGetErrorString       hipGetErrorString
#define cudaGetLastError         hipGetLastError
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaHostAlloc            hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemcpy               hipMemcpy
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaMemsetAsync          hipMemsetAsync
#define cudaSetDevice            hipSetDevice
#define cudaGetDevice            hipGetDevice
#define cudaDeviceReset          hipDeviceReset
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaDeviceGetAttribute   hipDeviceGetAttribute
#define cudaDevAttrClockRate     hipDeviceAttributeClockRate
#define cudaEventCreate          hipEventCreate
#define cudaEventRecord          hipEventRecord
#define cudaEventSynchronize     hipEventSynchronize
#define cudaEventElapsedTime     hipEventElapsedTime
#define cudaEventDestroy         hipEventDestroy
#define cudaFuncSetAttribute     hipFuncSetAttribute
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamSynchronize    hipStreamSynchronize
#define cudaStreamDestroy        hipStreamDestroy
#define cudaGraphDestroy         hipGraphDestroy
#define cudaGraphExecDestroy     hipGraphExecDestroy
#define cudaGraphLaunch          hipGraphLaunch
#define cudaStreamBeginCapture   hipStreamBeginCapture
#define cudaStreamEndCapture     hipStreamEndCapture

#endif /* __HIPCC__ / USE_CUDA */
#endif /* CUDA2HIP_H_ */
