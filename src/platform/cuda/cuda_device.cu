#include "cuda_ops.h"
#include <stdio.h>
#include <stdlib.h>

static int cuda_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d): %s\n", device_id, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

static void cuda_finalize(void) {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess)
        fprintf(stderr, "cudaDeviceReset: %s\n", cudaGetErrorString(err));
}

static int cuda_get_device_count(int* count) {
    return cudaGetDeviceCount(count);
}

static int cuda_get_device_props(cudaDeviceProp* props, int device_id) {
    return cudaGetDeviceProperties(props, device_id);
}

static int cuda_get_gpu_caps(gpu_caps_t* caps, int device_id) {
    if (!caps) return -1;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return (int)err;

    /* 基本信息 */
    caps->device_id = device_id;
    snprintf(caps->name, sizeof(caps->name), "%s", prop.name);
    caps->compute_major = prop.major;
    caps->compute_minor = prop.minor;

    /* SM 与线程 */
    caps->sm_count            = prop.multiProcessorCount;
    caps->max_threads_per_sm  = prop.maxThreadsPerMultiProcessor;
    caps->max_threads_per_block = prop.maxThreadsPerBlock;
    caps->warp_size           = prop.warpSize;

    /* 寄存器与共享内存 */
    caps->regs_per_sm         = prop.regsPerMultiprocessor;
    caps->regs_per_block      = prop.regsPerBlock;
    caps->shared_mem_per_sm   = (int)prop.sharedMemPerMultiprocessor;
    caps->shared_mem_per_block = (int)prop.sharedMemPerBlock;

    /* 显存 */
    caps->total_memory = prop.totalGlobalMem;

    /*
     * 理论峰值吞吐估算 (GFLOPS)
     * 公式: SM数 × 每SM FP32核心数 × 2 × 时钟频率(GHz)
     *
     * 每 SM FP32 核心数（CUDA cores per SM）:
     *   sm_50 (Maxwell):  128
     *   sm_60 (Pascal):   64
     *   sm_70 (Volta):    64
     *   sm_75 (Turing):   64
     *   sm_80 (Ampere):   64
     *   sm_86 (Ampere):   128
     *   sm_89 (Ada):      128
     *   sm_90 (Hopper):   128
     */
    int cores_per_sm = 64; /* default */
    if (prop.major == 5) cores_per_sm = 128;          /* Maxwell */
    else if (prop.major == 6 && prop.minor == 0) cores_per_sm = 64;   /* Pascal */
    else if (prop.major == 7 && prop.minor == 0) cores_per_sm = 64;   /* Volta */
    else if (prop.major == 7 && prop.minor == 5) cores_per_sm = 64;   /* Turing */
    else if (prop.major == 8 && prop.minor == 0) cores_per_sm = 64;   /* Ampere A100 */
    else if (prop.major == 8 && prop.minor == 6) cores_per_sm = 128;  /* Ampere RTX 30/40 */
    else if (prop.major == 8 && prop.minor == 9) cores_per_sm = 128;  /* Ada Lovelace */
    else if (prop.major == 9) cores_per_sm = 128;                     /* Hopper */

    /* 用 cudaDeviceGetAttribute 查询时钟频率（兼容 CUDA 13.2+） */
    int clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id);
    float clock_ghz = clock_khz * 1e-6f;  /* kHz → GHz */

    float fp32_gflops = (float)prop.multiProcessorCount * cores_per_sm * 2.0f * clock_ghz;
    caps->tflops_fp32 = fp32_gflops / 1000.0f;  /* GFLOPS → TFLOPS */

    /* FP16 = 2× FP32（Pascal+） */
    caps->tflops_fp16 = fp32_gflops * 2.0f / 1000.0f;

    /* Tensor Core: Volta=8× FP32, Ampere+=8× FP32 per SM */
    if (prop.major >= 7) {
        caps->tflops_tensor = fp32_gflops * 8.0f / 1000.0f;
    } else {
        caps->tflops_tensor = 0.0f;
    }

    return 0;
}

static int cuda_stream_create(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

static int cuda_stream_synchronize(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

static int cuda_stream_destroy(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

/* Memory functions are wired in by cuda_platform_init() */
cuda_ops_t g_cuda = {
    .init                = cuda_init,
    .finalize            = cuda_finalize,
    .device_alloc        = nullptr,
    .device_free         = nullptr,
    .host_alloc_pinned   = nullptr,
    .host_free_pinned    = nullptr,
    .memcpy_h2d          = nullptr,
    .memcpy_d2h          = nullptr,
    .memcpy_d2d          = nullptr,
    .stream_create       = cuda_stream_create,
    .stream_synchronize  = cuda_stream_synchronize,
    .stream_destroy      = cuda_stream_destroy,
    .get_device_count    = cuda_get_device_count,
    .get_device_props    = cuda_get_device_props,
    .get_gpu_caps        = cuda_get_gpu_caps,
};
