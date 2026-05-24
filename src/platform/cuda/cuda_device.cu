#include "cuda_ops.h"
#include <stdio.h>
#include <stdlib.h>

static int cuda_init(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    return 0;
}

static void cuda_finalize(void) {
    CUDA_CHECK(cudaDeviceReset());
}

static int cuda_get_device_count(int* count) {
    return cudaGetDeviceCount(count);
}

static int cuda_get_device_props(cudaDeviceProp* props, int device_id) {
    return cudaGetDeviceProperties(props, device_id);
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
};
