#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <stdlib.h>

void* cuda_device_alloc(size_t size) {
    void* ptr = NULL;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void cuda_device_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void* cuda_host_alloc_pinned(size_t size) {
    void* ptr = NULL;
    CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    return ptr;
}

void cuda_host_free_pinned(void* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

int cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}

int cuda_memcpy_d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}
