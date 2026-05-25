#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <stdlib.h>

void* cuda_device_alloc(size_t size) {
    void* ptr = NULL;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_device_alloc(%zu): %s\n", size, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void cuda_device_free(void* ptr) {
    if (ptr) {
        cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess)
            fprintf(stderr, "cuda_device_free: %s\n", cudaGetErrorString(err));
    }
}

void* cuda_host_alloc_pinned(size_t size) {
    void* ptr = NULL;
    cudaError_t err = cudaHostAlloc(&ptr, size, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_host_alloc_pinned(%zu): %s\n", size, cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

void cuda_host_free_pinned(void* ptr) {
    if (ptr) {
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess)
            fprintf(stderr, "cuda_host_free_pinned: %s\n", cudaGetErrorString(err));
    }
}

int cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_h2d: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_d2h: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}

int cuda_memcpy_d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_memcpy_d2d: %s\n", cudaGetErrorString(err));
        return (int)err;
    }
    return 0;
}
