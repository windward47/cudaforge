#include "cuda_platform.h"
#include "cuda_ops.h"

/* Functions defined in cuda_memory.cu */
extern void* cuda_device_alloc(size_t size);
extern void  cuda_device_free(void* ptr);
extern void* cuda_host_alloc_pinned(size_t size);
extern void  cuda_host_free_pinned(void* ptr);
extern int   cuda_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream);
extern int   cuda_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream);
extern int   cuda_memcpy_d2d(void* dst, const void* src, size_t bytes, cudaStream_t stream);

int cuda_platform_init(int device_id) {
    if (g_cuda.init(device_id) != 0) return -1;

    g_cuda.device_alloc      = cuda_device_alloc;
    g_cuda.device_free       = cuda_device_free;
    g_cuda.host_alloc_pinned = cuda_host_alloc_pinned;
    g_cuda.host_free_pinned  = cuda_host_free_pinned;
    g_cuda.memcpy_h2d        = cuda_memcpy_h2d;
    g_cuda.memcpy_d2h        = cuda_memcpy_d2h;
    g_cuda.memcpy_d2d        = cuda_memcpy_d2d;

    return 0;
}

void cuda_platform_finalize(void) {
    if (g_cuda.finalize) g_cuda.finalize();
}
