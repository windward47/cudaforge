#include "cuda_platform.h"
#include "cuda_ops.h"
#include <stdio.h>

/* Functions defined in cuda_memory.cu */
extern void* cuda_device_alloc(size_t size);
extern void  cuda_device_free(void* ptr);
extern void  cuda_device_free_sized(void* ptr, size_t size);
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

int cuda_get_gpu_caps(gpu_caps_t* caps, int device_id) {
    if (!g_cuda.get_gpu_caps) return -1;
    return g_cuda.get_gpu_caps(caps, device_id);
}

void cuda_print_gpu_caps(int device_id) {
    gpu_caps_t caps;
    if (cuda_get_gpu_caps(&caps, device_id) != 0) {
        fprintf(stderr, "Failed to get GPU capabilities\n");
        return;
    }

    fprintf(stderr, "=== GPU Capabilities ===\n");
    fprintf(stderr, "  Device:       %s\n", caps.name);
    fprintf(stderr, "  Compute:      %d.%d\n", caps.compute_major, caps.compute_minor);
    fprintf(stderr, "  SM count:     %d\n", caps.sm_count);
    fprintf(stderr, "  Threads/SM:   %d\n", caps.max_threads_per_sm);
    fprintf(stderr, "  Threads/block:%d\n", caps.max_threads_per_block);
    fprintf(stderr, "  Warp size:    %d\n", caps.warp_size);
    fprintf(stderr, "  Regs/SM:      %d\n", caps.regs_per_sm);
    fprintf(stderr, "  Regs/block:   %d\n", caps.regs_per_block);
    fprintf(stderr, "  SharedMem/SM: %d bytes\n", caps.shared_mem_per_sm);
    fprintf(stderr, "  SharedMem/blk:%d bytes\n", caps.shared_mem_per_block);
    fprintf(stderr, "  Total VRAM:   %.1f GB\n", caps.total_memory / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "  FP32 peak:    %.2f TFLOPS\n", caps.tflops_fp32);
    fprintf(stderr, "  FP16 peak:    %.2f TFLOPS\n", caps.tflops_fp16);
    fprintf(stderr, "  Tensor peak:  %.2f TFLOPS\n", caps.tflops_tensor);
    fprintf(stderr, "========================\n");
}
