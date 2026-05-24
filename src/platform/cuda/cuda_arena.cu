#include "cuda_arena.h"
#include "cuda_ops.h"
#include <cuda_runtime.h>

#define ARENA_ALIGN 256

static void*  s_pool     = NULL;
static size_t s_capacity = 0;
static size_t s_offset   = 0;

static size_t align_up(size_t n, size_t align) {
    return (n + align - 1) & ~(align - 1);
}

int cuda_arena_init(size_t total_bytes) {
    if (s_pool) cuda_arena_destroy();
    total_bytes = align_up(total_bytes, ARENA_ALIGN);
    cudaError_t err = cudaMalloc(&s_pool, total_bytes);
    if (err != cudaSuccess) {
        s_pool     = NULL;
        s_capacity = 0;
        s_offset   = 0;
        return -1;
    }
    s_capacity = total_bytes;
    s_offset   = 0;
    return 0;
}

void* cuda_arena_alloc(size_t bytes) {
    if (!s_pool) return NULL;
    size_t aligned = align_up(bytes, ARENA_ALIGN);
    if (s_offset + aligned > s_capacity) return NULL;
    void* ptr = (char*)s_pool + s_offset;
    s_offset += aligned;
    return ptr;
}

void cuda_arena_reset(void) {
    s_offset = 0;
}

void cuda_arena_destroy(void) {
    if (s_pool) {
        cudaFree(s_pool);
        s_pool     = NULL;
        s_capacity = 0;
        s_offset   = 0;
    }
}

size_t cuda_arena_used(void) {
    return s_offset;
}

size_t cuda_arena_capacity(void) {
    return s_capacity;
}
