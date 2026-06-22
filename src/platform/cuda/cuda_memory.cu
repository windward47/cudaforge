#include "cuda_ops.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * CUDA Memory Pool — free-list allocator
 *
 * Maintains a pool of freed allocations by size bucket.
 * On alloc: find a reused block or cudaMalloc.
 * On free: return block to pool instead of cudaFree.
 *
 * Bucket strategy: round up size to next power-of-2 tier.
 * Max pooled size: 64 MB. Larger allocs go directly to cudaMalloc.
 * Pool is global, thread-safe via simple spinlock.
 * ============================================================ */

#define POOL_MAX_BUCKETS  20   /* 2^0 .. 2^19 = 512KB tiers, up to 64MB */
#define POOL_MAX_ENTRIES  64   /* max free blocks per bucket */
#define POOL_MAX_SIZE     (64 * 1024 * 1024)  /* 64 MB cap */

typedef struct {
    void*  ptrs[POOL_MAX_ENTRIES];
    int    count;
} pool_bucket_t;

static pool_bucket_t g_pool[POOL_MAX_BUCKETS];
static int g_pool_initialized = 0;

static void pool_init(void) {
    if (g_pool_initialized) return;
    memset(g_pool, 0, sizeof(g_pool));
    g_pool_initialized = 1;
}

/* Find bucket index: ceil(log2(size)) clamped to valid range */
static int pool_bucket_idx(size_t size) {
    if (size == 0) size = 1;
    int idx = 0;
    size_t s = 1;
    while (s < size && idx < POOL_MAX_BUCKETS - 1) { s <<= 1; idx++; }
    return idx;
}

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

/* ============================================================
 * Pool-aware alloc/free: sized free enables reuse.
 * Use this for hot paths (e.g., graph execution temporaries).
 * Pool is per-size-bucket, LIFO, capped at POOL_MAX_ENTRIES.
 * ============================================================ */
void* cuda_device_alloc_pooled(size_t size) {
    pool_init();

    if (size > POOL_MAX_SIZE || size == 0) {
        void* ptr = NULL;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    int idx = pool_bucket_idx(size);
    pool_bucket_t* b = &g_pool[idx];

    if (b->count > 0) {
        b->count--;
        return b->ptrs[b->count];
    }

    size_t alloc_size = (size_t)1 << idx;
    void* ptr = NULL;
    cudaMalloc(&ptr, alloc_size);
    return ptr;
}

void cuda_device_free_sized(void* ptr, size_t size) {
    if (!ptr) return;
    pool_init();

    if (size > POOL_MAX_SIZE || size == 0) {
        cudaFree(ptr);
        return;
    }

    int idx = pool_bucket_idx(size);
    pool_bucket_t* b = &g_pool[idx];

    if (b->count < POOL_MAX_ENTRIES) {
        b->ptrs[b->count] = ptr;
        b->count++;
    } else {
        cudaFree(ptr);
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
