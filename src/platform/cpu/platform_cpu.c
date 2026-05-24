#include "platform.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * x86 platform ops implementation
 * ============================================================ */
#if defined(_WIN32) || defined(_WIN64)
  #include <windows.h>
  #define ALIGNED_ALLOC(size, align)  _aligned_malloc(size, align)
  #define ALIGNED_FREE(ptr)           _aligned_free(ptr)
#else
  #include <unistd.h>
  #define ALIGNED_ALLOC(size, align)  aligned_alloc(align, size)
  #define ALIGNED_FREE(ptr)           free(ptr)
#endif

static int s_cache_line_size = 64;
static int s_core_count      = 1;

static int x86_init(void) {
#if defined(_WIN32) || defined(_WIN64)
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    s_core_count = sysinfo.dwNumberOfProcessors;
    /* Windows: no portable way to get cache line size, default 64 */
#else
    s_core_count = (int)sysconf(_SC_NPROCESSORS_ONLN);
    long sz = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (sz > 0) s_cache_line_size = (int)sz;
#endif
    return 0;
}

static void x86_finalize(void) {}

static void* x86_alloc_aligned(size_t size, size_t align) {
    return ALIGNED_ALLOC(size, align);
}

static void x86_free_aligned(void* ptr) {
    ALIGNED_FREE(ptr);
}

static int x86_get_cache_line_size(void) {
    return s_cache_line_size;
}

static int x86_get_simd_flags(void) {
    int flags = 0;
#if defined(__AVX2__)
    flags |= SIMD_AVX2;
#endif
#if defined(__AVX512F__)
    flags |= SIMD_AVX512;
#endif
#if defined(__ARM_NEON)
    flags |= SIMD_NEON;
#endif
    return flags;
}

static int x86_get_core_count(void) {
    return s_core_count;
}

static int x86_pin_thread(int cpu_id) {
    (void)cpu_id;
    /* Not yet implemented — return 0 to indicate success */
    return 0;
}

static const platform_ops_t s_platform_x86 = {
    .name              = "x86",
    .init              = x86_init,
    .finalize          = x86_finalize,
    .alloc_aligned     = x86_alloc_aligned,
    .free_aligned      = x86_free_aligned,
    .get_cache_line_size = x86_get_cache_line_size,
    .get_simd_flags    = x86_get_simd_flags,
    .get_core_count    = x86_get_core_count,
    .pin_thread        = x86_pin_thread,
};

/* ============================================================
 * ARM platform ops (simple)
 * ============================================================ */
#if defined(PLATFORM_ARM)
static const platform_ops_t s_platform_arm = {
    .name              = "ARM",
    .init              = x86_init,
    .finalize          = x86_finalize,
    .alloc_aligned     = x86_alloc_aligned,
    .free_aligned      = x86_free_aligned,
    .get_cache_line_size = x86_get_cache_line_size,
    .get_simd_flags    = x86_get_simd_flags,
    .get_core_count    = x86_get_core_count,
    .pin_thread        = x86_pin_thread,
};
#endif

/* ============================================================
 * g_platform — set by platform_init()
 * ============================================================ */
const platform_ops_t* g_platform = NULL;

int platform_init(void) {
#if defined(PLATFORM_X86)
    g_platform = &s_platform_x86;
#elif defined(PLATFORM_ARM)
    g_platform = &s_platform_arm;
#else
    /* No known platform — malloc fallback via static ops */
    static const platform_ops_t s_fallback = {
        .name              = "fallback",
        .init              = x86_init,
        .finalize          = x86_finalize,
        .alloc_aligned     = x86_alloc_aligned,
        .free_aligned      = x86_free_aligned,
        .get_cache_line_size = x86_get_cache_line_size,
        .get_simd_flags    = x86_get_simd_flags,
        .get_core_count    = x86_get_core_count,
        .pin_thread        = x86_pin_thread,
    };
    g_platform = &s_fallback;
#endif
    return g_platform->init();
}

void platform_finalize(void) {
    if (g_platform) g_platform->finalize();
    g_platform = NULL;
}

/* ============================================================
 * Public API wrappers (forward to g_platform)
 * ============================================================ */
int platform_get_cache_line_size(void) {
    return g_platform ? g_platform->get_cache_line_size() : 64;
}

int platform_get_core_count(void) {
    return g_platform ? g_platform->get_core_count() : 1;
}

void* platform_alloc_aligned(size_t size, size_t alignment) {
    return g_platform ? g_platform->alloc_aligned(size, alignment) : NULL;
}

void platform_free_aligned(void* ptr) {
    if (g_platform) g_platform->free_aligned(ptr);
}
