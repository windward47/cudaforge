#ifndef PLATFORM_H_
#define PLATFORM_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* ============================================================
 * Data types
 * ============================================================ */
typedef enum {
    DATA_TYPE_F32 = 0,
    DATA_TYPE_F16,
    DATA_TYPE_BF16,
    DATA_TYPE_I32,
    DATA_TYPE_I8,
    DATA_TYPE_U8,
    DATA_TYPE_I64,
    DATA_TYPE_COUNT
} data_type_t;

typedef struct {
    const char* name;
    size_t      size;        /* bytes per element */
    int         is_float;
    int         is_signed;
} data_type_info_t;

const data_type_info_t* data_type_get_info(data_type_t dt);

/* ============================================================
 * Tensor
 * ============================================================ */
#define MAX_TENSOR_DIMS 8

typedef struct {
    data_type_t   dtype;
    int           ndim;
    int64_t       shape[MAX_TENSOR_DIMS];
    int64_t       strides[MAX_TENSOR_DIMS];
    int64_t       numel;
    void*         data;       /* host pointer */
    void*         data_device; /* device pointer (CUDA), NULL if not on GPU */
} tensor_t;

tensor_t* tensor_create(data_type_t dtype, int ndim, const int64_t* shape);
void      tensor_destroy(tensor_t* t);
int       tensor_copy_to_device(tensor_t* t);
int       tensor_copy_to_host(tensor_t* t);
bool       tensor_allclose(const tensor_t* a, const tensor_t* b,
                          float rtol, float atol);

/* ============================================================
 * Stream abstraction
 * ============================================================ */
typedef struct {
    void* cuda_stream;  /* cudaStream_t, NULL = default */
} stream_t;

/* ============================================================
 * Platform ops (CPU)
 * ============================================================ */
typedef struct {
    const char* name;
    int   (*init)(void);
    void  (*finalize)(void);
    void* (*alloc_aligned)(size_t size, size_t align);
    void  (*free_aligned)(void* ptr);
    int   (*get_cache_line_size)(void);
    int   (*get_simd_flags)(void);
    int   (*get_core_count)(void);
    int   (*pin_thread)(int cpu_id);
} platform_ops_t;

extern const platform_ops_t* g_platform;

/* ============================================================
 * Platform detection macros
 * ============================================================ */
#if defined(__x86_64__) || defined(__i386__) || defined(_M_AMD64) || defined(_M_IX86)
  #define PLATFORM_X86 1
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
  #define PLATFORM_ARM 1
#elif defined(__riscv)
  #define PLATFORM_RISCV 1
#endif

/* SIMD flags */
#define SIMD_AVX2   (1 << 0)
#define SIMD_AVX512 (1 << 1)
#define SIMD_NEON   (1 << 2)
#define SIMD_SVE    (1 << 3)

#ifdef __cplusplus
#define C_EXTERN extern "C"
#else
#define C_EXTERN
#endif

C_EXTERN int      platform_get_cache_line_size(void);
C_EXTERN int      platform_get_core_count(void);
C_EXTERN void*    platform_alloc_aligned(size_t size, size_t alignment);
C_EXTERN void     platform_free_aligned(void* ptr);

/* Top-level init / finalize */
C_EXTERN int      platform_init(void);
C_EXTERN void     platform_finalize(void);

#endif /* PLATFORM_H_ */
