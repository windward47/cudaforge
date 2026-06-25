/* ============================================================
 * operator_init.c — 算子注册初始化
 *
 * 使用 operator_registry.def 中的 X-macro 自动展开，
 * 新增算子只需在 .def 文件中添加条目，无需修改本文件。
 * ============================================================ */
#include "operator.h"

/* ---- 前向声明（CPU 算子）---- */
#define REGISTER_CPU(fn)  int fn(void);
#define REGISTER_CUDA(fn) /* no-op for forward decl of CUDA */
#define REGISTER_HIP(fn)  /* no-op for forward decl of HIP */
#include "operator_registry.def"
#undef REGISTER_CPU
#undef REGISTER_CUDA
#undef REGISTER_HIP

#ifdef USE_CUDA
/* ---- 前向声明（CUDA / HIP 算子）---- */
#define REGISTER_CPU(fn)  /* no-op */
#define REGISTER_CUDA(fn) int fn(void);
#define REGISTER_HIP(fn)  /* no-op */
#include "operator_registry.def"
#undef REGISTER_CPU
#undef REGISTER_CUDA
#undef REGISTER_HIP
#endif

#ifdef USE_HIP
/* ---- 前向声明（HIP 专属算子）---- */
#define REGISTER_CPU(fn)  /* no-op */
#define REGISTER_CUDA(fn) /* no-op */
#define REGISTER_HIP(fn)  int fn(void);
#include "operator_registry.def"
#undef REGISTER_CPU
#undef REGISTER_CUDA
#undef REGISTER_HIP
#endif

int operator_init_all(void) {
    int ret = 0;

    /* CPU 算子注册 */
    #define REGISTER_CPU(fn)  ret += fn();
    #define REGISTER_CUDA(fn) /* no-op */
    #define REGISTER_HIP(fn)  /* no-op */
    #include "operator_registry.def"
    #undef REGISTER_CPU
    #undef REGISTER_CUDA
    #undef REGISTER_HIP

#ifdef USE_CUDA
    /* CUDA 算子注册 */
    #define REGISTER_CPU(fn)  /* no-op */
    #define REGISTER_CUDA(fn) ret += fn();
    #define REGISTER_HIP(fn)  /* no-op */
    #include "operator_registry.def"
    #undef REGISTER_CPU
    #undef REGISTER_CUDA
    #undef REGISTER_HIP
#endif

#ifdef USE_HIP
    /* HIP 专属算子注册 */
    #define REGISTER_CPU(fn)  /* no-op */
    #define REGISTER_CUDA(fn) /* no-op */
    #define REGISTER_HIP(fn)  ret += fn();
    #include "operator_registry.def"
    #undef REGISTER_CPU
    #undef REGISTER_CUDA
    #undef REGISTER_HIP
#endif

    return ret;
}
