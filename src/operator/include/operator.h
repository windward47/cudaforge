#ifndef OPERATOR_H_
#define OPERATOR_H_

#include "platform.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * Operator parameter (opaque base — each op defines its own)
 * Each operator casts this to its own params struct (e.g. conv_params_t).
 * ============================================================ */
typedef struct {
    char _reserved;  /* non-empty for C11 compliance; unused */
} operator_params_t;

/* ============================================================
 * Operator function signature
 * ============================================================ */
typedef int (*operator_func_t)(const void* inputs[],
                                void* outputs[],
                                const operator_params_t* params,
                                stream_t* stream);

/* ============================================================
 * Operator registry entry
 * ============================================================ */
#define OP_FLAG_NONE      (0)
#define OP_FLAG_IN_PLACE  (1 << 0)
#define OP_FLAG_ALLOW_ALIAS (1 << 1)

typedef struct {
    const char*       name;       /* e.g. "relu_f32" */
    const char*       data_type;  /* "f32", "f16", "i8" */
    operator_func_t   func;       /* function pointer */
    int               version;
    uint32_t          flags;
} operator_registry_t;

/* ============================================================
 * Registry access
 * ============================================================ */
const operator_registry_t* operator_find(const char* name);
int operator_register(const operator_registry_t* op);

#ifdef __cplusplus
}
#endif

#endif /* OPERATOR_H_ */
