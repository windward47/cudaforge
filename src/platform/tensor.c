#include "platform.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_ops.h"
#endif

tensor_t* tensor_create(data_type_t dtype, int ndim, const int64_t* shape) {
    if (ndim < 0 || ndim > MAX_TENSOR_DIMS) return NULL;
    if (!shape) return NULL;

    tensor_t* t = (tensor_t*)calloc(1, sizeof(tensor_t));
    if (!t) return NULL;

    t->dtype = dtype;
    t->ndim  = ndim;
    t->numel = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->numel   *= shape[i];
    }

    const data_type_info_t* info = data_type_get_info(dtype);
    if (!info) { free(t); return NULL; }
    size_t bytes = (size_t)t->numel * info->size;

    t->data = platform_alloc_aligned(bytes, 64);
    if (!t->data && bytes > 0) {
        free(t);
        return NULL;
    }

    memset(t->data, 0, bytes);
    return t;
}

void tensor_destroy(tensor_t* t) {
    if (!t) return;
    if (t->data) platform_free_aligned(t->data);
#ifdef USE_CUDA
    if (t->data_device && g_cuda.device_free) {
        g_cuda.device_free(t->data_device);
    }
#endif
    free(t);
}

int tensor_copy_to_device(tensor_t* t) {
    if (!t || !t->data) return -1;
#ifdef USE_CUDA
    if (!g_cuda.device_alloc || !g_cuda.memcpy_h2d) return -1;
    if (t->data_device) return 0;  /* already on device */

    const data_type_info_t* info = data_type_get_info(t->dtype);
    if (!info) return -1;
    size_t bytes = (size_t)t->numel * info->size;

    t->data_device = g_cuda.device_alloc(bytes);
    if (!t->data_device) return -1;

    return g_cuda.memcpy_h2d(t->data_device, t->data, bytes, 0);
#else
    (void)t;
    return 0;
#endif
}

int tensor_copy_to_host(tensor_t* t) {
    if (!t) return -1;
#ifdef USE_CUDA
    if (!t->data_device) return 0;  /* nothing to copy */
    if (!g_cuda.memcpy_d2h) return -1;

    const data_type_info_t* info = data_type_get_info(t->dtype);
    if (!info) return -1;
    size_t bytes = (size_t)t->numel * info->size;

    /* Copy back via async D2H on stream 0. Device memory is retained for
       potential reuse by downstream nodes — it is freed in tensor_destroy(). */
    return g_cuda.memcpy_d2h(t->data, t->data_device, bytes, 0);
#else
    (void)t;
    return 0;
#endif
}

/* Software FP16 decode (IEEE 754 half-precision) for tensor_allclose */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)(h & 0x8000)) << 16;
    int32_t exp = ((h >> 10) & 0x1F);
    uint32_t mant = ((uint32_t)(h & 0x3FF)) << 13;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else { while (!(mant & 0x800000)) { mant <<= 1; exp--; }
               exp++; mant &= ~0x800000;
               bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | mant; }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | mant;
    } else {
        bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | mant;
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}

static float tensor_get_float(const tensor_t* t, int64_t i) {
    const data_type_info_t* info = data_type_get_info(t->dtype);
    if (info->size == 4) return ((float*)t->data)[i];
    if (info->size == 2) return f16_to_f32(((uint16_t*)t->data)[i]);
    return 0.0f;
}

bool tensor_allclose(const tensor_t* a, const tensor_t* b,
                     float rtol, float atol) {
    if (!a || !b || a->numel != b->numel) return false;
    if (a->dtype != b->dtype) return false;

    for (int64_t i = 0; i < a->numel; i++) {
        float va = tensor_get_float(a, i);
        float vb = tensor_get_float(b, i);
        float diff = fabsf(va - vb);
        float max  = fmaxf(fabsf(va), fabsf(vb));
        if (diff > atol && diff > rtol * max) {
            return false;
        }
    }
    return true;
}
