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
    if (!g_cuda.memcpy_d2h || !g_cuda.device_free) return -1;

    const data_type_info_t* info = data_type_get_info(t->dtype);
    if (!info) return -1;
    size_t bytes = (size_t)t->numel * info->size;

    /* memcpy_d2h uses cudaMemcpyAsync on stream 0. The device pointer is freed
       immediately after the async copy is queued. This is safe because:
       1. The default stream guarantees in-order execution — the copy is queued
          before cudaFree, so the memory is still valid when the copy runs.
       2. The caller (graph_execute) must call stream_synchronize(0) after all
          node outputs are copied back, before reading host data. */
    int ret = g_cuda.memcpy_d2h(t->data, t->data_device, bytes, 0);
    g_cuda.device_free(t->data_device);
    t->data_device = NULL;
    return ret;
#else
    (void)t;
    return 0;
#endif
}

bool tensor_allclose(const tensor_t* a, const tensor_t* b,
                     float rtol, float atol) {
    if (!a || !b || a->numel != b->numel) return false;

    for (int64_t i = 0; i < a->numel; i++) {
        float va = ((float*)a->data)[i];
        float vb = ((float*)b->data)[i];
        float diff = fabsf(va - vb);
        float max  = fmaxf(fabsf(va), fabsf(vb));
        if (diff > atol && diff > rtol * max) {
            return false;
        }
    }
    return true;
}
