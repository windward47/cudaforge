/**
 * FP16 operator unit tests.
 *
 * Verifies FP16 CUDA kernels produce reasonable outputs.
 * Uses uint16_t for FP16 data (software encode/decode).
 */
#include "platform.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

/* Software FP16 encode/decode */
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) return (uint16_t)(sign);
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

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

#ifdef USE_CUDA
/* ============================================================
 * Test: Softmax FP16
 * ============================================================ */
static int test_softmax_fp16(void) {
    fprintf(stderr, "\n=== Softmax FP16 Test ===\n");
    int64_t num_blocks = 4, block_size = 8;

    tensor_t* tIn = tensor_create(DATA_TYPE_F16, 2, (int64_t[]){num_blocks, block_size});
    tensor_t* tOut = tensor_create(DATA_TYPE_F16, 2, (int64_t[]){num_blocks, block_size});

    srand(42);
    uint16_t* in_h = (uint16_t*)tIn->data;
    for (int64_t i = 0; i < num_blocks * block_size; i++) {
        in_h[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 4.0f);
    }

    tensor_copy_to_device(tIn);
    tensor_copy_to_device(tOut);

    const operator_registry_t* softmax_f16 = operator_find("softmax_f16_cuda");
    if (!softmax_f16 || !softmax_f16->func) {
        fprintf(stderr, "  SKIP: softmax_f16_cuda not registered\n");
        tensor_destroy(tIn); tensor_destroy(tOut);
        return 0;
    }

    typedef struct { int64_t num_classes; int64_t num_blocks; } test_softmax_params_t;
    test_softmax_params_t sp = { block_size, num_blocks };

    const void* inputs[] = { tIn->data_device };
    void* outputs[] = { tOut->data_device };
    softmax_f16->func(inputs, outputs, (const operator_params_t*)&sp, NULL);

    tensor_copy_to_host(tOut);
    g_cuda.stream_synchronize(0);

    /* Verify: each block should sum to ~1.0 */
    uint16_t* out_h = (uint16_t*)tOut->data;
    int ok = 1;
    for (int64_t b = 0; b < num_blocks; b++) {
        float sum = 0.0f;
        for (int64_t i = 0; i < block_size; i++) {
            sum += f16_to_f32(out_h[b * block_size + i]);
        }
        if (fabsf(sum - 1.0f) > 0.02f) {
            fprintf(stderr, "  MISMATCH block %lld: sum=%.4f (expected 1.0)\n", (long long)b, sum);
            ok = 0;
        }
    }
    fprintf(stderr, "  Softmax FP16: %s\n", ok ? "PASS" : "FAIL");

    tensor_destroy(tIn); tensor_destroy(tOut);
    return ok ? 0 : 1;
}

/* ============================================================
 * Test: Conv2D FP16 (smoke test — verify non-zero output)
 * ============================================================ */
static int test_conv2d_fp16(void) {
    fprintf(stderr, "\n=== Conv2D FP16 Smoke Test ===\n");
    int64_t N = 1, C = 1, H = 4, W = 4, K = 1, KH = 3, KW = 3;
    int64_t OH = H, OW = W;

    tensor_t* tIn = tensor_create(DATA_TYPE_F16, 4, (int64_t[]){N, C, H, W});
    tensor_t* tW = tensor_create(DATA_TYPE_F16, 4, (int64_t[]){K, C, KH, KW});
    tensor_t* tB = tensor_create(DATA_TYPE_F16, 1, (int64_t[]){K});
    tensor_t* tOut = tensor_create(DATA_TYPE_F16, 4, (int64_t[]){N, K, OH, OW});

    srand(42);
    uint16_t* in_h = (uint16_t*)tIn->data;
    uint16_t* w_h = (uint16_t*)tW->data;
    uint16_t* b_h = (uint16_t*)tB->data;
    for (int64_t i = 0; i < N*C*H*W; i++) in_h[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    for (int64_t i = 0; i < K*C*KH*KW; i++) w_h[i] = f32_to_f16(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
    b_h[0] = f32_to_f16(0.0f);

    tensor_copy_to_device(tIn);
    tensor_copy_to_device(tW);
    tensor_copy_to_device(tB);
    tensor_copy_to_device(tOut);

    const operator_registry_t* conv_f16 = operator_find("conv2d_f16_cuda");
    if (!conv_f16 || !conv_f16->func) {
        fprintf(stderr, "  SKIP: conv2d_f16_cuda not registered\n");
        tensor_destroy(tIn); tensor_destroy(tW); tensor_destroy(tB); tensor_destroy(tOut);
        return 0;
    }

    typedef struct {
        int64_t N, C, H, W, K, kernel_h, kernel_w;
        int64_t pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
        int64_t groups, fuse_activation;
    } test_conv_params_t;
    test_conv_params_t cp = { N, C, H, W, K, KH, KW, 1, 1, 1, 1, 1, 1, 1, 0 };

    const void* inputs[] = { tIn->data_device, tW->data_device, tB->data_device };
    void* outputs[] = { tOut->data_device };
    int ret = conv_f16->func(inputs, outputs, (const operator_params_t*)&cp, NULL);

    if (ret != 0) {
        fprintf(stderr, "  Conv2D FP16: FAIL (returned %d)\n", ret);
        tensor_destroy(tIn); tensor_destroy(tW); tensor_destroy(tB); tensor_destroy(tOut);
        return 1;
    }

    tensor_copy_to_host(tOut);
    g_cuda.stream_synchronize(0);

    uint16_t* out_h = (uint16_t*)tOut->data;
    int has_nonzero = 0;
    for (int64_t i = 0; i < N * K * OH * OW; i++) {
        if (f16_to_f32(out_h[i]) != 0.0f) { has_nonzero = 1; break; }
    }
    fprintf(stderr, "  Conv2D FP16: %s\n", has_nonzero ? "PASS" : "FAIL");

    tensor_destroy(tIn); tensor_destroy(tW); tensor_destroy(tB); tensor_destroy(tOut);
    return has_nonzero ? 0 : 1;
}
#endif /* USE_CUDA */

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);

    int failures = 0;
    failures += test_softmax_fp16();
    failures += test_conv2d_fp16();

    fprintf(stderr, "\n=== FP16 Ops Tests: %s ===\n", failures == 0 ? "ALL PASS" : "SOME FAILED");

    cuda_platform_finalize();
#else
    fprintf(stderr, "SKIP: FP16 tests require CUDA\n");
#endif
    platform_finalize();
    return 0;
}
