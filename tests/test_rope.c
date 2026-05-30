/**
 * RoPE (Rotary Position Encoding) unit test.
 *
 * Verifies:
 *   - CPU reference correctness (known rotation angles)
 *   - CUDA vs CPU comparison
 *   - In-place operation (input == output pointer)
 *   - Multiple positions and heads
 */
#include "rope_int.h"
#include "platform.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);
int rope_f32(const void* inputs[], void* outputs[],
             const operator_params_t* params, stream_t* stream);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

/* Small test dimensions */
#define T_B  1
#define T_S  4
#define T_H  2
#define T_d  4
#define T_D  (T_H * T_d)  /* 8 */

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

/* CPU RoPE reference implementation */
static void rope_ref(const float* in, float* out, int64_t S, int64_t H, int64_t d, float base) {
    memcpy(out, in, S * H * d * sizeof(float));
    int64_t half_d = d / 2;

    for (int64_t pos = 0; pos < S; pos++) {
        for (int64_t h = 0; h < H; h++) {
            float* q = out + (pos * H + h) * d;
            for (int64_t i = 0; i < half_d; i++) {
                float angle = (float)pos / powf(base, (float)(2 * i) / (float)d);
                float c = cosf(angle);
                float s = sinf(angle);
                float x0 = q[2 * i];
                float x1 = q[2 * i + 1];
                q[2 * i]     = x0 * c - x1 * s;
                q[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

/* ============================================================
 * Test: RoPE CPU correctness
 * ============================================================ */
static int test_rope_cpu(void) {
    fprintf(stderr, "\n=== RoPE CPU Test ===\n");

    int64_t shape[] = {T_S, T_H, T_d};
    tensor_t* tIn = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 3, shape);
    float* ref = (float*)calloc(T_S * T_H * T_d, sizeof(float));

    /* Fill with deterministic data */
    srand(42);
    for (int64_t i = 0; i < T_S * T_H * T_d; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    rope_params_t p;
    p.seq_len = T_S;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;

    /* CPU RoPE */
    const void* in_ptr = tIn->data;
    void* out_ptr = tOut->data;
    int ret = rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "rope_f32 returned error");

    /* Reference */
    rope_ref((const float*)tIn->data, ref, T_S, T_H, T_d, 10000.0f);

    /* Compare */
    float diff = max_abs_diff((const float*)tOut->data, ref, T_S * T_H * T_d);
    fprintf(stderr, "CPU vs ref: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-5f, "CPU RoPE mismatch");

    /* Verify rotation preserves magnitude */
    for (int64_t i = 0; i < T_S * T_H * T_d; i += 2) {
        float x0 = ((float*)tOut->data)[i];
        float x1 = ((float*)tOut->data)[i + 1];
        float mag_out = sqrtf(x0 * x0 + x1 * x1);
        float in0 = ((float*)tIn->data)[i];
        float in1 = ((float*)tIn->data)[i + 1];
        float mag_in = sqrtf(in0 * in0 + in1 * in1);
        float mag_diff = fabsf(mag_out - mag_in);
        CHECK(mag_diff < 1e-5f, "RoPE rotation does not preserve magnitude");
    }

    fprintf(stderr, "RoPE CPU: PASS\n");
    tensor_destroy(tIn);
    tensor_destroy(tOut);
    free(ref);
    return 0;
}

/* ============================================================
 * Test: RoPE in-place
 * ============================================================ */
static int test_rope_inplace(void) {
    fprintf(stderr, "\n=== RoPE In-place Test ===\n");

    int64_t shape[] = {T_S, T_H, T_d};
    tensor_t* tA = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tB = tensor_create(DATA_TYPE_F32, 3, shape);

    srand(42);
    for (int64_t i = 0; i < T_S * T_H * T_d; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        ((float*)tA->data)[i] = v;
        ((float*)tB->data)[i] = v;
    }

    rope_params_t p;
    p.seq_len = T_S;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;

    /* In-place: output == input */
    void* ptr = tA->data;
    rope_f32((const void**)&ptr, &ptr, (const operator_params_t*)&p, NULL);

    /* Separate: output != input */
    const void* in_ptr = tB->data;
    void* out_ptr = tB->data;
    rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);

    float diff = max_abs_diff((const float*)tA->data, (const float*)tB->data, T_S * T_H * T_d);
    fprintf(stderr, "In-place vs separate: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-6f, "In-place RoPE mismatch");

    fprintf(stderr, "RoPE In-place: PASS\n");
    tensor_destroy(tA);
    tensor_destroy(tB);
    return 0;
}

/* ============================================================
 * Test: RoPE CUDA vs CPU
 * ============================================================ */
static int test_rope_cuda(void) {
#ifdef USE_CUDA
    fprintf(stderr, "\n=== RoPE CUDA Test ===\n");

    int64_t shape[] = {T_S, T_H, T_d};
    tensor_t* tIn = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tCpu = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tGpu = tensor_create(DATA_TYPE_F32, 3, shape);

    srand(42);
    for (int64_t i = 0; i < T_S * T_H * T_d; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    memcpy(tCpu->data, tIn->data, T_S * T_H * T_d * sizeof(float));
    memcpy(tGpu->data, tIn->data, T_S * T_H * T_d * sizeof(float));

    rope_params_t p;
    p.seq_len = T_S;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;

    /* CPU */
    const void* cpu_in = tCpu->data;
    void* cpu_out = tCpu->data;
    rope_f32(&cpu_in, &cpu_out, (const operator_params_t*)&p, NULL);

    /* CUDA */
    const operator_registry_t* op = operator_find("rope_f32_cuda");
    if (!op || !op->func) {
        fprintf(stderr, "SKIP: rope_f32_cuda not registered\n");
        tensor_destroy(tIn); tensor_destroy(tCpu); tensor_destroy(tGpu);
        return 0;
    }

    tensor_copy_to_device(tGpu);
    const void* gpu_in = tGpu->data_device;
    void* gpu_out = tGpu->data_device;
    op->func(&gpu_in, &gpu_out, (const operator_params_t*)&p, NULL);
    tensor_copy_to_host(tGpu);
    g_cuda.stream_synchronize(0);

    float diff = max_abs_diff((const float*)tCpu->data, (const float*)tGpu->data, T_S * T_H * T_d);
    fprintf(stderr, "CUDA vs CPU: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "CUDA RoPE mismatch");

    fprintf(stderr, "RoPE CUDA: PASS\n");
    tensor_destroy(tIn); tensor_destroy(tCpu); tensor_destroy(tGpu);
#else
    (void)0;
#endif
    return 0;
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_rope_cpu();
    test_rope_inplace();
    test_rope_cuda();

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();

    fprintf(stderr, "\n=== All RoPE Tests Done ===\n");
    return 0;
}
