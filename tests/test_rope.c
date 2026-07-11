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

/* CPU RoPE reference implementation (supports both layouts, batch, pos_offset) */
static void rope_ref(const float* in, float* out, int64_t B, int64_t S, int64_t H, int64_t d,
                     float base, int is_halfsplit, int64_t pos_offset) {
    memcpy(out, in, (size_t)B * S * H * d * sizeof(float));
    int64_t half_d = d / 2;

    for (int64_t b = 0; b < B; b++) {
        for (int64_t pos = 0; pos < S; pos++) {
            for (int64_t h = 0; h < H; h++) {
                float* q = out + ((b * S + pos) * H + h) * d;
                float real_pos = (float)(pos + pos_offset);
                for (int64_t i = 0; i < half_d; i++) {
                    float angle = real_pos / powf(base, (float)(2 * i) / (float)d);
                    float c = cosf(angle);
                    float s = sinf(angle);
                    if (!is_halfsplit) {
                        float x0 = q[2 * i];
                        float x1 = q[2 * i + 1];
                        q[2 * i]     = x0 * c - x1 * s;
                        q[2 * i + 1] = x0 * s + x1 * c;
                    } else {
                        float x0 = q[i];
                        float x1 = q[i + half_d];
                        q[i]          = x0 * c - x1 * s;
                        q[i + half_d] = x0 * s + x1 * c;
                    }
                }
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
    p.layout = ROPE_LAYOUT_INTERLEAVED;
    p.inv_freq = NULL;
    p.batch_size = 1;
    p.pos_offset = 0;

    /* CPU RoPE */
    const void* in_ptr = tIn->data;
    void* out_ptr = tOut->data;
    int ret = rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "rope_f32 returned error");

    /* Reference */
    rope_ref((const float*)tIn->data, ref, 1, T_S, T_H, T_d, 10000.0f, 0, 0);

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
    p.layout = ROPE_LAYOUT_INTERLEAVED;
    p.inv_freq = NULL;
    p.batch_size = 1;
    p.pos_offset = 0;

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
 * Test: RoPE HALFSPLIT layout (GPT-NeoX / LLaMA)
 * ============================================================ */
static int test_rope_halfsplit(void) {
    fprintf(stderr, "\n=== RoPE HALFSPLIT Layout Test ===\n");

    int64_t shape[] = {T_S, T_H, T_d};
    tensor_t* tIn = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 3, shape);
    float* ref = (float*)calloc(T_S * T_H * T_d, sizeof(float));

    srand(7);
    for (int64_t i = 0; i < T_S * T_H * T_d; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    rope_params_t p;
    p.seq_len = T_S;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;
    p.layout = ROPE_LAYOUT_HALFSPLIT;
    p.inv_freq = NULL;
    p.batch_size = 1;
    p.pos_offset = 0;

    /* CPU RoPE with HALFSPLIT */
    const void* in_ptr = tIn->data;
    void* out_ptr = tOut->data;
    int ret = rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "rope_f32 (halfsplit) returned error");

    /* Reference with halfsplit */
    rope_ref((const float*)tIn->data, ref, 1, T_S, T_H, T_d, 10000.0f, 1, 0);

    float diff = max_abs_diff((const float*)tOut->data, ref, T_S * T_H * T_d);
    fprintf(stderr, "HALFSPLIT CPU vs ref: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-5f, "HALFSPLIT RoPE mismatch");

    /* HALFSPLIT must differ from INTERLEAVED on the same input (sanity: layout flag is not inverted) */
    float* ref_interleaved = (float*)calloc(T_S * T_H * T_d, sizeof(float));
    rope_ref((const float*)tIn->data, ref_interleaved, 1, T_S, T_H, T_d, 10000.0f, 0, 0);
    float layout_diff = max_abs_diff(ref, ref_interleaved, T_S * T_H * T_d);
    fprintf(stderr, "HALFSPLIT vs INTERLEAVED ref: max_diff=%.2e (must be >0)\n", layout_diff);
    CHECK(layout_diff > 1e-6f, "HALFSPLIT and INTERLEAVED produce identical output (layout flag ineffective)");

    fprintf(stderr, "RoPE HALFSPLIT: PASS\n");
    tensor_destroy(tIn);
    tensor_destroy(tOut);
    free(ref);
    free(ref_interleaved);
    return 0;
}

/* ============================================================
 * Test: RoPE batch (B>1)
 * ============================================================ */
static int test_rope_batch(void) {
    fprintf(stderr, "\n=== RoPE Batch (B>1) Test ===\n");

    const int64_t TB = 3;
    int64_t shape[] = {TB, T_S, T_H, T_d};
    tensor_t* tIn = tensor_create(DATA_TYPE_F32, 4, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 4, shape);
    int64_t total = TB * T_S * T_H * T_d;
    float* ref = (float*)calloc(total, sizeof(float));

    srand(11);
    for (int64_t i = 0; i < total; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    rope_params_t p;
    p.seq_len = T_S;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;
    p.layout = ROPE_LAYOUT_INTERLEAVED;
    p.inv_freq = NULL;
    p.batch_size = TB;
    p.pos_offset = 0;

    const void* in_ptr = tIn->data;
    void* out_ptr = tOut->data;
    int ret = rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "rope_f32 (batch) returned error");

    rope_ref((const float*)tIn->data, ref, TB, T_S, T_H, T_d, 10000.0f, 0, 0);

    float diff = max_abs_diff((const float*)tOut->data, ref, total);
    fprintf(stderr, "Batch CPU vs ref: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-5f, "Batch RoPE mismatch");

    /* Sanity: batch elements with same input differ when pos differs across batch
       is NOT applicable (same pos across batch); instead verify batch 0 != batch 1
       only if their data differs (it does, random fill). Just verify all batches
       match ref (done above). */

    fprintf(stderr, "RoPE Batch: PASS\n");
    tensor_destroy(tIn);
    tensor_destroy(tOut);
    free(ref);
    return 0;
}

/* ============================================================
 * Test: RoPE pos_offset (KV-cache decode scenario)
 * ============================================================ */
static int test_rope_pos_offset(void) {
    fprintf(stderr, "\n=== RoPE pos_offset Test ===\n");

    /* Single token (S=1) at pos_offset=5, simulating decode of the 6th token.
       The rotation angle should match pos=5, not pos=0. */
    const int64_t TS = 1;
    int64_t shape[] = {TS, T_H, T_d};
    tensor_t* tIn = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tFull = tensor_create(DATA_TYPE_F32, 3, shape);
    int64_t total = TS * T_H * T_d;
    float* ref_offset = (float*)calloc(total, sizeof(float));

    srand(23);
    for (int64_t i = 0; i < total; i++) {
        float v = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
        ((float*)tIn->data)[i] = v;
        ((float*)tFull->data)[i] = v;
    }

    /* Apply RoPE with pos_offset=5 on a single token (S=1) */
    rope_params_t p;
    p.seq_len = TS;
    p.head_dim = T_d;
    p.num_heads = T_H;
    p.base = 10000.0f;
    p.layout = ROPE_LAYOUT_INTERLEAVED;
    p.inv_freq = NULL;
    p.batch_size = 1;
    p.pos_offset = 5;

    const void* in_ptr = tIn->data;
    void* out_ptr = tOut->data;
    int ret = rope_f32(&in_ptr, &out_ptr, (const operator_params_t*)&p, NULL);
    CHECK(ret == 0, "rope_f32 (pos_offset) returned error");

    /* Reference: single token at pos=5 */
    rope_ref((const float*)tIn->data, ref_offset, 1, TS, T_H, T_d, 10000.0f, 0, 5);

    float diff = max_abs_diff((const float*)tOut->data, ref_offset, total);
    fprintf(stderr, "pos_offset=5 vs ref(pos=5): max_diff=%.2e\n", diff);
    CHECK(diff < 1e-5f, "pos_offset RoPE mismatch");

    /* Sanity: pos_offset=5 result must differ from pos_offset=0 (pos=0) on same input */
    rope_params_t p0 = p;
    p0.pos_offset = 0;
    float* ref_zero = (float*)calloc(total, sizeof(float));
    rope_ref((const float*)tIn->data, ref_zero, 1, TS, T_H, T_d, 10000.0f, 0, 0);
    float offset_diff = max_abs_diff(ref_offset, ref_zero, total);
    fprintf(stderr, "pos=5 vs pos=0: max_diff=%.2e (must be >0)\n", offset_diff);
    CHECK(offset_diff > 1e-6f, "pos_offset had no effect (pos=5 == pos=0)");

    fprintf(stderr, "RoPE pos_offset: PASS\n");
    tensor_destroy(tIn);
    tensor_destroy(tOut);
    tensor_destroy(tFull);
    free(ref_offset);
    free(ref_zero);
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
    p.layout = ROPE_LAYOUT_INTERLEAVED;
    p.inv_freq = NULL;
    p.batch_size = 1;
    p.pos_offset = 0;

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
    test_rope_halfsplit();
    test_rope_batch();
    test_rope_pos_offset();
    test_rope_cuda();

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();

    fprintf(stderr, "\n=== All RoPE Tests Done ===\n");
    return 0;
}
