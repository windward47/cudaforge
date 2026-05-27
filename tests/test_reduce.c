/**                                                                                                                                                                                                                                                                                                                * ReduceSum / ReduceMax operator test (CPU + CUDA).
 */
#include "unity.h"
#include "operator.h"
#include "reduce_int.h"
#include "platform.h"
#include <stdio.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);
#ifdef USE_CUDA
extern int cuda_platform_init(int device_id);
extern void cuda_platform_finalize(void);
#endif

void setUp(void) {}
void tearDown(void) {}

/* --------------------------------------------------------------------------
 * Find op helper
 * -------------------------------------------------------------------------- */
static const operator_registry_t* find_op(const char* name) {
    const operator_registry_t* op = operator_find(name);
    if (!op) {
        printf("SKIP: operator '%s' not registered\n", name);
        TEST_IGNORE();
    }
    return op;
}

/* --------------------------------------------------------------------------
 * ReduceSum CPU
 * -------------------------------------------------------------------------- */
static void test_reduce_sum_f32(void) {
    /* Input: 2x3 = [1,2,3 ; 4,5,6], reduce axis=1 -> [6, 15] */
    float in[] = {1, 2, 3, 4, 5, 6};
    float out[2] = {0};
    reduce_params_t rp = { .reduce_size=3, .num_blocks=2, .total_elems=6, .op=REDUCE_SUM };

    const void* inputs[] = {in};
    void* outputs[] = {out};
    const operator_registry_t* op = operator_find("reduce_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(inputs, outputs, (const operator_params_t*)&rp, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, out[0]);   /* 1+2+3 */
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 15.0f, out[1]);  /* 4+5+6 */
}

/* --------------------------------------------------------------------------
 * ReduceMax CPU
 * -------------------------------------------------------------------------- */
static void test_reduce_max_f32(void) {
    float in[] = {1, 5, 3, 4, 2, 6};
    float out[2] = {0};
    reduce_params_t rp = { .reduce_size=3, .num_blocks=2, .total_elems=6, .op=REDUCE_MAX };

    const void* inputs[] = {in};
    void* outputs[] = {out};
    const operator_registry_t* op = operator_find("reduce_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(inputs, outputs, (const operator_params_t*)&rp, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.0f, out[0]);  /* max(1,5,3) */
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f, out[1]);  /* max(4,2,6) */
}

/* --------------------------------------------------------------------------
 * ReduceSum single element
 * -------------------------------------------------------------------------- */
static void test_reduce_sum_single(void) {
    float in[] = {3.14f};
    float out[1] = {0};
    reduce_params_t rp = { .reduce_size=1, .num_blocks=1, .total_elems=1, .op=REDUCE_SUM };

    const void* inputs[] = {in};
    void* outputs[] = {out};
    const operator_registry_t* op = operator_find("reduce_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(inputs, outputs, (const operator_params_t*)&rp, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.14f, out[0]);
}

#ifdef USE_CUDA
/* --------------------------------------------------------------------------
 * ReduceSum CUDA
 * -------------------------------------------------------------------------- */
static void test_reduce_sum_f32_cuda(void) {
    float in[] = {1, 2, 3, 4, 5, 6};
    float out[2] = {0};
    reduce_params_t rp = { .reduce_size=3, .num_blocks=2, .total_elems=6, .op=REDUCE_SUM };

    float* d_in = (float*)g_cuda.device_alloc(sizeof(in));
    float* d_out = (float*)g_cuda.device_alloc(sizeof(out));
    g_cuda.memcpy_h2d(d_in, in, sizeof(in), 0);

    const void* inputs[] = {d_in};
    void* outputs[] = {d_out};
    const operator_registry_t* op = operator_find("reduce_f32_cuda");
    if (!op) { printf("SKIP: reduce_f32_cuda not registered\n"); goto cleanup; }
    int ret = op->func(inputs, outputs, (const operator_params_t*)&rp, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    g_cuda.memcpy_d2h(out, d_out, sizeof(out), 0);
    g_cuda.stream_synchronize(0);

    TEST_ASSERT_FLOAT_WITHIN(1e-3, 6.0f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 15.0f, out[1]);

cleanup:
    g_cuda.device_free(d_in);
    g_cuda.device_free(d_out);
}

/* --------------------------------------------------------------------------
 * ReduceMax CUDA
 * -------------------------------------------------------------------------- */
static void test_reduce_max_f32_cuda(void) {
    float in[] = {1, 5, 3, 4, 2, 6};
    float out[2] = {0};
    reduce_params_t rp = { .reduce_size=3, .num_blocks=2, .total_elems=6, .op=REDUCE_MAX };

    float* d_in = (float*)g_cuda.device_alloc(sizeof(in));
    float* d_out = (float*)g_cuda.device_alloc(sizeof(out));
    g_cuda.memcpy_h2d(d_in, in, sizeof(in), 0);

    const void* inputs[] = {d_in};
    void* outputs[] = {d_out};
    const operator_registry_t* op = operator_find("reduce_f32_cuda");
    if (!op) { printf("SKIP: reduce_f32_cuda not registered\n"); goto cleanup; }
    int ret = op->func(inputs, outputs, (const operator_params_t*)&rp, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    g_cuda.memcpy_d2h(out, d_out, sizeof(out), 0);
    g_cuda.stream_synchronize(0);

    TEST_ASSERT_FLOAT_WITHIN(1e-3, 5.0f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-3, 6.0f, out[1]);

cleanup:
    g_cuda.device_free(d_in);
    g_cuda.device_free(d_out);
}
#endif

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main(void) {
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    UNITY_BEGIN();
    RUN_TEST(test_reduce_sum_f32);
    RUN_TEST(test_reduce_max_f32);
    RUN_TEST(test_reduce_sum_single);
#ifdef USE_CUDA
    RUN_TEST(test_reduce_sum_f32_cuda);
    RUN_TEST(test_reduce_max_f32_cuda);
#endif

    int result = UNITY_END();
#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    return result;
}
