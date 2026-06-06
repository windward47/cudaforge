#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "softmax_int.h"
#include "test_utils.h"
#include <stdlib.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);
extern int register_softmax_f32(void);

void test_softmax_simple(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("softmax_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [1, 3] → softmax along axis 1 */
    int64_t in_shape[] = {1, 3};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, in_shape);
    float* idata = (float*)in->data;
    idata[0] = 1.0f; idata[1] = 2.0f; idata[2] = 3.0f;

    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, in_shape);

    softmax_params_t p = { .num_classes = 3, .num_blocks = 1 };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652] */
    float* od = (float*)out->data;
    float sum = od[0] + od[1] + od[2];
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, sum);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0900f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.2447f, od[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.6652f, od[2]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_softmax_batch(void) {
    const operator_registry_t* op = operator_find("softmax_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [2, 4] → softmax along axis 1 for each row */
    int64_t in_shape[] = {2, 4};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, in_shape);
    float* idata = (float*)in->data;
    /* Row 0: [0, 0, 0, 0] → uniform [0.25, 0.25, 0.25, 0.25] */
    /* Row 1: [0, 1, 2, 3] → softmax */
    idata[0]=0; idata[1]=0; idata[2]=0; idata[3]=0;
    idata[4]=0; idata[5]=1; idata[6]=2; idata[7]=3;

    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, in_shape);

    softmax_params_t p = { .num_classes = 4, .num_blocks = 2 };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    /* Row 0: uniform */
    for (int i = 0; i < 4; i++) TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.25f, od[i]);
    /* Row 1: sums to 1 */
    float sum1 = od[4] + od[5] + od[6] + od[7];
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, sum1);
    TEST_ASSERT_TRUE(od[4] < od[5] && od[5] < od[6] && od[6] < od[7]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_softmax_null_check(void) {
    const operator_registry_t* op = operator_find("softmax_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    softmax_params_t p = { .num_classes = 4, .num_blocks = 1 };

    const void* inputs_null[] = {NULL};
    void* outputs_ok[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func((const void*[]){buf}, outputs_ok, NULL, NULL));
}

/*
 * Multi-random-trial verification: 5 trials with random inputs.
 * Verifies: (1) output sums to 1.0, (2) all outputs >= 0, (3) determinism.
 */
static void softmax_random_trial(int trial) {
    unsigned seed = 42 + trial * 1000;
    int64_t rows = 4 + trial * 4;  /* 4, 8, 12, 16, 20 */
    int64_t cols = 32 + trial * 16; /* 32, 48, 64, 80, 96 */
    int64_t n = rows * cols;

    float* in  = (float*)malloc(n * sizeof(float));
    float* out1 = (float*)malloc(n * sizeof(float));
    float* out2 = (float*)malloc(n * sizeof(float));

    test_random_fill(in, n, seed);

    softmax_params_t p = { .num_classes = cols, .num_blocks = rows };
    const operator_registry_t* op = operator_find("softmax_f32");
    TEST_ASSERT_NOT_NULL(op);

    const void* inputs[] = {in};

    /* Run 1 */
    void* outputs1[] = {out1};
    int ret1 = op->func(inputs, outputs1, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret1);

    /* Verify: each row sums to 1.0, all values >= 0 */
    for (int64_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int64_t c = 0; c < cols; c++) {
            float v = out1[r * cols + c];
            TEST_ASSERT_TRUE(v >= 0.0f);
            sum += v;
        }
        TEST_ASSERT_FLOAT_WITHIN(1e-4, 1.0f, sum);
    }

    /* Run 2 — determinism check */
    void* outputs2[] = {out2};
    int ret2 = op->func(inputs, outputs2, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret2);

    float max_diff = test_max_abs_diff(out1, out2, n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, max_diff);

    free(in); free(out1); free(out2);
}

void test_softmax_random_trials(void) {
    RUN_N_TRIALS(5, softmax_random_trial);
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_softmax_simple);
    RUN_TEST(test_softmax_batch);
    RUN_TEST(test_softmax_null_check);
    RUN_TEST(test_softmax_random_trials);
    return UNITY_END();
}
