#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "softmax_int.h"
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

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_softmax_simple);
    RUN_TEST(test_softmax_batch);
    RUN_TEST(test_softmax_null_check);
    return UNITY_END();
}
