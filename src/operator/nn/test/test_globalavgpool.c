#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "globalavgpool_int.h"
#include <stdlib.h>
#include <math.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);
extern int register_globalavgpool_f32(void);

void test_globalavgpool_4x4(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("globalavgpool_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [1, 2, 2, 2], each channel filled with known values */
    int64_t in_shape[] = {1, 2, 2, 2};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, in_shape);
    float* idata = (float*)in->data;
    /* C0: [1,2,3,4] → avg=2.5; C1: [5,6,7,8] → avg=6.5 */
    for (int i = 0; i < 8; i++) idata[i] = (float)(i + 1);

    int64_t out_shape[] = {1, 2, 1, 1};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, out_shape);

    globalavgpool_params_t p = { .N = 1, .C = 2, .H = 2, .W = 2 };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.5f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 6.5f, od[1]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_globalavgpool_batch(void) {
    const operator_registry_t* op = operator_find("globalavgpool_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [2, 1, 3, 3], all ones */
    int64_t in_shape[] = {2, 1, 3, 3};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, in_shape);
    float* idata = (float*)in->data;
    for (int i = 0; i < 18; i++) idata[i] = 1.0f;

    int64_t out_shape[] = {2, 1, 1, 1};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, out_shape);

    globalavgpool_params_t p = { .N = 2, .C = 1, .H = 3, .W = 3 };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, od[1]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_globalavgpool_null_check(void) {
    const operator_registry_t* op = operator_find("globalavgpool_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    globalavgpool_params_t p = { .N = 1, .C = 1, .H = 2, .W = 2 };

    const void* inputs_null[] = {NULL};
    void* outputs_ok[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func((const void*[]){buf}, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_globalavgpool_4x4);
    RUN_TEST(test_globalavgpool_batch);
    RUN_TEST(test_globalavgpool_null_check);
    return UNITY_END();
}
