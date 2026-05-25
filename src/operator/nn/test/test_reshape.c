#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "reshape_int.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);
extern int register_reshape_f32(void);

void test_reshape_4d_to_2d(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("reshape_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [1, 3, 4, 4] → Output: [1, 48] */
    int64_t in_shape[] = {1, 3, 4, 4};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, in_shape);
    float* idata = (float*)in->data;
    for (int64_t i = 0; i < 48; i++) idata[i] = (float)i;

    int64_t out_shape[] = {1, 48};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, out_shape);

    reshape_params_t p = { .numel = 48, .ndim = 2, .shape = {1, 48} };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    for (int64_t i = 0; i < 48; i++)
        TEST_ASSERT_FLOAT_WITHIN(0.001f, idata[i], od[i]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_reshape_2d_to_4d(void) {
    const operator_registry_t* op = operator_find("reshape_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: [2, 10] → Output: [2, 5, 2, 1] */
    int64_t in_shape[] = {2, 10};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, in_shape);
    float* idata = (float*)in->data;
    for (int64_t i = 0; i < 20; i++) idata[i] = (float)(i + 1);

    int64_t out_shape[] = {2, 5, 2, 1};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, out_shape);

    reshape_params_t p = { .numel = 20, .ndim = 4, .shape = {2, 5, 2, 1} };
    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    for (int64_t i = 0; i < 20; i++)
        TEST_ASSERT_FLOAT_WITHIN(0.001f, idata[i], od[i]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_reshape_null_check(void) {
    const operator_registry_t* op = operator_find("reshape_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    reshape_params_t p = { .numel = 4, .ndim = 1, .shape = {4} };

    const void* inputs_null[] = {NULL};
    void* outputs_ok[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func((const void*[]){buf}, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_reshape_4d_to_2d);
    RUN_TEST(test_reshape_2d_to_4d);
    RUN_TEST(test_reshape_null_check);
    return UNITY_END();
}
