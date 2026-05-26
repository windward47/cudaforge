#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "split_int.h"
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_split_equal(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("split_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* 2D [2, 6], split axis 1 into 2 equal parts → [2, 3] each */
    int64_t in_shape[] = {2, 6};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, in_shape);
    float* id = (float*)in->data;
    for (int64_t i = 0; i < 12; i++) id[i] = (float)i;

    int64_t out_shape[] = {2, 3};
    tensor_t* out0 = tensor_create(DATA_TYPE_F32, 2, out_shape);
    tensor_t* out1 = tensor_create(DATA_TYPE_F32, 2, out_shape);

    split_params_t p;
    p.axis = 1;
    p.num_outputs = 2;
    p.ndim = 2;
    p.in_shape[0] = 2; p.in_shape[1] = 6;
    p.splits[0] = 3; p.splits[1] = 3;
    p.offsets[0] = 0; p.offsets[1] = 3;
    p.out_numel[0] = 6; p.out_numel[1] = 6;

    const void* inputs[] = {in->data};
    void* outputs[] = {out0->data, out1->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od0 = (float*)out0->data;
    float* od1 = (float*)out1->data;
    /* out0: [0,1,2,6,7,8] */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, od0[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f, od0[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 6.0f, od0[3]);
    /* out1: [3,4,5,9,10,11] */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.0f, od1[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, od1[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 9.0f, od1[3]);

    tensor_destroy(in); tensor_destroy(out0); tensor_destroy(out1);
}

void test_split_unequal(void) {
    const operator_registry_t* op = operator_find("split_f32");
    TEST_ASSERT_NOT_NULL(op);

    int64_t in_shape[] = {4};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 1, in_shape);
    float* id = (float*)in->data;
    for (int64_t i = 0; i < 4; i++) id[i] = (float)(i * 10);

    int64_t s0[] = {1}; tensor_t* out0 = tensor_create(DATA_TYPE_F32, 1, s0);
    int64_t s1[] = {3}; tensor_t* out1 = tensor_create(DATA_TYPE_F32, 1, s1);

    split_params_t p;
    p.axis = 0; p.num_outputs = 2; p.ndim = 1;
    p.in_shape[0] = 4;
    p.splits[0] = 1; p.splits[1] = 3;
    p.offsets[0] = 0; p.offsets[1] = 1;
    p.out_numel[0] = 1; p.out_numel[1] = 3;

    const void* inputs[] = {in->data};
    void* outputs[] = {out0->data, out1->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, ((float*)out0->data)[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.0f, ((float*)out1->data)[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 30.0f, ((float*)out1->data)[2]);

    tensor_destroy(in); tensor_destroy(out0); tensor_destroy(out1);
}

void test_split_null_check(void) {
    const operator_registry_t* op = operator_find("split_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    const void* inputs_ok[] = {buf};
    void* outputs_ok[] = {buf};
    split_params_t p;
    p.axis = 0; p.num_outputs = 1; p.ndim = 1;
    p.in_shape[0] = 4;
    p.splits[0] = 4; p.offsets[0] = 0; p.out_numel[0] = 4;

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_ok, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_split_equal);
    RUN_TEST(test_split_unequal);
    RUN_TEST(test_split_null_check);
    return UNITY_END();
}
