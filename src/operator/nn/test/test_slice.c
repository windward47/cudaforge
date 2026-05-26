#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "slice_int.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_slice_basic(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("slice_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* 2D input [3, 4], slice rows 1:3, cols 0:2 (step 2) */
    int64_t in_shape[] = {3, 4};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, in_shape);
    float* id = (float*)in->data;
    for (int64_t i = 0; i < 12; i++) id[i] = (float)i;

    int64_t out_shape[] = {2, 1};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, out_shape);

    slice_params_t p;
    p.numel = 2;
    p.in_numel = 12;
    p.ndim = 2;
    p.in_shape[0] = 3; p.in_shape[1] = 4;
    p.out_shape[0] = 2; p.out_shape[1] = 1;
    p.starts[0] = 1; p.starts[1] = 0;
    p.steps[0] = 1; p.steps[1] = 2;
    p.in_strides[0] = 4; p.in_strides[1] = 1;

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    /* Row 1 col 0 = 4, Row 2 col 0 = 8 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 4.0f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 8.0f, od[1]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_slice_3d(void) {
    const operator_registry_t* op = operator_find("slice_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* 3D [2, 3, 4], slice all of dim 0/1, half of dim 2 */
    int64_t in_shape[] = {2, 3, 4};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 3, in_shape);
    float* id = (float*)in->data;
    for (int64_t i = 0; i < 24; i++) id[i] = (float)i;

    int64_t out_shape[] = {2, 3, 2};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 3, out_shape);

    slice_params_t p;
    p.numel = 12;
    p.in_numel = 24;
    p.ndim = 3;
    p.in_shape[0] = 2; p.in_shape[1] = 3; p.in_shape[2] = 4;
    p.out_shape[0] = 2; p.out_shape[1] = 3; p.out_shape[2] = 2;
    p.starts[0] = 0; p.starts[1] = 0; p.starts[2] = 1;
    p.steps[0] = 1; p.steps[1] = 1; p.steps[2] = 1;
    p.in_strides[0] = 12; p.in_strides[1] = 4; p.in_strides[2] = 1;

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    /* First element: [0,0,1] = 1; last: [1,2,2] = 22 */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 22.0f, od[11]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_slice_null_check(void) {
    const operator_registry_t* op = operator_find("slice_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    const void* inputs_ok[] = {buf};
    void* outputs_ok[] = {buf};
    slice_params_t p;
    p.numel = 4; p.in_numel = 4; p.ndim = 1;
    p.in_shape[0] = 4; p.out_shape[0] = 4;
    p.starts[0] = 0; p.steps[0] = 1; p.in_strides[0] = 1;

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, outputs_ok, (const operator_params_t*)&p, NULL));
    const void* inputs_null[] = {NULL};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_ok, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_slice_basic);
    RUN_TEST(test_slice_3d);
    RUN_TEST(test_slice_null_check);
    return UNITY_END();
}
