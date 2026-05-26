#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "mul_int.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_mul_same_shape(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("mul_f32");
    TEST_ASSERT_NOT_NULL(op);

    int64_t N = 16;
    int64_t shape[] = {N};
    tensor_t* a = tensor_create(DATA_TYPE_F32, 1, shape);
    tensor_t* b = tensor_create(DATA_TYPE_F32, 1, shape);
    tensor_t* out = tensor_create(DATA_TYPE_F32, 1, shape);
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (int64_t i = 0; i < N; i++) { ad[i] = (float)(i + 1); bd[i] = (float)(i + 1); }

    mul_params_t p = { .numel = N, .B_numel = N };
    const void* inputs[] = {a->data, b->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    for (int64_t i = 0; i < N; i++)
        TEST_ASSERT_FLOAT_WITHIN(0.001f, (float)((i + 1) * (i + 1)), od[i]);

    tensor_destroy(a); tensor_destroy(b); tensor_destroy(out);
}

void test_mul_broadcast_c(void) {
    const operator_registry_t* op = operator_find("mul_f32");
    TEST_ASSERT_NOT_NULL(op);

    int64_t N = 3, C = 4;
    int64_t shape_a[] = {N, C};
    int64_t shape_b[] = {C};
    tensor_t* a = tensor_create(DATA_TYPE_F32, 2, shape_a);
    tensor_t* b = tensor_create(DATA_TYPE_F32, 1, shape_b);
    int64_t out_shape[] = {N, C};
    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, out_shape);
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (int64_t i = 0; i < N * C; i++) ad[i] = (float)(i + 1);
    for (int64_t i = 0; i < C; i++) bd[i] = (float)(i + 1);

    mul_params_t p = { .numel = N * C, .B_numel = C };
    const void* inputs[] = {a->data, b->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    for (int64_t n = 0; n < N; n++)
        for (int64_t c = 0; c < C; c++)
            TEST_ASSERT_FLOAT_WITHIN(0.001f, (float)(n * C + c + 1) * (float)(c + 1), od[n * C + c]);

    tensor_destroy(a); tensor_destroy(b); tensor_destroy(out);
}

void test_mul_scalar(void) {
    const operator_registry_t* op = operator_find("mul_f32");
    TEST_ASSERT_NOT_NULL(op);

    int64_t N = 8;
    int64_t shape[] = {N};
    tensor_t* a = tensor_create(DATA_TYPE_F32, 1, shape);
    tensor_t* b = tensor_create(DATA_TYPE_F32, 1, shape);
    tensor_t* out = tensor_create(DATA_TYPE_F32, 1, shape);
    float* ad = (float*)a->data;
    float* bd = (float*)b->data;
    for (int64_t i = 0; i < N; i++) ad[i] = (float)(i + 1);
    bd[0] = 3.0f;

    mul_params_t p = { .numel = N, .B_numel = 1 };
    const void* inputs[] = {a->data, b->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    float* od = (float*)out->data;
    for (int64_t i = 0; i < N; i++)
        TEST_ASSERT_FLOAT_WITHIN(0.001f, (float)(i + 1) * 3.0f, od[i]);

    tensor_destroy(a); tensor_destroy(b); tensor_destroy(out);
}

void test_mul_null_check(void) {
    const operator_registry_t* op = operator_find("mul_f32");
    TEST_ASSERT_NOT_NULL(op);

    float buf[4];
    const void* inputs_ok[] = {buf, buf};
    void* outputs_ok[] = {buf};
    mul_params_t p = { .numel = 4, .B_numel = 4 };

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, outputs_ok, (const operator_params_t*)&p, NULL));
    const void* inputs_null[] = {NULL, buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_ok, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_mul_same_shape);
    RUN_TEST(test_mul_broadcast_c);
    RUN_TEST(test_mul_scalar);
    RUN_TEST(test_mul_null_check);
    return UNITY_END();
}
