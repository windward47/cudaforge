#include "unity.h"
#include "operator.h"
#include <math.h>

static float s_input[8]  = {-1.0f, 0.0f, 2.5f, -3.0f, 5.0f, -0.5f, 0.0f, 100.0f};
static float s_output[8] = {0};
static int64_t s_n = 8;

void setUp(void) {
    for (int i = 0; i < 8; i++) s_output[i] = 0.0f;
}

void tearDown(void) {}

void test_relu_f32_basic(void) {
    const void* inputs[]  = {s_input, &s_n};
    void*       outputs[] = {s_output};

    const operator_registry_t* op = operator_find("relu_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_EQUAL_FLOAT(0.0f,   s_output[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   s_output[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.5f,   s_output[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   s_output[3]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f,   s_output[4]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   s_output[5]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   s_output[6]);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, s_output[7]);
}

void test_relu_f32_in_place(void) {
    float buf[8] = {-1.0f, 0.0f, 2.5f, -3.0f, 5.0f, -0.5f, 0.0f, 100.0f};
    const void* inputs[]  = {buf, &s_n};
    void*       outputs[] = {buf};  /* same buffer: in-place */

    const operator_registry_t* op = operator_find("relu_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_EQUAL_FLOAT(0.0f,   buf[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   buf[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.5f,   buf[2]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   buf[3]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f,   buf[4]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   buf[5]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f,   buf[6]);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, buf[7]);
}

void test_relu_f32_null_input(void) {
    const operator_registry_t* op = operator_find("relu_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(NULL, NULL, NULL, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

extern int operator_init_all(void);

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_relu_f32_basic);
    RUN_TEST(test_relu_f32_in_place);
    RUN_TEST(test_relu_f32_null_input);
    return UNITY_END();
}
