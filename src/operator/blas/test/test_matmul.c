#include "unity.h"
#include "operator.h"
#include "matmul_int.h"
#include <stdlib.h>
#include <string.h>

extern int operator_init_all(void);

#define TEST_M 4
#define TEST_N 3
#define TEST_K 5

static float s_A[TEST_M * TEST_K];
static float s_B[TEST_K * TEST_N];
static float s_C[TEST_M * TEST_N];

void setUp(void) {
    /* Fill with deterministic values */
    for (int i = 0; i < TEST_M * TEST_K; i++) s_A[i] = (float)(i + 1);
    for (int i = 0; i < TEST_K * TEST_N; i++) s_B[i] = (float)(i + 1);
    memset(s_C, 0, sizeof(s_C));
}

void tearDown(void) {}

void test_matmul_f32_basic(void) {
    matmul_params_t p = {.M = TEST_M, .N = TEST_N, .K = TEST_K};

    const void* inputs[]  = {s_A, s_B};
    void*       outputs[] = {s_C};

    const operator_registry_t* op = operator_find("matmul_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* Verify result: C[0][0] = sum_{k=0}^{K-1} A[0][k] * B[k][0] */
    float expected = 0;
    for (int k = 0; k < TEST_K; k++) expected += s_A[0 * TEST_K + k] * s_B[k * TEST_N + 0];
    TEST_ASSERT_EQUAL_FLOAT(expected, s_C[0]);
}

void test_matmul_f32_1x1(void) {
    float a = 2.0f, b = 3.0f, c = 0.0f;
    matmul_params_t p = {.M = 1, .N = 1, .K = 1};

    const void* inputs[]  = {&a, &b};
    void*       outputs[] = {&c};

    const operator_registry_t* op = operator_find("matmul_f32");
    TEST_ASSERT_NOT_NULL(op);

    op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_FLOAT(6.0f, c);
}

void test_matmul_f32_null_input(void) {
    matmul_params_t p = {.M = 2, .N = 2, .K = 2};
    const operator_registry_t* op = operator_find("matmul_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(NULL, NULL, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_matmul_f32_basic);
    RUN_TEST(test_matmul_f32_1x1);
    RUN_TEST(test_matmul_f32_null_input);
    return UNITY_END();
}
