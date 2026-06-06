#include "unity.h"
#include "operator.h"
#include "matmul_int.h"
#include "test_utils.h"
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

    const void* inputs[]  = {s_A, s_B, NULL};
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

    const void* inputs[]  = {&a, &b, NULL};
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

/*
 * Multi-random-trial verification: 5 trials with random inputs.
 * Runs CPU matmul twice with same input, checks determinism.
 */
static void matmul_random_trial(int trial) {
    unsigned seed = 42 + trial * 1000;
    int M = 16 + trial * 8;  /* 16, 24, 32, 40, 48 */
    int N = 16 + trial * 8;
    int K = 32 + trial * 16; /* 32, 48, 64, 80, 96 */

    float* A = (float*)malloc(M * K * sizeof(float));
    float* B = (float*)malloc(K * N * sizeof(float));
    float* C1 = (float*)malloc(M * N * sizeof(float));
    float* C2 = (float*)malloc(M * N * sizeof(float));

    test_random_fill(A, M * K, seed);
    test_random_fill(B, K * N, seed + 1);
    memset(C1, 0, M * N * sizeof(float));
    memset(C2, 0, M * N * sizeof(float));

    matmul_params_t p = {.M = M, .N = N, .K = K};
    const operator_registry_t* op = operator_find("matmul_f32");
    TEST_ASSERT_NOT_NULL(op);

    const void* inputs[] = {A, B, NULL};

    void* outputs1[] = {C1};
    int ret1 = op->func(inputs, outputs1, (const operator_params_t*)&p, NULL);
    void* outputs2[] = {C2};
    int ret2 = op->func(inputs, outputs2, (const operator_params_t*)&p, NULL);

    TEST_ASSERT_EQUAL_INT(0, ret1);
    TEST_ASSERT_EQUAL_INT(0, ret2);

    /* Verify determinism */
    float max_diff = test_max_abs_diff(C1, C2, M * N);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, max_diff);

    free(A); free(B); free(C1); free(C2);
}

void test_matmul_f32_random_trials(void) {
    RUN_N_TRIALS(5, matmul_random_trial);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_matmul_f32_basic);
    RUN_TEST(test_matmul_f32_1x1);
    RUN_TEST(test_matmul_f32_null_input);
    RUN_TEST(test_matmul_f32_random_trials);
    return UNITY_END();
}
