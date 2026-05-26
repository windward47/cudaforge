#include "unity.h"
#include "operator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

static float s_in[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 100.0f};
static float s_out[6] = {0};
static int64_t s_n = 6;

void setUp(void) {
    for (int i = 0; i < 6; i++) s_out[i] = 0.0f;
}

void tearDown(void) {}

void test_sigmoid_f32(void) {
    const void* inputs[] = {s_in, &s_n};
    void* outputs[] = {s_out};
    const operator_registry_t* op = operator_find("sigmoid_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.5f, s_out[2]);       /* sigmoid(0) = 0.5 */
    TEST_ASSERT(s_out[0] > 0.0f && s_out[0] < 0.5f);      /* sigmoid(-2) near 0 */
    TEST_ASSERT(s_out[4] > 0.5f && s_out[4] < 1.0f);      /* sigmoid(2) near 1 */
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, s_out[5]);       /* sigmoid(100) ≈ 1 */
}

void test_gelu_f32(void) {
    const void* inputs[] = {s_in, &s_n};
    void* outputs[] = {s_out};
    const operator_registry_t* op = operator_find("gelu_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, s_out[2]);       /* gelu(0) = 0 */
    TEST_ASSERT(s_out[4] > s_out[3]);                     /* gelu(2) > gelu(1) */
}

void test_silu_f32(void) {
    const void* inputs[] = {s_in, &s_n};
    void* outputs[] = {s_out};
    const operator_registry_t* op = operator_find("silu_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 0.0f, s_out[2]);        /* silu(0) = 0 */
    TEST_ASSERT(s_out[2] > -0.1f && s_out[2] < 0.1f);      /* silu(0) ≈ 0 */
    TEST_ASSERT(s_out[4] > s_out[3]);                       /* silu(2) > silu(1) */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.7616f, s_out[4]);      /* silu(2) = 2/(1+e^(-2)) ≈ 1.7616 */
}

static void check_in_place(const char* op_name, int64_t n,
                            const float* expected, float tol) {
    float* buf = (float*)calloc((size_t)n, sizeof(float));
    float ref[4] = {-2.0f, 0.0f, 1.0f, 2.0f};
    memcpy(buf, ref, (size_t)n * sizeof(float));
    const void* inputs[] = {buf, &n};
    void* outputs[] = {buf};
    int ret = operator_find(op_name)->func(inputs, outputs, NULL, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    for (int64_t i = 0; i < n; i++)
        TEST_ASSERT_FLOAT_WITHIN(tol, expected[i], buf[i]);
    free(buf);
}

void test_sigmoid_in_place(void) {
    float expected[4] = {0.1192f, 0.5f, 0.7311f, 0.8808f};
    check_in_place("sigmoid_f32", 4, expected, 0.01f);
}

void test_gelu_in_place(void) {
    float expected[4] = {-0.0455f, 0.0f, 0.8413f, 1.9545f};
    check_in_place("gelu_f32", 4, expected, 0.01f);
}

void test_silu_in_place(void) {
    float expected[4] = {-0.2384f, 0.0f, 0.7311f, 1.7616f};
    check_in_place("silu_f32", 4, expected, 0.01f);
}

void test_activation_null_input(void) {
    const operator_registry_t* op = operator_find("sigmoid_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(NULL, NULL, NULL, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_sigmoid_f32);
    RUN_TEST(test_gelu_f32);
    RUN_TEST(test_silu_f32);
    RUN_TEST(test_sigmoid_in_place);
    RUN_TEST(test_gelu_in_place);
    RUN_TEST(test_silu_in_place);
    RUN_TEST(test_activation_null_input);
    return UNITY_END();
}
