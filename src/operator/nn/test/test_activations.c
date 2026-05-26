#include "unity.h"
#include "operator.h"
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
    RUN_TEST(test_activation_null_input);
    return UNITY_END();
}
