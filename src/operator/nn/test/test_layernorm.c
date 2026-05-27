#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "layernorm_int.h"
#include <math.h>

extern int operator_init_all(void);

void setUp(void) {}
void tearDown(void) {}

void test_layernorm_f32_basic(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("layernorm_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* N=2, D=4: normalize each row of 4 elements */
    float x[8]     = {0.0f, 2.0f, 4.0f, 6.0f,  1.0f, 1.0f, 1.0f, 1.0f};
    float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float beta[4]  = {0.0f, 0.0f, 0.0f, 0.0f};
    float y[8]     = {0};

    layernorm_params_t p = {.N = 2, .normalized_size = 4, .epsilon = 1e-5f};
    const void* inputs[] = {x, gamma, beta};
    void* outputs[] = {y};

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* Row 0: mean=(0+2+4+6)/4=3, var=((9+1+1+9)/4)+eps=5+eps, std≈2.236 */
    /* Expected: (x-3)/2.236 ≈ [-1.3416, -0.4472, 0.4472, 1.3416] */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -1.3416f, y[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -0.4472f, y[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,  0.4472f, y[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,  1.3416f, y[3]);

    /* Row 1: mean=1, var=0, std=sqrt(eps)≈0.00316 → all 0 */
    for (int i = 4; i < 8; i++)
        TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, y[i]);
}

void test_layernorm_f32_with_affine(void) {
    const operator_registry_t* op = operator_find("layernorm_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Single row of 3 elements */
    float x[3]     = {0.0f, 3.0f, 6.0f};
    float gamma[3] = {2.0f, 1.0f, 0.5f};
    float beta[3]  = {1.0f, 0.0f, -1.0f};
    float y[3]     = {0};

    layernorm_params_t p = {.N = 1, .normalized_size = 3, .epsilon = 1e-5f};
    const void* inputs[] = {x, gamma, beta};
    void* outputs[] = {y};

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* mean=(0+3+6)/3=3, var=((9+0+9)/3)+eps=6+eps, std≈2.4495 */
    /* Row: [-1.2247, 0, 1.2247] * gamma + beta = [-1.4495, 0, -0.3876] */
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -1.4495f, y[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f,  0.0f,    y[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, -0.3876f, y[2]);
}

void test_layernorm_null_input(void) {
    const operator_registry_t* op = operator_find("layernorm_f32");
    TEST_ASSERT_NOT_NULL(op);
    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, NULL, NULL, NULL));

    float buf[4];
    const void* inputs[] = {buf, buf, buf};
    void* outputs[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs, outputs, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_layernorm_f32_basic);
    RUN_TEST(test_layernorm_f32_with_affine);
    RUN_TEST(test_layernorm_null_input);
    return UNITY_END();
}
