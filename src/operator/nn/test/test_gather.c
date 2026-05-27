#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "gather_int.h"
#include <stdlib.h>
#include <string.h>

extern int operator_init_all(void);

void setUp(void) {}
void tearDown(void) {}

void test_gather_f32_axis0(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("gather_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: 3×2 matrix [[0,1],[2,3],[4,5]], gather rows [2,0] → [[4,5],[0,1]] */
    float data[6] = {0,1, 2,3, 4,5};
    float indices[2] = {2.0f, 0.0f};
    float out[4] = {0};

    gather_params_t p;
    memset(&p, 0, sizeof(p));
    p.axis = 0;
    p.num_indices = 2;
    p.block_size = 2;        /* 2 elements per row */
    p.outer_size = 1;        /* no outer batch dim */
    p.inner_size = 6;        /* total input = outer_size * dim_size * block_size = 1*3*2 = 6 */

    const void* inputs[] = {data, indices};
    void* outputs[] = {out};

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(0.001f, 4.0f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, out[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, out[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, out[3]);
}

void test_gather_f32_axis1(void) {
    const operator_registry_t* op = operator_find("gather_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Input: 2×3 matrix [[0,1,2],[3,4,5]], gather columns [2,0] → [[2,0],[5,3]] */
    float data[6] = {0,1,2, 3,4,5};
    float indices[2] = {2.0f, 0.0f};
    float out[4] = {0};

    gather_params_t p;
    memset(&p, 0, sizeof(p));
    p.axis = 1;
    p.num_indices = 2;
    p.block_size = 1;        /* 1 element per column entry */
    p.outer_size = 2;        /* 2 rows */
    p.inner_size = 3;        /* 3 cols * 1 block_elem = 3 */

    const void* inputs[] = {data, indices};
    void* outputs[] = {out};

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* Row 0: cols [2,0] → [2, 0]; Row 1: cols [2,0] → [5, 3] */
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f, out[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, out[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, out[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.0f, out[3]);
}

void test_gather_null_input(void) {
    const operator_registry_t* op = operator_find("gather_f32");
    TEST_ASSERT_NOT_NULL(op);
    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, NULL, NULL, NULL));

    float buf[4];
    float idx[2] = {0.0f, 1.0f};
    const void* inputs[] = {buf, idx};
    void* outputs[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs, outputs, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_gather_f32_axis0);
    RUN_TEST(test_gather_f32_axis1);
    RUN_TEST(test_gather_null_input);
    return UNITY_END();
}
