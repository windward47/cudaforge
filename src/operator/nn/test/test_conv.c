#include "unity.h"
#include "operator.h"
#include "conv_int.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

/* Simple 1x1 conv test: input 1×1×4×4, weight 2×1×1×1, pad=0, stride=1 */
#define IN_N 1
#define IN_C 1
#define IN_H 4
#define IN_W 4
#define OUT_K 2
#define KH 3
#define KW 3

static float s_input[IN_N * IN_C * IN_H * IN_W];
static float s_weight[OUT_K * IN_C * KH * KW];
static float s_output[IN_N * OUT_K * 2 * 2]; /* stride=2 → 2×2 output */

void setUp(void) {
    for (int i = 0; i < IN_N * IN_C * IN_H * IN_W; i++)
        s_input[i] = (float)(i % 5);
    for (int i = 0; i < OUT_K * IN_C * KH * KW; i++)
        s_weight[i] = (float)(i % 3 - 1);  /* -1, 0, 1 pattern */
    memset(s_output, 0, sizeof(s_output));
}

void tearDown(void) {}

void test_conv2d_f32_basic(void) {
    conv_params_t p = {
        .N = IN_N, .C = IN_C, .H = IN_H, .W = IN_W,
        .K = OUT_K,
        .kernel_h = KH, .kernel_w = KW,
        .pad_h = 0, .pad_w = 0,
        .stride_h = 2, .stride_w = 2,
        .dilation_h = 1, .dilation_w = 1,
        .groups = 1,
    };

    const void* inputs[]  = {s_input, s_weight};
    void*       outputs[] = {s_output};

    const operator_registry_t* op = operator_find("conv2d_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    /* Verify first output element isn't NaN or 0 for all elements */
    float max_val = 0.0f;
    for (int i = 0; i < IN_N * OUT_K * 2 * 2; i++) {
        if (fabsf(s_output[i]) > max_val) max_val = fabsf(s_output[i]);
    }
    TEST_ASSERT(max_val > 0.0f);
}

void test_conv2d_f32_null_input(void) {
    conv_params_t p = {
        .N = 1, .C = 1, .H = 4, .W = 4, .K = 2,
        .kernel_h = 3, .kernel_w = 3,
    };
    const operator_registry_t* op = operator_find("conv2d_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(NULL, NULL, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_f32_basic);
    RUN_TEST(test_conv2d_f32_null_input);
    return UNITY_END();
}
