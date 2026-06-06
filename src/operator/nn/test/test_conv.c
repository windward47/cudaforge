#include "unity.h"
#include "operator.h"
#include "conv_int.h"
#include "test_utils.h"

extern int operator_init_all(void);

void setUp(void) {}
void tearDown(void) {}

void test_conv2d_f32_basic(void) {
    /* Simple 1x3x3 input, 2x2 kernel identity-like, golden values computed by hand */
    int64_t N=1,C=1,H=3,W=3,K=1,KH=2,KW=2;
    int64_t OH=(H-KH)/1+1, OW=(W-KW)/1+1;
    float input[9] = {1,2,3, 4,5,6, 7,8,9};
    float weight[4] = {1,0, 0,1};
    float output[4] = {0};
    /* Expected: [1*1+2*0+4*0+5*1=6, 2*1+3*0+5*0+6*1=8,
                   4*1+5*0+7*0+8*1=12, 5*1+6*0+8*0+9*1=14] */
    conv_params_t p = {
        .N=N, .C=C, .H=H, .W=W, .K=K,
        .kernel_h=KH, .kernel_w=KW,
        .pad_h=0, .pad_w=0,
        .stride_h=1, .stride_w=1,
        .dilation_h=1, .dilation_w=1,
        .groups=1,
    };

    const void* inputs[]  = {input, weight, NULL};
    void*       outputs[] = {output};

    const operator_registry_t* op = operator_find("conv2d_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 6.0f,  output[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 8.0f,  output[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 12.0f, output[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 14.0f, output[3]);

    /* stride=2 variant: input 5x5, kernel 3x3, pad=0, stride=2
       OH = (5-3)/2+1 = 2, output is 2x2, each = sum of 3x3 ones = 9 */
    float input55[25];
    for (int i = 0; i < 25; i++) input55[i] = 1.0f;
    float w33[9] = {1,1,1, 1,1,1, 1,1,1};
    float out22[4] = {0};
    conv_params_t p2 = {
        .N=1,.C=1,.H=5,.W=5,.K=1,.kernel_h=3,.kernel_w=3,
        .pad_h=0,.pad_w=0,.stride_h=2,.stride_w=2,
        .dilation_h=1,.dilation_w=1,.groups=1,
    };
    const void* inputs2[] = {input55, w33, NULL};
    void* outputs2[] = {out22};
    ret = op->func(inputs2, outputs2, (const operator_params_t*)&p2, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, out22[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, out22[1]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, out22[2]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 9.0f, out22[3]);
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

/*
 * Multi-random-trial verification: 5 trials with random inputs.
 * Runs CPU conv2d twice with same input, checks determinism.
 */
static void conv_random_trial(int trial) {
    unsigned seed = 42 + trial * 1000;
    int64_t N=1, C=3, H=16, W=16, K=8, KH=3, KW=3;
    int64_t OH = H - KH + 1, OW = W - KW + 1;
    int64_t in_n = N * C * H * W;
    int64_t w_n  = K * C * KH * KW;
    int64_t out_n = N * K * OH * OW;

    float* input  = (float*)malloc(in_n * sizeof(float));
    float* weight = (float*)malloc(w_n  * sizeof(float));
    float* out1   = (float*)malloc(out_n * sizeof(float));
    float* out2   = (float*)malloc(out_n * sizeof(float));

    test_random_fill(input,  in_n, seed);
    test_random_fill(weight, w_n,  seed + 1);
    memset(out1, 0, out_n * sizeof(float));
    memset(out2, 0, out_n * sizeof(float));

    conv_params_t p = {
        .N=N, .C=C, .H=H, .W=W, .K=K,
        .kernel_h=KH, .kernel_w=KW,
        .pad_h=0, .pad_w=0, .stride_h=1, .stride_w=1,
        .dilation_h=1, .dilation_w=1, .groups=1,
    };

    const operator_registry_t* op = operator_find("conv2d_f32");
    TEST_ASSERT_NOT_NULL(op);

    const void* inputs[]  = {input, weight, NULL};

    /* Run twice — should be deterministic */
    void* outputs1[] = {out1};
    int ret1 = op->func(inputs, outputs1, (const operator_params_t*)&p, NULL);
    void* outputs2[] = {out2};
    int ret2 = op->func(inputs, outputs2, (const operator_params_t*)&p, NULL);

    TEST_ASSERT_EQUAL_INT(0, ret1);
    TEST_ASSERT_EQUAL_INT(0, ret2);

    /* Verify determinism: same input → same output */
    float max_diff = test_max_abs_diff(out1, out2, out_n);
    TEST_ASSERT_FLOAT_WITHIN(1e-6, 0.0f, max_diff);

    free(input); free(weight); free(out1); free(out2);
}

void test_conv2d_f32_random_trials(void) {
    RUN_N_TRIALS(5, conv_random_trial);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_conv2d_f32_basic);
    RUN_TEST(test_conv2d_f32_null_input);
    RUN_TEST(test_conv2d_f32_random_trials);
    return UNITY_END();
}
