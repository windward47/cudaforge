#include "unity.h"
#include "operator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

#define BATCH 1
#define CHANS 1
#define HEI 4
#define WID 4

static float s_in[BATCH*CHANS*HEI*WID] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
static float s_out[BATCH*CHANS*2*2];

void setUp(void) {
    memset(s_out, 0, sizeof(s_out));
}
void tearDown(void) {}

/* Params struct matching the one in pooling.c */
typedef struct { int64_t n,c,h,w, kh,kw, sh,sw, ph,pw; } pool_p;

void test_maxpool2d_f32_basic(void) {
    pool_p p = {BATCH,CHANS,HEI,WID, 2,2, 2,2, 0,0};
    const void* inputs[] = {s_in};
    void* outputs[] = {s_out};
    const operator_registry_t* op = operator_find("maxpool2d_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_EQUAL_FLOAT(6.0f,  s_out[0]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f,  s_out[1]);
    TEST_ASSERT_EQUAL_FLOAT(14.0f, s_out[2]);
    TEST_ASSERT_EQUAL_FLOAT(16.0f, s_out[3]);
}

void test_avgpool2d_f32_basic(void) {
    pool_p p = {BATCH,CHANS,HEI,WID, 2,2, 2,2, 0,0};
    const void* inputs[] = {s_in};
    void* outputs[] = {s_out};
    const operator_registry_t* op = operator_find("avgpool2d_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT_FLOAT_WITHIN(1e-5, 3.5f, s_out[0]);
    TEST_ASSERT_FLOAT_WITHIN(1e-5, 5.5f, s_out[1]);
}

void test_pool_wide_ow(void) {
    /* OW > 256 exercises the grid tiling path in pooling kernels.
     * W=260, KH=2, KW=2, stride=2, pad=0 → OW=129.
     * All ones input → max=1, avg=1 for every output element. */
    int64_t H=2, W=260, KH=2, KW=2, stride=2, pad=0;
    int64_t OH=1, OW=129;
    int64_t N=H*W;
    float* in = (float*)malloc((size_t)N * sizeof(float));
    float* out_max = (float*)calloc((size_t)OH*OW, sizeof(float));
    float* out_avg = (float*)calloc((size_t)OH*OW, sizeof(float));
    for (int64_t i = 0; i < N; i++) in[i] = 1.0f;

    pool_p p = {1,1,H,W, KH,KW, stride,stride, pad,pad};
    const void* inputs[] = {in};

    const operator_registry_t* max_op = operator_find("maxpool2d_f32");
    TEST_ASSERT_NOT_NULL(max_op);
    void* outputs_max[] = {out_max};
    int ret = max_op->func(inputs, outputs_max, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    for (int64_t i = 0; i < OH*OW; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, out_max[i]);

    const operator_registry_t* avg_op = operator_find("avgpool2d_f32");
    TEST_ASSERT_NOT_NULL(avg_op);
    void* outputs_avg[] = {out_avg};
    ret = avg_op->func(inputs, outputs_avg, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    for (int64_t i = 0; i < OH*OW; i++)
        TEST_ASSERT_FLOAT_WITHIN(1e-5, 1.0f, out_avg[i]);

    free(in); free(out_max); free(out_avg);
}

void test_pool_null_input(void) {
    const operator_registry_t* op = operator_find("maxpool2d_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(NULL, NULL, NULL, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_maxpool2d_f32_basic);
    RUN_TEST(test_avgpool2d_f32_basic);
    RUN_TEST(test_pool_wide_ow);
    RUN_TEST(test_pool_null_input);
    return UNITY_END();
}
