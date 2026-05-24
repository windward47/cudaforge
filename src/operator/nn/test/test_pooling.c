#include "unity.h"
#include "operator.h"
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
    RUN_TEST(test_pool_null_input);
    return UNITY_END();
}
