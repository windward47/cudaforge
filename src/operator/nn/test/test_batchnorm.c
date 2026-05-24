#include "unity.h"
#include "operator.h"
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

#define BN_CHANS 2
#define BN_HW 4

static float s_x[BN_CHANS*BN_HW]  = {1,2,3,4, 5,6,7,8};
static float s_gamma[BN_CHANS]     = {1.0f, 2.0f};
static float s_beta[BN_CHANS]      = {0.0f, 1.0f};
static float s_mean[BN_CHANS]      = {2.5f, 6.5f};
static float s_var[BN_CHANS]       = {1.25f, 1.25f};
static float s_y[BN_CHANS*BN_HW];
static int64_t s_hw = BN_HW;

void setUp(void) { memset(s_y, 0, sizeof(s_y)); }
void tearDown(void) {}

typedef struct { int64_t c; float eps; } bnp;

void test_batchnorm_f32_basic(void) {
    bnp p = {BN_CHANS, 1e-5f};
    const void* inputs[] = {s_x, s_gamma, s_beta, s_mean, s_var, &s_hw};
    void* outputs[] = {s_y};
    const operator_registry_t* op = operator_find("batchnorm_f32");
    TEST_ASSERT_NOT_NULL(op);

    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);

    TEST_ASSERT(s_y[0] < 0);
    TEST_ASSERT(s_y[1] < 0);
    TEST_ASSERT(s_y[2] > 0);
    TEST_ASSERT(s_y[3] > 0);
    TEST_ASSERT(s_y[4] < 1);
    TEST_ASSERT(s_y[7] > 1);
}

void test_batchnorm_null_input(void) {
    const operator_registry_t* op = operator_find("batchnorm_f32");
    TEST_ASSERT_NOT_NULL(op);
    int ret = op->func(NULL, NULL, NULL, NULL);
    TEST_ASSERT_TRUE(ret < 0);
}

int main(void) {
    operator_init_all();
    UNITY_BEGIN();
    RUN_TEST(test_batchnorm_f32_basic);
    RUN_TEST(test_batchnorm_null_input);
    return UNITY_END();
}
