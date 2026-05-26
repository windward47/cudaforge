#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "resize_int.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_resize_nearest_2x(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("resize_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* 2x upsample: [1, 1, 2, 2] → [1, 1, 4, 4] */
    int64_t N = 1, C = 1, H = 2, W = 2;
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C, H, W});
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C, H * 2, W * 2});
    float* id = (float*)in->data;
    id[0] = 1.0f; id[1] = 2.0f;
    id[2] = 3.0f; id[3] = 4.0f;

    resize_params_t p;
    memset(&p, 0, sizeof(p));
    p.N = N; p.C = C;
    p.H_in = H; p.W_in = W;
    p.H_out = H * 2; p.W_out = W * 2;
    p.scale_h = 2.0f; p.scale_w = 2.0f;

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* Verify: each input pixel is expanded to a 2x2 block */
    float* od = (float*)out->data;
    int64_t expect_map[4][4] = {
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        {3, 3, 4, 4},
        {3, 3, 4, 4},
    };
    for (int h = 0; h < 4; h++)
        for (int w = 0; w < 4; w++)
            TEST_ASSERT_FLOAT_WITHIN(0.001f, (float)expect_map[h][w], od[h * 4 + w]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_resize_null_check(void) {
    const operator_registry_t* op = operator_find("resize_f32");
    TEST_ASSERT_NOT_NULL(op);

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, NULL, NULL, NULL));

    float buf[16];
    const void* inputs[] = {buf};
    void* outputs[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs, outputs, NULL, NULL));
}

void test_resize_bilinear_2x(void) {
    const operator_registry_t* op = operator_find("resize_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* 2x upsample bilinear: [1, 1, 2, 2] -> [1, 1, 4, 4] */
    int64_t N = 1, C = 1, H = 2, W = 2;
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C, H, W});
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C, H * 2, W * 2});
    float* id = (float*)in->data;
    id[0] = 1.0f; id[1] = 2.0f;
    id[2] = 3.0f; id[3] = 4.0f;

    resize_params_t p;
    memset(&p, 0, sizeof(p));
    p.N = N; p.C = C;
    p.H_in = H; p.W_in = W;
    p.H_out = H * 2; p.W_out = W * 2;
    p.scale_h = 2.0f; p.scale_w = 2.0f;
    p.mode = 1;  /* bilinear */

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* Verify no NaN/inf in output */
    float* od = (float*)out->data;
    for (int i = 0; i < 16; i++) {
        TEST_ASSERT(!isnan(od[i]) && !isinf(od[i]));
    }

    /* Bilinear: center region should be between corner values */
    TEST_ASSERT(od[5] > 1.0f && od[5] < 4.0f);  /* near center */
    TEST_ASSERT(od[0] < od[15]);  /* top-left < bottom-right */

    tensor_destroy(in); tensor_destroy(out);
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_resize_nearest_2x);
    RUN_TEST(test_resize_bilinear_2x);
    RUN_TEST(test_resize_null_check);
    return UNITY_END();
}
