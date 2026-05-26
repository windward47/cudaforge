#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "concat_int.h"
#include <stdlib.h>
#include <string.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_concat_two_inputs(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("concat_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* Two inputs: [1, 2, 2, 2] and [1, 3, 2, 2] → output [1, 5, 2, 2] */
    int64_t N = 1, H = 2, W = 2;
    int64_t C0 = 2, C1 = 3;
    tensor_t* in0 = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C0, H, W});
    tensor_t* in1 = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C1, H, W});
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, (int64_t[]){N, C0 + C1, H, W});

    float* d0 = (float*)in0->data;
    float* d1 = (float*)in1->data;
    for (int64_t i = 0; i < N * C0 * H * W; i++) d0[i] = (float)(i + 1);
    for (int64_t i = 0; i < N * C1 * H * W; i++) d1[i] = (float)(i + 100);

    concat_params_t p;
    memset(&p, 0, sizeof(p));
    p.axis = 1; p.num_inputs = 2;
    p.H = H; p.W = W;
    p.C_total = C0 + C1;
    p.C_per_input[0] = C0; p.C_per_input[1] = C1;
    p.C_offset[0] = 0; p.C_offset[1] = C0;
    p.total_numel = N * (C0 + C1) * H * W;

    const void* inputs[] = {in0->data, in1->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* Verify: for each n,h,w, in0's channels come first, then in1's */
    float* od = (float*)out->data;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t h = 0; h < H; h++) {
            for (int64_t w = 0; w < W; w++) {
                for (int64_t c = 0; c < C0; c++) {
                    int64_t out_idx = ((n * (C0 + C1) + c) * H + h) * W + w;
                    int64_t in0_idx = ((n * C0 + c) * H + h) * W + w;
                    TEST_ASSERT_FLOAT_WITHIN(0.001f, d0[in0_idx], od[out_idx]);
                }
                for (int64_t c = 0; c < C1; c++) {
                    int64_t out_idx = ((n * (C0 + C1) + C0 + c) * H + h) * W + w;
                    int64_t in1_idx = ((n * C1 + c) * H + h) * W + w;
                    TEST_ASSERT_FLOAT_WITHIN(0.001f, d1[in1_idx], od[out_idx]);
                }
            }
        }
    }

    tensor_destroy(in0); tensor_destroy(in1); tensor_destroy(out);
}

void test_concat_null_check(void) {
    const operator_registry_t* op = operator_find("concat_f32");
    TEST_ASSERT_NOT_NULL(op);

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, NULL, NULL, NULL));

    float buf[16];
    const void* inputs[] = {buf, buf};
    void* outputs[] = {buf};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs, outputs, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_concat_two_inputs);
    RUN_TEST(test_concat_null_check);
    return UNITY_END();
}
