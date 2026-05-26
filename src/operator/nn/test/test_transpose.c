#include "unity.h"
#include "operator.h"
#include "platform.h"
#include "transpose_int.h"
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

extern int operator_init_all(void);

void test_transpose_2d(void) {
    operator_init_all();
    const operator_registry_t* op = operator_find("transpose_f32");
    TEST_ASSERT_NOT_NULL(op);

    int64_t shape[] = {2, 3};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 2, shape);
    tensor_t* out = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){3, 2});
    float* id = (float*)in->data;
    for (int64_t i = 0; i < 6; i++) id[i] = (float)(i + 1);

    transpose_params_t p;
    p.ndim = 2;
    p.shape[0] = 2; p.shape[1] = 3;
    p.out_shape[0] = 3; p.out_shape[1] = 2;
    p.perm[0] = 1; p.perm[1] = 0;

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* Input [1,2,3; 4,5,6] → Output [1,4; 2,5; 3,6] */
    float* od = (float*)out->data;
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, od[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 4.0f, od[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f, od[2]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, od[3]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.0f, od[4]);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 6.0f, od[5]);

    tensor_destroy(in); tensor_destroy(out);
}

void test_transpose_nchw_to_nhwc(void) {
    const operator_registry_t* op = operator_find("transpose_f32");
    TEST_ASSERT_NOT_NULL(op);

    /* YOLOv8n typical: [1, 64, 80, 80] → [1, 80, 80, 64] */
    int64_t N = 1, C = 2, H = 2, W = 3;
    int64_t in_shape[] = {N, C, H, W};
    int64_t out_shape[] = {N, H, W, C};
    tensor_t* in = tensor_create(DATA_TYPE_F32, 4, in_shape);
    tensor_t* out = tensor_create(DATA_TYPE_F32, 4, out_shape);
    float* id = (float*)in->data;
    for (int64_t i = 0; i < N * C * H * W; i++) id[i] = (float)(i + 1);

    transpose_params_t p;
    p.ndim = 4;
    p.perm[0] = 0; p.perm[1] = 2; p.perm[2] = 3; p.perm[3] = 1;  /* NCHW → NHWC */
    for (int d = 0; d < 4; d++) p.shape[d] = in_shape[d];
    for (int d = 0; d < 4; d++) p.out_shape[d] = out_shape[d];

    const void* inputs[] = {in->data};
    void* outputs[] = {out->data};
    int ret = op->func(inputs, outputs, (const operator_params_t*)&p, NULL);
    TEST_ASSERT_EQUAL(0, ret);

    /* Verify: output[n][h][w][c] == input[n][c][h][w] */
    float* od = (float*)out->data;
    int64_t out_idx = 0;
    for (int64_t n = 0; n < N; n++)
        for (int64_t h = 0; h < H; h++)
            for (int64_t w = 0; w < W; w++)
                for (int64_t c = 0; c < C; c++) {
                    int64_t in_idx = ((n * C + c) * H + h) * W + w;
                    TEST_ASSERT_FLOAT_WITHIN(0.001f, id[in_idx], od[out_idx]);
                    out_idx++;
                }

    tensor_destroy(in); tensor_destroy(out);
}

void test_transpose_null_check(void) {
    const operator_registry_t* op = operator_find("transpose_f32");
    TEST_ASSERT_NOT_NULL(op);

    TEST_ASSERT_NOT_EQUAL(0, op->func(NULL, NULL, NULL, NULL));

    float buf[6];
    const void* inputs_ok[] = {buf, buf};
    void* outputs_ok[] = {buf};
    transpose_params_t p = {0};
    const void* inputs_null[] = {NULL};
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_null, outputs_ok, (const operator_params_t*)&p, NULL));
    TEST_ASSERT_NOT_EQUAL(0, op->func(inputs_ok, outputs_ok, NULL, NULL));
}

int main(void) {
    platform_init();
    UNITY_BEGIN();
    RUN_TEST(test_transpose_2d);
    RUN_TEST(test_transpose_nchw_to_nhwc);
    RUN_TEST(test_transpose_null_check);
    return UNITY_END();
}
