#include "unity.h"
#include "inference_engine.h"
#include "onnx_loader.h"
#include "operator.h"
#include "platform.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

extern int operator_init_all(void);
#ifdef USE_CUDA
extern int cuda_platform_init(int device_id);
#endif

void setUp(void) {}
void tearDown(void) {}

static float ref_output[10];

static int load_binary(const char* path, void* buf, size_t expected_size) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(buf, 1, expected_size, fp);
    fclose(fp);
    return (n == expected_size) ? 0 : -1;
}

static int check_close(float a, float b, float tol) {
    float diff = (a > b) ? (a - b) : (b - a);
    return diff <= tol;
}

void test_classify_cpu(void) {
    /* Load reference */
    int rc = load_binary("tests/mnist_ref_output.bin", ref_output, sizeof(ref_output));
    TEST_ASSERT_EQUAL(0, rc);

    /* Load ONNX model */
    inference_session_t* session = inference_session_load("tests/mnist_classifier.onnx");
    TEST_ASSERT_NOT_NULL(session);
    TEST_ASSERT_EQUAL(1, inference_session_num_inputs(session));
    TEST_ASSERT_EQUAL(1, inference_session_num_outputs(session));

    /* Create input tensor */
    int64_t in_shape[] = {1, 1, 28, 28};
    tensor_t* input = tensor_create(DATA_TYPE_F32, 4, in_shape);
    TEST_ASSERT_NOT_NULL(input);
    rc = load_binary("tests/mnist_input.bin", input->data, 1 * 1 * 28 * 28 * sizeof(float));
    TEST_ASSERT_EQUAL(0, rc);

    /* Create output tensor */
    int64_t out_shape[] = {1, 10};
    tensor_t* output = tensor_create(DATA_TYPE_F32, 2, out_shape);
    TEST_ASSERT_NOT_NULL(output);

    /* Run CPU inference */
    tensor_t* inputs[]  = {input};
    tensor_t* outputs[] = {output};
    rc = inference_session_run(session, inputs, outputs, 0);
    TEST_ASSERT_EQUAL(0, rc);

    /* Verify - compare with reference. Allow some tolerance for fp differences */
    float* odata = (float*)output->data;
    float sum = 0.0f;
    int predicted = 0;
    float max_prob = odata[0];
    printf("\n  Classification results (CPU):\n");
    for (int i = 0; i < 10; i++) {
        printf("  Class %d: %.6f  (ref: %.6f)\n", i, odata[i], ref_output[i]);
        sum += odata[i];
        if (odata[i] > max_prob) { max_prob = odata[i]; predicted = i; }
    }
    printf("  Predicted: class %d (prob=%.4f)\n", predicted, max_prob);
    /* Use relaxed tolerance: 1e-3 for value comparison, 1e-3 for sum check */
    for (int i = 0; i < 10; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-3f, ref_output[i], odata[i]);
    }
    TEST_ASSERT_FLOAT_WITHIN(1e-3f, 1.0f, sum);

    tensor_destroy(input);
    tensor_destroy(output);
    inference_session_destroy(session);
}

#ifdef USE_CUDA
void test_classify_cuda(void) {
    /* Step 1: model load */
    inference_session_t* session = inference_session_load("tests/mnist_classifier.onnx");
    TEST_ASSERT_NOT_NULL(session);

    /* Step 2: create tensors */
    int64_t in_shape[] = {1, 1, 28, 28};
    tensor_t* input = tensor_create(DATA_TYPE_F32, 4, in_shape);
    TEST_ASSERT_NOT_NULL(input);
    load_binary("tests/mnist_input.bin", input->data, 1 * 1 * 28 * 28 * sizeof(float));

    /* Step 3: copy to device */
    tensor_copy_to_device(input);

    inference_session_destroy(session);
    tensor_destroy(input);
}
#endif

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    UNITY_BEGIN();
    RUN_TEST(test_classify_cpu);
#ifdef USE_CUDA
    RUN_TEST(test_classify_cuda);
#endif
    return UNITY_END();
}
