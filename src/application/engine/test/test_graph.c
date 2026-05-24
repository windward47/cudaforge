#include "unity.h"
#include "graph.h"
#include "operator.h"
#include "platform.h"
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#endif
#include "conv_int.h"

extern int operator_init_all(void);

#define N 4

void setUp(void) {}
void tearDown(void) {}

/* Basic Input → ReLU → Output */
void test_graph_relu_basic(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    TEST_ASSERT_NOT_NULL(t0);
    TEST_ASSERT_NOT_NULL(t1);

    inference_graph_t* g = graph_create();
    TEST_ASSERT_NOT_NULL(g);

    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);
    TEST_ASSERT_TRUE(tt0 >= 0);
    TEST_ASSERT_TRUE(tt1 >= 0);

    /* Input node → t0 */
    int n_in = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    TEST_ASSERT_TRUE(n_in >= 0);

    /* ReLU: t0 → t1 */
    int n_relu = graph_add_node(g, OP_RELU, 1, (int[]){tt0}, 1, (int[]){tt1}, 0, NULL, NULL, 0);
    TEST_ASSERT_TRUE(n_relu >= 0);

    /* Output: t1 → */
    int n_out = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt1}, 0, NULL, 0, NULL, NULL, 0);
    TEST_ASSERT_TRUE(n_out >= 0);

    TEST_ASSERT_EQUAL_INT(0, graph_set_input(g, n_in));
    TEST_ASSERT_EQUAL_INT(0, graph_set_output(g, n_out));
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    /* Input data: {-1, 0, 2, -3} → expected ReLU: {0, 0, 2, 0} */
    float in_data[N] = {-1.0f, 0.0f, 2.0f, -3.0f};
    float expected[N] = {0.0f, 0.0f, 2.0f, 0.0f};
    memcpy(t0->data, in_data, sizeof(in_data));

    tensor_t* inputs[] = {t0};
    tensor_t* outputs[] = {t1};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, false));

    float* result = (float*)t1->data;
    for (int i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL_FLOAT(expected[i], result[i]);
    }

    graph_destroy(g);
}

/* Chain: Input → ReLU → ReLU → Output */
void test_graph_relu_chain(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    tensor_t* t2 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    TEST_ASSERT_NOT_NULL(t0);
    TEST_ASSERT_NOT_NULL(t1);
    TEST_ASSERT_NOT_NULL(t2);

    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);
    int tt2 = graph_add_tensor(g, t2);

    int n_in = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    (void)graph_add_node(g, OP_RELU, 1, (int[]){tt0}, 1, (int[]){tt1}, 0, NULL, NULL, 0);
    (void)graph_add_node(g, OP_RELU, 1, (int[]){tt1}, 1, (int[]){tt2}, 0, NULL, NULL, 0);
    int n_out = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt2}, 0, NULL, 0, NULL, NULL, 0);

    TEST_ASSERT_EQUAL_INT(0, graph_set_input(g, n_in));
    TEST_ASSERT_EQUAL_INT(0, graph_set_output(g, n_out));
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    float in_data[N] = {-2.0f, -1.0f, 0.0f, 5.0f};
    memcpy(t0->data, in_data, sizeof(in_data));

    tensor_t* inputs[] = {t0};
    tensor_t* outputs[] = {t2};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, false));

    /* ReLU(ReLU(x)) == ReLU(x) */
    float* result = (float*)t2->data;
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result[1]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result[2]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, result[3]);

    graph_destroy(g);
}

/* Cycle detection */
void test_graph_cycle(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    TEST_ASSERT_NOT_NULL(t0);
    TEST_ASSERT_NOT_NULL(t1);

    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);

    /* Node 0 produces t0, consumes t1 → cycle edge */
    graph_add_node(g, OP_RELU, 1, (int[]){tt1}, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    /* Node 1 produces t1, consumes t0 → cycle edge */
    graph_add_node(g, OP_RELU, 1, (int[]){tt0}, 1, (int[]){tt1}, 0, NULL, NULL, 0);

    TEST_ASSERT_TRUE(graph_build(g) < 0);

    graph_destroy(g);
}

/* Null safety */
void test_graph_null(void) {
    TEST_ASSERT_NULL(graph_create() ? NULL : (void*)1); /* just check it runs */
    TEST_ASSERT_TRUE(graph_add_node(NULL, OP_RELU, 0, NULL, 0, NULL, 0, NULL, NULL, 0) < 0);
    TEST_ASSERT_TRUE(graph_add_tensor(NULL, NULL) < 0);
    TEST_ASSERT_TRUE(graph_set_input(NULL, 0) < 0);
    TEST_ASSERT_TRUE(graph_set_output(NULL, 0) < 0);
    TEST_ASSERT_TRUE(graph_build(NULL) < 0);
    TEST_ASSERT_TRUE(graph_execute(NULL, NULL, NULL, false) < 0);
    graph_destroy(NULL);
}

/* Empty graph: no nodes, both build and execute should succeed (no-op) */
void test_graph_empty(void) {
    inference_graph_t* g = graph_create();
    TEST_ASSERT_NOT_NULL(g);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, NULL, NULL, false));
    graph_destroy(g);
}

/* Single-node graph: Input → (nothing else) */
void test_graph_input_only(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});
    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int n_in = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    graph_set_input(g, n_in);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    float in_data[4] = {1, 2, 3, 4};
    memcpy(t0->data, in_data, sizeof(in_data));
    tensor_t* inputs[] = {t0};
    tensor_t* outputs[] = {t0};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, false));

    graph_destroy(g);
}

/* Multi-input / multi-output: two ReLUs in parallel, Input→ReLU→Output for each */
void test_graph_multi_io(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){2});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){2});
    tensor_t* t2 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){2});
    tensor_t* t3 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){2});

    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);
    int tt2 = graph_add_tensor(g, t2);
    int tt3 = graph_add_tensor(g, t3);

    /* Two parallel inputs */
    int n_in0 = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    int n_in1 = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt1}, 0, NULL, NULL, 0);

    /* Two parallel ReLUs */
    graph_add_node(g, OP_RELU, 1, (int[]){tt0}, 1, (int[]){tt2}, 0, NULL, NULL, 0);
    graph_add_node(g, OP_RELU, 1, (int[]){tt1}, 1, (int[]){tt3}, 0, NULL, NULL, 0);

    /* Two outputs */
    int n_out0 = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt2}, 0, NULL, 0, NULL, NULL, 0);
    int n_out1 = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt3}, 0, NULL, 0, NULL, NULL, 0);

    graph_set_input(g, n_in0);
    graph_set_input(g, n_in1);
    graph_set_output(g, n_out0);
    graph_set_output(g, n_out1);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    float in0[2] = {-1, 3}, in1[2] = {2, -4};
    memcpy(t0->data, in0, sizeof(in0));
    memcpy(t1->data, in1, sizeof(in1));

    tensor_t* inputs[]  = {t0, t1};
    tensor_t* outputs[] = {t2, t3};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, false));

    float* r0 = (float*)t2->data;
    float* r1 = (float*)t3->data;
    TEST_ASSERT_EQUAL_FLOAT(0.0f, r0[0]);
    TEST_ASSERT_EQUAL_FLOAT(3.0f, r0[1]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, r1[0]);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, r1[1]);

    graph_destroy(g);
}

/* NULL weight in a node should be tolerated (weights are optional) */
void test_graph_null_weight(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});

    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);

    /* BatchNorm with NULL weights — should still execute (weights are checked as NULL) */
    int n_in = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);

    /* BatchNorm needs weight tensors; pass NULL pointer for weights array and 0 for num_weights */
    graph_add_node(g, OP_BATCHNORM, 1, (int[]){tt0}, 1, (int[]){tt1},
                   0, NULL, NULL, 0);
    int n_out = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt1}, 0, NULL, 0, NULL, NULL, 0);

    graph_set_input(g, n_in);
    graph_set_output(g, n_out);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    float in_data[4] = {1, 2, 3, 4};
    memcpy(t0->data, in_data, sizeof(in_data));

    tensor_t* inputs[]  = {t0};
    tensor_t* outputs[] = {t1};
    /* BatchNorm needs 5 inputs (x, gamma, beta, mean, var, hw) but we pass empty slots.
     * The op->func will reject because inputs are NULL. graph_execute should handle this. */
    int ret = graph_execute(g, inputs, outputs, false);
    /* BatchNorm with NULL weight data should fail at operator level */
    TEST_ASSERT_TRUE(ret < 0);

    graph_destroy(g);
}

/* End-to-end CNN pipeline: Input(3x3) → Conv2D(2x2) → ReLU → Output */
void test_graph_cnn_pipeline(void) {
    tensor_t* t_in  = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){9});
    tensor_t* t_cnv = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});
    tensor_t* t_rel = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});
    /* weight: 2x2 identity */
    tensor_t* t_w   = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){4});
    TEST_ASSERT_NOT_NULL(t_in);
    TEST_ASSERT_NOT_NULL(t_cnv);
    TEST_ASSERT_NOT_NULL(t_rel);
    TEST_ASSERT_NOT_NULL(t_w);

    inference_graph_t* g = graph_create();
    int tt_in  = graph_add_tensor(g, t_in);
    int tt_cnv = graph_add_tensor(g, t_cnv);
    int tt_rel = graph_add_tensor(g, t_rel);

    tensor_t* conv_weights[] = {t_w};
    int n_in, n_cnv, n_rel, n_out;
    /* N C H W K KH KW pad_h pad_w stride_h stride_w dil_h dil_w groups */
    conv_params_t cp = {1,1,3,3, 1, 2,2, 0,0, 1,1, 1,1, 1};

    n_in  = graph_add_node(g, OP_INPUT,   0, NULL,           1, (int[]){tt_in},  0, NULL,      NULL, 0);
    n_cnv = graph_add_node(g, OP_CONV2D,  1, (int[]){tt_in}, 1, (int[]){tt_cnv}, 1, conv_weights, &cp, sizeof(cp));
    n_rel = graph_add_node(g, OP_RELU,    1, (int[]){tt_cnv},1, (int[]){tt_rel}, 0, NULL,      NULL, 0);
    n_out = graph_add_node(g, OP_OUTPUT,  1, (int[]){tt_rel},0, NULL,            0, NULL,      NULL, 0);

    graph_set_input(g, n_in);
    graph_set_output(g, n_out);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    /* Input: 3x3 = [1,2,3,4,5,6,7,8,9], weight: identity 2x2 */
    float in_data[9] = {1,2,3,4,5,6,7,8,9};
    memcpy(t_in->data, in_data, sizeof(in_data));
    float w_data[4] = {1,0,0,1};
    memcpy(t_w->data, w_data, sizeof(w_data));

    tensor_t* inputs[]  = {t_in};
    tensor_t* outputs[] = {t_rel};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, false));

    /* Conv output: [6,8,12,14], ReLU: [6,8,12,14] */
    float* result = (float*)t_rel->data;
    float expected[4] = {6.0f, 8.0f, 12.0f, 14.0f};
    for (int i = 0; i < 4; i++)
        TEST_ASSERT_EQUAL_FLOAT(expected[i], result[i]);

    graph_destroy(g);
}

#ifdef USE_CUDA
/* CUDA: Input → ReLU → Output */
void test_graph_relu_cuda(void) {
    tensor_t* t0 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    tensor_t* t1 = tensor_create(DATA_TYPE_F32, 1, (int64_t[]){N});
    TEST_ASSERT_NOT_NULL(t0);
    TEST_ASSERT_NOT_NULL(t1);

    inference_graph_t* g = graph_create();
    int tt0 = graph_add_tensor(g, t0);
    int tt1 = graph_add_tensor(g, t1);

    int n_in = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tt0}, 0, NULL, NULL, 0);
    graph_add_node(g, OP_RELU, 1, (int[]){tt0}, 1, (int[]){tt1}, 0, NULL, NULL, 0);
    int n_out = graph_add_node(g, OP_OUTPUT, 1, (int[]){tt1}, 0, NULL, 0, NULL, NULL, 0);

    graph_set_input(g, n_in);
    graph_set_output(g, n_out);
    TEST_ASSERT_EQUAL_INT(0, graph_build(g));

    float in_data[N] = {-1.0f, 0.0f, 2.0f, -3.0f};
    float expected[N] = {0.0f, 0.0f, 2.0f, 0.0f};
    memcpy(t0->data, in_data, sizeof(in_data));

    tensor_t* inputs[]  = {t0};
    tensor_t* outputs[] = {t1};
    TEST_ASSERT_EQUAL_INT(0, graph_execute(g, inputs, outputs, true));

    float* result = (float*)t1->data;
    for (int i = 0; i < N; i++)
        TEST_ASSERT_EQUAL_FLOAT(expected[i], result[i]);

    graph_destroy(g);
}
#endif /* USE_CUDA */

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif
    UNITY_BEGIN();
    RUN_TEST(test_graph_relu_basic);
    RUN_TEST(test_graph_relu_chain);
    RUN_TEST(test_graph_cycle);
    RUN_TEST(test_graph_null);
    RUN_TEST(test_graph_empty);
    RUN_TEST(test_graph_input_only);
    RUN_TEST(test_graph_multi_io);
    RUN_TEST(test_graph_null_weight);
    RUN_TEST(test_graph_cnn_pipeline);
#ifdef USE_CUDA
    RUN_TEST(test_graph_relu_cuda);
#endif
    int result = UNITY_END();
    platform_finalize();
    return result;
}
