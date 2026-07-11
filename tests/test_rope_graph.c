/**
 * RoPE end-to-end integration test (R9-f).
 *
 * Verifies RoPE is correctly wired into the inference graph engine:
 * builds a native graph [OP_INPUT -> OP_ROPE -> OP_OUTPUT] and runs it
 * through graph_execute (the real dispatch path: enum -> op_name ->
 * operator_find -> execute), on both CPU and CUDA.
 *
 * This is NOT a unit test of the rope kernel (that's test_rope.c) -- it
 * validates the integration: OP_ROPE enum, op_name() mapping, registry
 * dispatch, params deep-copy through graph_add_node, and in-place execution
 * within the graph executor.
 */
#include "graph.h"
#include "platform.h"
#include "operator.h"
#include "rope_int.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

#define T_S 4
#define T_H 2
#define T_d 4

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

/* Reference RoPE (interleaved, B=1, pos_offset=0) */
static void rope_ref(const float* in, float* out, int64_t S, int64_t H, int64_t d, float base) {
    memcpy(out, in, S * H * d * sizeof(float));
    int64_t half_d = d / 2;
    for (int64_t pos = 0; pos < S; pos++) {
        for (int64_t h = 0; h < H; h++) {
            float* q = out + (pos * H + h) * d;
            for (int64_t i = 0; i < half_d; i++) {
                float angle = (float)pos / powf(base, (float)(2 * i) / (float)d);
                float c = cosf(angle), s = sinf(angle);
                float x0 = q[2 * i], x1 = q[2 * i + 1];
                q[2 * i]     = x0 * c - x1 * s;
                q[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

static int test_rope_graph_cpu(void) {
    fprintf(stderr, "\n=== RoPE Graph Integration (CPU) ===\n");

    int64_t total = T_S * T_H * T_d;
    int64_t shape[] = {T_S, T_H, T_d};

    tensor_t* tIn  = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 3, shape);
    CHECK(tIn && tOut, "tensor_create");

    srand(99);
    for (int64_t i = 0; i < total; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    float* ref = (float*)malloc(total * sizeof(float));
    rope_ref((const float*)tIn->data, ref, T_S, T_H, T_d, 10000.0f);

    /* Build native graph: INPUT -> ROPE -> OUTPUT */
    inference_graph_t* g = graph_create();
    CHECK(g, "graph_create");

    int tidIn  = graph_add_tensor(g, tIn);
    int tidOut = graph_add_tensor(g, tOut);
    CHECK(tidIn >= 0 && tidOut >= 0, "graph_add_tensor");

    rope_params_t params;
    memset(&params, 0, sizeof(params));
    params.seq_len    = T_S;
    params.head_dim   = T_d;
    params.num_heads  = T_H;
    params.base       = 10000.0f;
    params.layout     = ROPE_LAYOUT_INTERLEAVED;
    params.inv_freq   = NULL;
    params.batch_size = 1;
    params.pos_offset = 0;

    /* OP_ROPE: in-place on tidOut (input==output so no copy needed).
       We pass tidIn as input and tidOut as output; rope_f32 copies in->out
       then rotates. This exercises the non-in-place graph path too. */
    int input_tids[]  = {tidIn};
    int output_tids[] = {tidOut};
    int nid = graph_add_node(g, OP_ROPE,
                              1, input_tids,
                              1, output_tids,
                              0, NULL,
                              &params, sizeof(params));
    CHECK(nid >= 0, "graph_add_node OP_ROPE");

    int in_nid  = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tidIn}, 0, NULL, NULL, 0);
    graph_set_input(g, in_nid);
    int out_nid = graph_add_node(g, OP_OUTPUT, 1, (int[]){tidOut}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_output(g, out_nid);

    CHECK(graph_build(g) == 0, "graph_build");

    tensor_t* inputs[]  = {tIn};
    tensor_t* outputs[] = {tOut};
    int rc = graph_execute(g, inputs, outputs, false);
    CHECK(rc == 0, "graph_execute (CPU)");

    float diff = max_abs_diff((const float*)tOut->data, ref, total);
    fprintf(stderr, "Graph CPU vs ref: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-5f, "RoPE graph CPU mismatch");

    fprintf(stderr, "RoPE Graph CPU: PASS\n");
    /* graph_destroy frees all tensors added via graph_add_tensor */
    graph_destroy(g);
    free(ref);
    return 0;
}

#ifdef USE_CUDA
static int test_rope_graph_cuda(void) {
    fprintf(stderr, "\n=== RoPE Graph Integration (CUDA) ===\n");

    int64_t total = T_S * T_H * T_d;
    int64_t shape[] = {T_S, T_H, T_d};

    tensor_t* tIn  = tensor_create(DATA_TYPE_F32, 3, shape);
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 3, shape);
    CHECK(tIn && tOut, "tensor_create");

    srand(99);
    for (int64_t i = 0; i < total; i++) {
        ((float*)tIn->data)[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }

    float* ref = (float*)malloc(total * sizeof(float));
    rope_ref((const float*)tIn->data, ref, T_S, T_H, T_d, 10000.0f);

    inference_graph_t* g = graph_create();
    CHECK(g, "graph_create");

    int tidIn  = graph_add_tensor(g, tIn);
    int tidOut = graph_add_tensor(g, tOut);

    rope_params_t params;
    memset(&params, 0, sizeof(params));
    params.seq_len    = T_S;
    params.head_dim   = T_d;
    params.num_heads  = T_H;
    params.base       = 10000.0f;
    params.layout     = ROPE_LAYOUT_INTERLEAVED;
    params.inv_freq   = NULL;
    params.batch_size = 1;
    params.pos_offset = 0;

    int input_tids[]  = {tidIn};
    int output_tids[] = {tidOut};
    int nid = graph_add_node(g, OP_ROPE,
                              1, input_tids,
                              1, output_tids,
                              0, NULL,
                              &params, sizeof(params));
    CHECK(nid >= 0, "graph_add_node OP_ROPE");

    int in_nid  = graph_add_node(g, OP_INPUT, 0, NULL, 1, (int[]){tidIn}, 0, NULL, NULL, 0);
    graph_set_input(g, in_nid);
    int out_nid = graph_add_node(g, OP_OUTPUT, 1, (int[]){tidOut}, 0, NULL, 0, NULL, NULL, 0);
    graph_set_output(g, out_nid);

    CHECK(graph_build(g) == 0, "graph_build");

    /* Copy input to device; output device mem allocated by graph_execute */
    tensor_copy_to_device(tIn);
    tensor_t* inputs[]  = {tIn};
    tensor_t* outputs[] = {tOut};
    int rc = graph_execute(g, inputs, outputs, true);  /* use_cuda=true */
    CHECK(rc == 0, "graph_execute (CUDA)");
    tensor_copy_to_host(tOut);
    g_cuda.stream_synchronize(0);

    float diff = max_abs_diff((const float*)tOut->data, ref, total);
    fprintf(stderr, "Graph CUDA vs ref: max_diff=%.2e\n", diff);
    CHECK(diff < 1e-4f, "RoPE graph CUDA mismatch");

    fprintf(stderr, "RoPE Graph CUDA: PASS\n");
    /* graph_destroy frees all tensors added via graph_add_tensor */
    graph_destroy(g);
    free(ref);
    return 0;
}
#endif

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_rope_graph_cpu();
#ifdef USE_CUDA
    test_rope_graph_cuda();
#endif

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();

    fprintf(stderr, "\n=== RoPE Graph Integration Tests Done ===\n");
    return 0;
}
