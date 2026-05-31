/**
 * Minimal Gather test — verify embedding lookup works.
 * Creates a graph with: Gather(weight, input) → output
 * and checks if the output matches expected values.
 */
#include "graph.h"
#include "gather_int.h"
#include "platform.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

static float max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

int main(void) {
    platform_init();
    operator_init_all();

    fprintf(stderr, "=== Minimal Gather Test ===\n");

    /* Create a simple embedding table: 4 tokens, 3 dims */
    int64_t vocab = 4, dims = 3;
    int64_t weight_shape[] = {vocab, dims};
    tensor_t* tWeight = tensor_create(DATA_TYPE_F32, 2, weight_shape);
    float* w = (float*)tWeight->data;
    for (int i = 0; i < vocab * dims; i++) w[i] = (float)(i + 1);

    /* Input indices: [0, 2, 3] as float */
    int64_t num_ids = 3;
    int64_t idx_shape[] = {num_ids};
    tensor_t* tIdx = tensor_create(DATA_TYPE_F32, 1, idx_shape);
    float* idx = (float*)tIdx->data;
    idx[0] = 0.0f; idx[1] = 2.0f; idx[2] = 3.0f;

    /* Output: (3, 3) */
    int64_t out_shape[] = {num_ids, dims};
    tensor_t* tOut = tensor_create(DATA_TYPE_F32, 2, out_shape);

    /* Build graph: Gather(weight, idx) → out */
    inference_graph_t* g = graph_create();
    int w_tid = graph_add_tensor(g, tWeight);
    int idx_tid = graph_add_tensor(g, tIdx);
    int out_tid = graph_add_tensor(g, tOut);

    int input_tids[] = {w_tid, idx_tid};
    int output_tids[] = {out_tid};

    gather_params_t gp;
    memset(&gp, 0, sizeof(gp));
    gp.axis = 0;
    gp.block_size = dims;
    gp.outer_size = 1;
    gp.inner_size = vocab * dims;
    gp.num_indices = num_ids;

    graph_add_node(g, OP_GATHER, 2, input_tids, 1, output_tids,
                   0, NULL, &gp, sizeof(gp));
    graph_set_input(g, 0);
    graph_set_output(g, 1);
    graph_build(g);

    /* Run CPU */
    tensor_t* inputs[] = {tWeight};
    tensor_t* outputs[] = {tOut};
    int ret = graph_execute(g, inputs, outputs, 0);
    fprintf(stderr, "graph_execute returned: %d\n", ret);

    /* Expected: [[1,2,3], [7,8,9], [10,11,12]] */
    float* out = (float*)tOut->data;
    fprintf(stderr, "Output: [%.0f,%.0f,%.0f] [%.0f,%.0f,%.0f] [%.0f,%.0f,%.0f]\n",
            out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8]);

    float expected[] = {1,2,3, 7,8,9, 10,11,12};
    float diff = max_abs_diff(out, expected, 9);
    fprintf(stderr, "max_diff: %.6f\n", diff);
    if (diff < 1e-5f) {
        fprintf(stderr, "Gather Test: PASS\n");
    } else {
        fprintf(stderr, "Gather Test: FAIL\n");
    }

    graph_destroy(g);
    tensor_destroy(tWeight);
    tensor_destroy(tIdx);
    tensor_destroy(tOut);
    platform_finalize();
    return (diff < 1e-5f) ? 0 : 1;
}
