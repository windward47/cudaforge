#include "graph.h"
#include "conv_int.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_ops.h"
#endif

/* ============================================================
 * Graph construction
 * ============================================================ */
inference_graph_t* graph_create(void) {
    inference_graph_t* g = (inference_graph_t*)calloc(1, sizeof(inference_graph_t));
    if (!g) return NULL;
    g->num_nodes    = 0;
    g->num_tensors  = 0;
    g->num_inputs   = 0;
    g->num_outputs  = 0;
    g->nodes        = NULL;
    g->tensors      = NULL;
    g->topo_order   = NULL;
    g->input_node_ids  = NULL;
    g->output_node_ids = NULL;
    return g;
}

int graph_add_tensor(inference_graph_t* g, tensor_t* t) {
    if (!g || !t) return -1;
    int id = g->num_tensors;
    g->num_tensors++;
    g->tensors = (graph_tensor_t*)realloc(g->tensors,
                    (size_t)g->num_tensors * sizeof(graph_tensor_t));
    if (!g->tensors) return -1;

    g->tensors[id].tensor    = t;
    g->tensors[id].producer  = -1;
    g->tensors[id].consumer  = -1;
    return id;
}

int graph_add_node(inference_graph_t* g, op_type_t type,
                   int num_inputs, const int* input_tensors,
                   int num_outputs, const int* output_tensors,
                   int num_weights, tensor_t** weights,
                   void* params, size_t params_size) {
    if (!g) return -1;
    int id = g->num_nodes;
    g->num_nodes++;
    g->nodes = (graph_node_t*)realloc(g->nodes,
                  (size_t)g->num_nodes * sizeof(graph_node_t));
    if (!g->nodes) return -1;

    graph_node_t* n = &g->nodes[id];
    memset(n, 0, sizeof(graph_node_t));
    n->type        = type;
    n->num_inputs  = num_inputs;
    n->num_outputs = num_outputs;
    n->num_weights = num_weights;
    n->weights     = NULL;

    /* Copy weights array (caller retains ownership of the tensor_t* pointers) */
    if (num_weights > 0 && weights) {
        n->weights = (tensor_t**)malloc((size_t)num_weights * sizeof(tensor_t*));
        if (n->weights)
            memcpy(n->weights, weights, (size_t)num_weights * sizeof(tensor_t*));
    }

    if (params && params_size > 0) {
        n->params      = malloc(params_size);
        n->params_size = params_size;
        memcpy(n->params, params, params_size);
    }

    /* Copy tensor indices */
    if (num_inputs > 0) {
        n->input_tensors = (int*)malloc((size_t)num_inputs * sizeof(int));
        memcpy(n->input_tensors, input_tensors, (size_t)num_inputs * sizeof(int));
    }
    if (num_outputs > 0) {
        n->output_tensors = (int*)malloc((size_t)num_outputs * sizeof(int));
        memcpy(n->output_tensors, output_tensors, (size_t)num_outputs * sizeof(int));
    }

    /* Update tensor consumers/producers */
    for (int i = 0; i < num_inputs; i++) {
        if (input_tensors[i] >= 0 && input_tensors[i] < g->num_tensors)
            g->tensors[input_tensors[i]].consumer = id;
    }
    for (int i = 0; i < num_outputs; i++) {
        if (output_tensors[i] >= 0 && output_tensors[i] < g->num_tensors)
            g->tensors[output_tensors[i]].producer = id;
    }

    return id;
}

int graph_set_input(inference_graph_t* g, int node_id) {
    if (!g || node_id < 0 || node_id >= g->num_nodes) return -1;
    int id = g->num_inputs;
    g->num_inputs++;
    g->input_node_ids = (int*)realloc(g->input_node_ids,
                          (size_t)g->num_inputs * sizeof(int));
    if (!g->input_node_ids) return -1;
    g->input_node_ids[id] = node_id;
    return 0;
}

int graph_set_output(inference_graph_t* g, int node_id) {
    if (!g || node_id < 0 || node_id >= g->num_nodes) return -1;
    int id = g->num_outputs;
    g->num_outputs++;
    g->output_node_ids = (int*)realloc(g->output_node_ids,
                           (size_t)g->num_outputs * sizeof(int));
    if (!g->output_node_ids) return -1;
    g->output_node_ids[id] = node_id;
    return 0;
}

/* ============================================================
 * Topological sort (Kahn's algorithm)
 * ============================================================ */
int graph_build(inference_graph_t* g) {
    if (!g) return -1;

    /* Compute in-degree for each node */
    int* in_degree = (int*)calloc((size_t)g->num_nodes, sizeof(int));
    if (!in_degree) return -1;

    for (int i = 0; i < g->num_nodes; i++) {
        for (int j = 0; j < g->nodes[i].num_inputs; j++) {
            int tid = g->nodes[i].input_tensors[j];
            if (tid >= 0 && tid < g->num_tensors) {
                int prod = g->tensors[tid].producer;
                if (prod >= 0) in_degree[i]++;
            }
        }
    }

    /* Queue nodes with in_degree == 0 */
    int* queue = (int*)malloc((size_t)g->num_nodes * sizeof(int));
    int head = 0, tail = 0;
    int* order = (int*)malloc((size_t)g->num_nodes * sizeof(int));
    int ordered = 0;

    for (int i = 0; i < g->num_nodes; i++) {
        if (in_degree[i] == 0) queue[tail++] = i;
    }

    while (head < tail) {
        int node = queue[head++];
        order[ordered++] = node;

        /* Decrement in-degree of all nodes that consume this node's output tensors.
           Scan all nodes because a tensor can have multiple consumers (e.g. residual
           skip connections in ResNet). */
        for (int j = 0; j < g->nodes[node].num_outputs; j++) {
            int tid = g->nodes[node].output_tensors[j];
            if (tid >= 0 && tid < g->num_tensors) {
                for (int k = 0; k < g->num_nodes; k++) {
                    if (in_degree[k] <= 0) continue;
                    for (int l = 0; l < g->nodes[k].num_inputs; l++) {
                        if (g->nodes[k].input_tensors[l] == tid) {
                            in_degree[k]--;
                            if (in_degree[k] == 0) queue[tail++] = k;
                            break; /* node found; no need to check remaining inputs */
                        }
                    }
                }
            }
        }
    }

    free(in_degree);
    free(queue);

    if (ordered != g->num_nodes) {
        free(order);
        return -1; /* cycle detected */
    }

    if (g->topo_order) free(g->topo_order);
    g->topo_order = order;
    return 0;
}

void graph_destroy(inference_graph_t* g) {
    if (!g) return;
    for (int i = 0; i < g->num_nodes; i++) {
        free(g->nodes[i].input_tensors);
        free(g->nodes[i].output_tensors);
        free(g->nodes[i].weights);
        free(g->nodes[i].params);
    }
    free(g->nodes);
    for (int i = 0; i < g->num_tensors; i++) {
        tensor_destroy(g->tensors[i].tensor);
    }
    free(g->tensors);
    free(g->topo_order);
    free(g->input_node_ids);
    free(g->output_node_ids);
    free(g);
}

/* ============================================================
 * Op type → name mapping
 * ============================================================ */
static const char* op_name(op_type_t type) {
    switch (type) {
        case OP_RELU:       return "relu_f32";
        case OP_SIGMOID:    return "sigmoid_f32";
        case OP_GELU:       return "gelu_f32";
        case OP_MATMUL:     return "matmul_f32";
        case OP_CONV2D:     return "conv2d_f32";
        case OP_MAXPOOL2D:  return "maxpool2d_f32";
        case OP_AVGPOOL2D:  return "avgpool2d_f32";
        case OP_BATCHNORM:  return "batchnorm_f32";
        case OP_ADD:        return "add_f32";
        case OP_RESHAPE:    return "reshape_f32";
        case OP_GLOBALAVGPOOL: return "globalavgpool_f32";
        case OP_SOFTMAX:    return "softmax_f32";
        case OP_SILU:       return "silu_f32";
        case OP_MUL:        return "mul_f32";
        case OP_CONCAT:     return "concat_f32";
        case OP_RESIZE:     return "resize_f32";
        case OP_TRANSPOSE:  return "transpose_f32";
        default:            return NULL;
    }
}

/* ============================================================
 * Graph execution
 * ============================================================ */
int graph_execute(inference_graph_t* g, tensor_t* inputs[],
                  tensor_t* outputs[], bool use_cuda) {
    if (!g || !g->topo_order) return -1;

    int ret = 0;
    int* fused_skip = NULL;

    /* Copy graph inputs to input node tensors (skip self-copy when user passes graph tensor) */
    for (int i = 0; i < g->num_inputs; i++) {
        int node_id = g->input_node_ids[i];
        if (node_id < 0 || node_id >= g->num_nodes) continue;
        graph_node_t* n = &g->nodes[node_id];
        if (n->num_outputs > 0) {
            int tid = n->output_tensors[0];
            if (tid >= 0 && tid < g->num_tensors) {
                tensor_t* dst = g->tensors[tid].tensor;
                if (dst->data != inputs[i]->data) {
                    const data_type_info_t* info = data_type_get_info(dst->dtype);
                    size_t bytes = (size_t)dst->numel * info->size;
                    memcpy(dst->data, inputs[i]->data, bytes);
                }
            }
        }
    }

    /* ========================================================
     * Kernel fusion pre-pass: detect Conv/MatMul → Activation
     * patterns and fold the activation into the compute kernel.
     * Mark fused activation nodes so they are skipped below.
     * ======================================================== */
    int fused_count = 0;
    if (use_cuda && g->num_nodes > 1) {
        fused_skip = (int*)calloc((size_t)g->num_nodes, sizeof(int));
        if (fused_skip) {
            for (int oi = 0; oi < g->num_nodes - 1; oi++) {
                int nid = g->topo_order[oi];
                int next_nid = g->topo_order[oi + 1];
                graph_node_t* n = &g->nodes[nid];
                graph_node_t* next_n = &g->nodes[next_nid];

                if ((n->type == OP_CONV2D || n->type == OP_MATMUL)
                    && next_n->num_inputs == 1
                    && n->num_outputs == 1
                    && next_n->input_tensors[0] == n->output_tensors[0]
                    && (next_n->type == OP_RELU
                        || next_n->type == OP_SIGMOID
                        || next_n->type == OP_GELU)) {

                    if (n->type == OP_CONV2D && n->params
                        && n->params_size >= sizeof(conv_params_t)) {
                        conv_params_t* cp = (conv_params_t*)n->params;
                        cp->fuse_activation = (int64_t)(next_n->type + 1);
                        fused_skip[next_nid] = 1;
                        fused_count++;
                    }
                }
            }
        }
    }

    /* Execute nodes in topological order */
    for (int oi = 0; oi < g->num_nodes; oi++) {
        int node_id = g->topo_order[oi];
        graph_node_t* n = &g->nodes[node_id];

        /* Skip fused activation nodes */
        if (fused_skip && fused_skip[node_id]) {
            continue;
        }

        /* Input: copy tensor data from earlier node outputs */
        if (n->type == OP_INPUT) continue;
        if (n->type == OP_OUTPUT) continue;

        /* Build op name */
        char name_buf[64];
        const char* base = op_name(n->type);
        if (!base) { ret = -1; goto cleanup; }
        if (use_cuda) {
            snprintf(name_buf, sizeof(name_buf), "%s_cuda", base);
        } else {
            snprintf(name_buf, sizeof(name_buf), "%s", base);
        }

        const operator_registry_t* op = operator_find(name_buf);
        if (!op) {
            /* Fall back to CPU version if CUDA variant not found */
            if (use_cuda) {
                op = operator_find(base);
            }
        }
        if (!op) {
            ret = -2; goto cleanup;
        }

        /* Input slots needed: tensor inputs + weights + extra metadata slots */
        int total_inputs = n->num_inputs + n->num_weights;
        int max_input_idx = total_inputs > 0 ? total_inputs - 1 : 0;
        if (n->type == OP_RELU || n->type == OP_SIGMOID || n->type == OP_GELU
            || n->type == OP_SILU)
            if (max_input_idx < 1) max_input_idx = 1;
        if (n->type == OP_CONV2D || n->type == OP_MATMUL)
            if (max_input_idx < 2) max_input_idx = 2;  /* reserve for optional bias */
        if (n->type == OP_BATCHNORM)
            if (max_input_idx < 5) max_input_idx = 5;
        int input_slots = max_input_idx + 1;

        /* Stack buffer for common case (avoids per-node malloc/free) */
        const void* local_buf[32];
        const void** op_inputs = local_buf;
        void** op_outputs = NULL;
        int total_slots = input_slots + n->num_outputs;
        if (total_slots > 32) {
            op_inputs = (const void**)malloc((size_t)total_slots * sizeof(void*));
        }
        op_outputs = (void**)&op_inputs[input_slots];

        for (int i = 0; i < n->num_inputs; i++) {
            int tid = n->input_tensors[i];
            if (tid >= 0 && tid < g->num_tensors)
                op_inputs[i] = g->tensors[tid].tensor->data;
            else
                op_inputs[i] = NULL;
        }
        for (int w = 0; w < n->num_weights; w++) {
            if (n->weights[w])
                op_inputs[n->num_inputs + w] = n->weights[w]->data;
            else
                op_inputs[n->num_inputs + w] = NULL;
        }
        /* NULL-fill any remaining input slots (e.g. reserved bias slot for Conv/Gemm) */
        for (int i = n->num_inputs + n->num_weights; i < input_slots; i++) {
            op_inputs[i] = NULL;
        }

        /* Effective output tensor IDs: for fused Conv/MatMul nodes,
         * redirect to the activation's output so downstream consumers
         * see the post-activation result directly. */
        int effective_output_tids[8] = {0}; /* max outputs per node is small */
        if (fused_skip && oi + 1 < g->num_nodes) {
            int next_nid = g->topo_order[oi + 1];
            if (fused_skip[next_nid]) {
                graph_node_t* next_n = &g->nodes[next_nid];
                if (next_n->input_tensors[0] == n->output_tensors[0]
                    && (next_n->type == OP_RELU || next_n->type == OP_SIGMOID
                        || next_n->type == OP_GELU)) {
                    effective_output_tids[0] = next_n->output_tensors[0];
                }
            }
        }

        /* Pack outputs */
        for (int i = 0; i < n->num_outputs; i++) {
            int tid = effective_output_tids[i] ? effective_output_tids[i]
                                               : n->output_tensors[i];
            if (tid >= 0 && tid < g->num_tensors)
                op_outputs[i] = g->tensors[tid].tensor->data;
            else
                op_outputs[i] = NULL;
        }

        /* Handle ops that need extra metadata via inputs[] slot */
        /* relu/sigmoid/gelu: need numel as inputs[1] */
        if (n->type == OP_RELU || n->type == OP_SIGMOID || n->type == OP_GELU
            || n->type == OP_SILU) {
            int tid = n->input_tensors[0];
            if (tid >= 0 && tid < g->num_tensors) {
                op_inputs[1] = &g->tensors[tid].tensor->numel;
            }
        }
        /* batchnorm: need hw as inputs[5] */
        if (n->type == OP_BATCHNORM) {
            int tid = n->input_tensors[0];
            if (tid >= 0 && tid < g->num_tensors) {
                op_inputs[5] = &g->tensors[tid].tensor->numel;
            }
        }

        /* Copy to device if CUDA and swap pointers */
        if (use_cuda) {
            for (int i = 0; i < n->num_inputs; i++) {
                int tid = n->input_tensors[i];
                if (tid >= 0) {
                    tensor_copy_to_device(g->tensors[tid].tensor);
                    if (g->tensors[tid].tensor->data_device)
                        op_inputs[i] = g->tensors[tid].tensor->data_device;
                }
            }
            for (int w = 0; w < n->num_weights; w++) {
                if (n->weights[w]) {
                    tensor_copy_to_device(n->weights[w]);
                    if (n->weights[w]->data_device)
                        op_inputs[n->num_inputs + w] = n->weights[w]->data_device;
                }
            }
            for (int i = 0; i < n->num_outputs; i++) {
                int tid = effective_output_tids[i] ? effective_output_tids[i]
                                                   : n->output_tensors[i];
                if (tid >= 0) {
                    tensor_t* ot = g->tensors[tid].tensor;
                    if (!ot->data_device) {
                        const data_type_info_t* info = data_type_get_info(ot->dtype);
                        size_t bytes = (size_t)ot->numel * info->size;
                        ot->data_device = g_cuda.device_alloc(bytes);
                    }
                    if (ot->data_device)
                        op_outputs[i] = ot->data_device;
                }
            }
        }

        /* Dispatch */
        stream_t stream = {0};
        ret = op->func(op_inputs, op_outputs,
                       (const operator_params_t*)n->params,
                       use_cuda ? &stream : NULL);
        if (ret != 0) {
            if (op_inputs != local_buf) free((void*)op_inputs);
            goto cleanup;
        }

        /* Copy back to host if CUDA — must happen BEFORE any host-side verify */
        if (use_cuda) {
            for (int i = 0; i < n->num_outputs; i++) {
                int tid = effective_output_tids[i] ? effective_output_tids[i]
                                                   : n->output_tensors[i];
                if (tid >= 0) tensor_copy_to_host(g->tensors[tid].tensor);
            }
            g_cuda.stream_synchronize(0);
        }

        if (op_inputs != local_buf) free((void*)op_inputs);
    }

    /* Copy final outputs (skip self-copy when user passes graph tensors as outputs) */
    for (int i = 0; i < g->num_outputs; i++) {
        int node_id = g->output_node_ids[i];
        if (node_id < 0 || node_id >= g->num_nodes) continue;
        graph_node_t* n = &g->nodes[node_id];
        if (n->type == OP_OUTPUT && n->num_inputs > 0) {
            int tid = n->input_tensors[0];
            if (tid >= 0 && tid < g->num_tensors && i < g->num_outputs) {
                tensor_t* src = g->tensors[tid].tensor;
                if (outputs[i]->data != src->data) {
                    const data_type_info_t* info = data_type_get_info(src->dtype);
                    size_t bytes = (size_t)src->numel * info->size;
                    memcpy(outputs[i]->data, src->data, bytes);
                }
            }
        }
    }

    ret = 0;

cleanup:
    free(fused_skip);
    return ret;
}
