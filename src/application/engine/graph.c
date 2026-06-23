#include "graph.h"
#include "conv_int.h"
#include "mha_fused_int.h"
#include "mha_decode_int.h"
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
    g->kv_cache_K_tid  = -1;
    g->kv_cache_V_tid  = -1;
    return g;
}

int graph_add_tensor(inference_graph_t* g, tensor_t* t) {
    if (!g || !t) return -1;
    int id = g->num_tensors;
    graph_tensor_t* p = (graph_tensor_t*)realloc(g->tensors,
                    (size_t)(g->num_tensors + 1) * sizeof(graph_tensor_t));
    if (!p) return -1;
    g->tensors = p;
    g->num_tensors++;

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
    graph_node_t* p = (graph_node_t*)realloc(g->nodes,
                  (size_t)(g->num_nodes + 1) * sizeof(graph_node_t));
    if (!p) return -1;
    g->nodes = p;
    g->num_nodes++;

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
        if (!n->weights) return -1;
        memcpy(n->weights, weights, (size_t)num_weights * sizeof(tensor_t*));
    }

    if (params && params_size > 0) {
        n->params = malloc(params_size);
        if (!n->params) return -1;
        n->params_size = params_size;
        memcpy(n->params, params, params_size);
    }

    /* Copy tensor indices */
    if (num_inputs > 0) {
        n->input_tensors = (int*)malloc((size_t)num_inputs * sizeof(int));
        if (!n->input_tensors) return -1;
        memcpy(n->input_tensors, input_tensors, (size_t)num_inputs * sizeof(int));
    }
    if (num_outputs > 0) {
        n->output_tensors = (int*)malloc((size_t)num_outputs * sizeof(int));
        if (!n->output_tensors) return -1;
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
    int* p = (int*)realloc(g->input_node_ids,
                          (size_t)(g->num_inputs + 1) * sizeof(int));
    if (!p) return -1;
    g->input_node_ids = p;
    g->num_inputs++;
    g->input_node_ids[id] = node_id;
    return 0;
}

int graph_set_output(inference_graph_t* g, int node_id) {
    if (!g || node_id < 0 || node_id >= g->num_nodes) return -1;
    int id = g->num_outputs;
    int* p = (int*)realloc(g->output_node_ids,
                           (size_t)(g->num_outputs + 1) * sizeof(int));
    if (!p) return -1;
    g->output_node_ids = p;
    g->num_outputs++;
    g->output_node_ids[id] = node_id;
    return 0;
}

void graph_set_kv_cache(inference_graph_t* g, int K_tensor_id, int V_tensor_id) {
    if (!g) return;
    g->kv_cache_K_tid = K_tensor_id;
    g->kv_cache_V_tid = V_tensor_id;
}

void graph_update_cache_len(inference_graph_t* g, int64_t new_cache_len) {
    if (!g) return;
    for (int i = 0; i < g->num_nodes; i++) {
        graph_node_t* n = &g->nodes[i];
        if (n->type == OP_MHA_DECODE && n->params
            && n->params_size >= sizeof(mha_decode_params_t)) {
            mha_decode_params_t* p = (mha_decode_params_t*)n->params;
            p->cache_len = new_cache_len;
        }
    }
}

void graph_set_permanent_fusion(inference_graph_t* g, int enable) {
    if (g) g->permanent_fusion = enable;
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
#ifdef USE_CUDA
    if (g->cuda_graph_exec) {
        cudaGraphExecDestroy((cudaGraphExec_t)g->cuda_graph_exec);
        g->cuda_graph_exec = NULL;
    }
    if (g->cuda_graph) {
        cudaGraphDestroy((cudaGraph_t)g->cuda_graph);
        g->cuda_graph = NULL;
    }
#endif
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
        case OP_SUB:        return "sub_f32";
        case OP_DIV:        return "div_f32";
        case OP_SLICE:      return "slice_f32";
        case OP_SPLIT:              return "split_f32";
        case OP_LAYERNORM:          return "layernorm_f32";
        case OP_GATHER:             return "gather_f32";
        case OP_SQUEEZE_UNSQUEEZE:  return "squeeze_unsqueeze_f32";
        case OP_EXP:                return "exp_f32";
        case OP_REDUCE:             return "reduce_f32";
        case OP_CAST:               return "cast_f32";
        case OP_ARGMAX:             return "argmax_f32";
        case OP_MHA_FUSED:          return "mha_fused_f32";
        case OP_MHA_DECODE:         return "mha_decode_f32";
        case OP_CAUSAL_MASK:        return "causal_mask_f32";
        case OP_ROPE:               return "rope_f32";
        case OP_PAD:                return "pad_f32";
        case OP_CLIP:               return "clip_f32";
        case OP_WHERE:              return "where_f32";
        case OP_TANH:               return "tanh_f32";
        default:                    return NULL;
    }
}

/* ============================================================
 * MHA fusion helpers
 * ============================================================ */

/* Find the unique consumer node of a tensor. Returns -1 if 0 or >1 consumers. */
static int mha_unique_consumer(inference_graph_t* g, int tid, const int* skip) {
    int found = -1;
    for (int i = 0; i < g->num_nodes; i++) {
        if (skip && skip[i]) continue;
        graph_node_t* n = &g->nodes[i];
        for (int j = 0; j < n->num_inputs; j++) {
            if (n->input_tensors[j] == tid) {
                if (found >= 0) return -1;
                found = i;
                break;
            }
        }
    }
    return found;
}

/* Count consumers of a tensor across all non-skipped nodes */
static int mha_count_consumers(inference_graph_t* g, int tid, const int* skip) {
    int count = 0;
    for (int i = 0; i < g->num_nodes; i++) {
        if (skip && skip[i]) continue;
        graph_node_t* n = &g->nodes[i];
        for (int j = 0; j < n->num_inputs; j++) {
            if (n->input_tensors[j] == tid) { count++; break; }
        }
    }
    return count;
}

/* ============================================================
 * MHA node restore info — for non-destructive fusion
 * ============================================================ */
typedef struct {
    int        node_id;  /* -1 if no restore needed */
    op_type_t  saved_type;
    int        saved_num_inputs;
    int*       saved_input_tensors;
    int        saved_num_outputs;
    int*       saved_output_tensors;
    int        saved_num_weights;
    tensor_t** saved_weights;
    void*      saved_params;
    size_t     saved_params_size;
} mha_restore_info_t;

/* ============================================================
 * MHA subgraph fusion detection
 *
 * Detects the standard BERT self-attention pattern:
 *   X → 3×[MatMul+Add → Reshape → Transpose] → Q, K, V
 *   Q·K^T → Mul(scale) → Softmax → ×V
 *   Transpose → Reshape → MatMul(W_O)+Add(b_O) → Add(residual)
 *
 * Converts the first QKV MatMul node into OP_MHA_FUSED and marks
 * the remaining 20+ nodes as skipped.  Saves the original node
 * state in *restore so the caller can undo the conversion after
 * execution (non-destructive fusion).
 * ============================================================ */
static int detect_and_fuse_mha(inference_graph_t* g, int* fused_skip,
                               mha_restore_info_t* restore) {
    if (g->num_nodes < 20) return 0;
    if (restore) restore->node_id = -1;

    /* Phase 1: scan for a tensor consumed by exactly 3 MatMul nodes */
    for (int tid = 0; tid < g->num_tensors; tid++) {
        int mm_nodes[3];
        int mm_count = 0;
        for (int i = 0; i < g->num_nodes && mm_count <= 3; i++) {
            if (fused_skip[i]) continue;
            graph_node_t* n = &g->nodes[i];
            if (n->type != OP_MATMUL || n->num_inputs < 1 || n->num_outputs < 1) continue;
            for (int j = 0; j < n->num_inputs; j++) {
                if (n->input_tensors[j] == tid) {
                    if (mm_count < 3) mm_nodes[mm_count] = i;
                    mm_count++;
                    break;
                }
            }
        }
        if (mm_count != 3) continue;

        int anchor_tid = tid;

        /* Phase 2: trace each chain: MatMul → (Add) → Reshape → Transpose */
        int chain_add[3]       = {-1, -1, -1};
        int chain_reshape[3]   = {-1, -1, -1};
        int chain_transpose[3] = {-1, -1, -1};
        int chain_out_tid[3]   = {-1, -1, -1};
        tensor_t* chain_w[3] = {NULL, NULL, NULL};
        tensor_t* chain_b[3] = {NULL, NULL, NULL};
        bool chains_ok = true;

        for (int c = 0; c < 3 && chains_ok; c++) {
            int mm_nid = mm_nodes[c];
            graph_node_t* mm = &g->nodes[mm_nid];

            /* Extract weight tensor (second input to MatMul) */
            if (mm->num_weights >= 1 && mm->weights && mm->weights[0])
                chain_w[c] = mm->weights[0];

            int cur_tid = mm->output_tensors[0];

            /* Optional Add (bias) */
            int add_nid = mha_unique_consumer(g, cur_tid, fused_skip);
            if (add_nid >= 0 && g->nodes[add_nid].type == OP_ADD
                && g->nodes[add_nid].num_inputs >= 2
                && g->nodes[add_nid].input_tensors[0] == cur_tid) {
                chain_add[c] = add_nid;
                if (g->nodes[add_nid].num_weights >= 1 && g->nodes[add_nid].weights)
                    chain_b[c] = g->nodes[add_nid].weights[0];
                cur_tid = g->nodes[add_nid].output_tensors[0];
            }

            /* Reshape */
            int rs_nid = mha_unique_consumer(g, cur_tid, fused_skip);
            if (rs_nid < 0 || g->nodes[rs_nid].type != OP_RESHAPE) {
                chains_ok = false; break;
            }
            chain_reshape[c] = rs_nid;
            cur_tid = g->nodes[rs_nid].output_tensors[0];

            /* Transpose */
            int tr_nid = mha_unique_consumer(g, cur_tid, fused_skip);
            if (tr_nid < 0 || g->nodes[tr_nid].type != OP_TRANSPOSE) {
                chains_ok = false; break;
            }
            chain_transpose[c] = tr_nid;
            chain_out_tid[c] = g->nodes[tr_nid].output_tensors[0];
        }
        if (!chains_ok) continue;

        /* Phase 3: identify Q, K, V.
         * Q = chain whose output is input[0] of a MatMul (attention scores)
         * K = chain whose output is input[1] of the SAME MatMul
         * V = the remaining chain */
        int q_idx = -1, k_idx = -1, v_idx = -1;
        int attn_mm_nid = -1;

        for (int ci = 0; ci < 3 && attn_mm_nid < 0; ci++) {
            int out_tid = chain_out_tid[ci];
            for (int i = 0; i < g->num_nodes; i++) {
                if (fused_skip[i]) continue;
                graph_node_t* n = &g->nodes[i];
                if (n->type != OP_MATMUL || n->num_inputs < 2) continue;
                if (n->input_tensors[0] == out_tid) {
                    /* Found a MatMul consuming this chain as first input → Q */
                    q_idx = ci;
                    attn_mm_nid = i;
                    break;
                }
            }
        }
        if (q_idx < 0 || attn_mm_nid < 0) continue;

        /* The second input of attn_mm is K */
        graph_node_t* attn_mm = &g->nodes[attn_mm_nid];
        int k_tid = attn_mm->input_tensors[1];
        for (int ci = 0; ci < 3; ci++) {
            if (ci == q_idx) continue;
            if (chain_out_tid[ci] == k_tid) {
                k_idx = ci;
                break;
            }
        }
        if (k_idx < 0) continue;

        /* The remaining chain is V */
        for (int ci = 0; ci < 3; ci++) {
            if (ci != q_idx && ci != k_idx) { v_idx = ci; break; }
        }

        /* Phase 4: trace attention computation */
        int attn_out_tid = attn_mm->output_tensors[0];

        /* Mul(scale) */
        int mul_nid = mha_unique_consumer(g, attn_out_tid, fused_skip);
        if (mul_nid < 0 || g->nodes[mul_nid].type != OP_MUL) continue;
        int mul_out_tid = g->nodes[mul_nid].output_tensors[0];

        /* Softmax */
        int softmax_nid = mha_unique_consumer(g, mul_out_tid, fused_skip);
        if (softmax_nid < 0 || g->nodes[softmax_nid].type != OP_SOFTMAX) continue;
        int probs_tid = g->nodes[softmax_nid].output_tensors[0];

        /* MatMul(probs, V) */
        int v_mm_nid = mha_unique_consumer(g, probs_tid, fused_skip);
        if (v_mm_nid < 0 || g->nodes[v_mm_nid].type != OP_MATMUL
            || g->nodes[v_mm_nid].num_inputs < 2) continue;
        /* Verify V input: either input[0] or input[1] */
        graph_node_t* v_mm = &g->nodes[v_mm_nid];
        int v_tid = chain_out_tid[v_idx];
        if (v_mm->input_tensors[0] != probs_tid
            || (v_mm->input_tensors[1] != v_tid)) continue;

        /* Phase 5: trace merge + output projection + residual */
        int v_out_tid = v_mm->output_tensors[0];

        /* Transpose */
        int merge_tr_nid = mha_unique_consumer(g, v_out_tid, fused_skip);
        if (merge_tr_nid < 0 || g->nodes[merge_tr_nid].type != OP_TRANSPOSE) continue;
        int merge_tr_out = g->nodes[merge_tr_nid].output_tensors[0];

        /* Reshape */
        int merge_rs_nid = mha_unique_consumer(g, merge_tr_out, fused_skip);
        if (merge_rs_nid < 0 || g->nodes[merge_rs_nid].type != OP_RESHAPE) continue;
        int merged_tid = g->nodes[merge_rs_nid].output_tensors[0];

        /* MatMul(output projection) */
        int out_mm_nid = mha_unique_consumer(g, merged_tid, fused_skip);
        if (out_mm_nid < 0 || g->nodes[out_mm_nid].type != OP_MATMUL) continue;
        graph_node_t* out_mm = &g->nodes[out_mm_nid];
        tensor_t* wo_weight = (out_mm->num_weights >= 1 && out_mm->weights)
                              ? out_mm->weights[0] : NULL;
        int out_mm_out = out_mm->output_tensors[0];

        /* Optional Add(b_O) */
        int out_add_nid = -1;
        tensor_t* bo_weight = NULL;
        int proj_tid = out_mm_out;
        {
            int candidate = mha_unique_consumer(g, out_mm_out, fused_skip);
            if (candidate >= 0 && g->nodes[candidate].type == OP_ADD
                && g->nodes[candidate].num_inputs >= 2
                && g->nodes[candidate].input_tensors[0] == out_mm_out) {
                out_add_nid = candidate;
                if (g->nodes[candidate].num_weights >= 1 && g->nodes[candidate].weights)
                    bo_weight = g->nodes[candidate].weights[0];
                proj_tid = g->nodes[candidate].output_tensors[0];
            }
        }

        /* Optional residual Add */
        int res_add_nid = -1;
        int residual_tid = -1;
        int final_out_tid = proj_tid;
        {
            int candidate = mha_unique_consumer(g, proj_tid, fused_skip);
            if (candidate >= 0 && g->nodes[candidate].type == OP_ADD
                && g->nodes[candidate].num_inputs >= 2) {
                graph_node_t* ra = &g->nodes[candidate];
                if (ra->input_tensors[0] == proj_tid) {
                    res_add_nid = candidate;
                    residual_tid = ra->input_tensors[1];
                    final_out_tid = ra->output_tensors[0];
                }
            }
        }

        /* Phase 6: extract dimensions from tensor shapes */
        tensor_t* anchor_tensor = g->tensors[anchor_tid].tensor;
        if (!anchor_tensor || anchor_tensor->ndim < 3) continue;
        int64_t B = anchor_tensor->shape[0];
        int64_t S = anchor_tensor->shape[1];
        int64_t D = anchor_tensor->shape[2];

        /* Get H and d from Q Transpose output shape: (B, H, S, d) */
        int q_out_tid = chain_out_tid[q_idx];
        tensor_t* q_tensor = g->tensors[q_out_tid].tensor;
        if (!q_tensor || q_tensor->ndim < 4) continue;
        int64_t H = q_tensor->shape[1];
        int64_t d = q_tensor->shape[3];

        /* Phase 7: collect all nodes to skip */
        int skip_list[32];
        int skip_count = 0;
        for (int c = 0; c < 3; c++) {
            if (mm_nodes[c] != mm_nodes[q_idx])  /* skip the other 2 MatMuls */
                skip_list[skip_count++] = mm_nodes[c];
            if (chain_add[c] >= 0)
                skip_list[skip_count++] = chain_add[c];
            skip_list[skip_count++] = chain_reshape[c];
            skip_list[skip_count++] = chain_transpose[c];
        }
        skip_list[skip_count++] = attn_mm_nid;
        skip_list[skip_count++] = mul_nid;
        skip_list[skip_count++] = softmax_nid;
        skip_list[skip_count++] = v_mm_nid;
        skip_list[skip_count++] = merge_tr_nid;
        skip_list[skip_count++] = merge_rs_nid;
        skip_list[skip_count++] = out_mm_nid;
        if (out_add_nid >= 0) skip_list[skip_count++] = out_add_nid;
        if (res_add_nid >= 0) skip_list[skip_count++] = res_add_nid;

        /* Verify skip count is reasonable (≥ 18 for full BERT pattern) */
        if (skip_count < 18 || skip_count > 32) continue;

        /* Phase 8: save original state, then convert Q MatMul into MHA_Fused */
        int target_nid = mm_nodes[q_idx];
        graph_node_t* target = &g->nodes[target_nid];

        if (restore) {
            restore->node_id            = target_nid;
            restore->saved_type         = target->type;
            restore->saved_num_inputs   = target->num_inputs;
            restore->saved_input_tensors = target->input_tensors;
            restore->saved_num_outputs  = target->num_outputs;
            restore->saved_output_tensors = target->output_tensors;
            restore->saved_num_weights  = target->num_weights;
            restore->saved_weights      = target->weights;
            restore->saved_params       = target->params;
            restore->saved_params_size  = target->params_size;
        }

        /* Free old arrays */
        free(target->input_tensors);
        free(target->output_tensors);
        free(target->weights);
        free(target->params);

        /* Build new input_tensors: [anchor_tid, residual_tid] */
        target->num_inputs = 2;
        target->input_tensors = (int*)malloc(2 * sizeof(int));
        if (!target->input_tensors) return 0;
        target->input_tensors[0] = anchor_tid;
        target->input_tensors[1] = residual_tid;

        /* Build new output_tensors: [final_out_tid] */
        target->num_outputs = 1;
        target->output_tensors = (int*)malloc(1 * sizeof(int));
        if (!target->output_tensors) return 0;
        target->output_tensors[0] = final_out_tid;

        /* Build weights: [W_Q, b_Q, W_K, b_K, W_V, b_V, W_O, b_O] */
        target->num_weights = 8;
        target->weights = (tensor_t**)malloc(8 * sizeof(tensor_t*));
        if (!target->weights) return 0;
        target->weights[0] = chain_w[q_idx];
        target->weights[1] = chain_b[q_idx];
        target->weights[2] = chain_w[k_idx];
        target->weights[3] = chain_b[k_idx];
        target->weights[4] = chain_w[v_idx];
        target->weights[5] = chain_b[v_idx];
        target->weights[6] = wo_weight;
        target->weights[7] = bo_weight;

        /* Build params */
        mha_fused_params_t* fp = (mha_fused_params_t*)malloc(sizeof(mha_fused_params_t));
        if (!fp) return 0;
        fp->batch_size   = B;
        fp->seq_len      = S;
        fp->hidden_size  = D;
        fp->num_heads    = H;
        fp->num_kv_heads = H;   /* default: no GQA (H_kv = H_q) */
        fp->head_dim     = d;
        fp->scale        = 1.0f / sqrtf((float)d);
        fp->has_residual = (residual_tid >= 0);
        fp->causal       = 0;   /* default: bidirectional (BERT-style) */
        target->params      = fp;
        target->params_size = sizeof(mha_fused_params_t);
        target->type        = OP_MHA_FUSED;

        /* Update tensor producer: the final output tensor is now produced by target */
        if (final_out_tid >= 0 && final_out_tid < g->num_tensors)
            g->tensors[final_out_tid].producer = target_nid;

        /* Phase 9: mark all other nodes as skipped */
        for (int s = 0; s < skip_count; s++) {
            fused_skip[skip_list[s]] = 1;
        }

        return 1;  /* fused one pattern */
    }

    return 0;
}

/* ============================================================
 * Graph execution
 * ============================================================ */
int graph_execute(inference_graph_t* g, tensor_t* inputs[],
                  tensor_t* outputs[], bool use_cuda) {
    if (!g || !g->topo_order) return -1;

    int ret = 0;
    int* fused_skip = NULL;
    mha_restore_info_t mha_restore;
    memset(&mha_restore, 0, sizeof(mha_restore));
    mha_restore.node_id = -1;

#ifdef USE_CUDA
    /* ---- CUDA Graph fast path: replay cached graph if shapes match ---- */
    if (use_cuda && g->cuda_graph_state == 2 && g->cuda_graph_exec) {
        int64_t cur_B = 0, cur_S = 0;
        if (g->num_inputs > 0 && inputs[0] && inputs[0]->ndim >= 2) {
            cur_B = inputs[0]->shape[0];
            cur_S = inputs[0]->shape[1];
        }
        if (cur_B == g->graph_cache_B && cur_S == g->graph_cache_S) {
            /* Copy inputs to device */
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
                            memcpy(dst->data, inputs[i]->data, (size_t)dst->numel * info->size);
                        }
                        tensor_copy_to_device(dst);
                    }
                }
            }
            /* Use dedicated graph stream for replay */
            cudaStream_t graph_stream = (cudaStream_t)g->cuda_graph;
            cudaGraphLaunch((cudaGraphExec_t)g->cuda_graph_exec, graph_stream);
            cudaStreamSynchronize(graph_stream);
            /* Copy outputs back */
            for (int i = 0; i < g->num_outputs; i++) {
                int node_id = g->output_node_ids[i];
                if (node_id < 0 || node_id >= g->num_nodes) continue;
                graph_node_t* n = &g->nodes[node_id];
                if (n->type == OP_OUTPUT && n->num_inputs > 0) {
                    int tid = n->input_tensors[0];
                    if (tid >= 0 && tid < g->num_tensors) {
                        tensor_t* src = g->tensors[tid].tensor;
                        tensor_copy_to_host(src);
                        if (outputs[i]->data != src->data) {
                            const data_type_info_t* info = data_type_get_info(src->dtype);
                            memcpy(outputs[i]->data, src->data, (size_t)src->numel * info->size);
                        }
                    }
                }
            }
            return 0;
        }
        /* Shape changed — invalidate */
        cudaGraphExecDestroy((cudaGraphExec_t)g->cuda_graph_exec);
        g->cuda_graph_exec = NULL;
        cudaGraphDestroy((cudaGraph_t)g->cuda_graph);
        g->cuda_graph = NULL;
        g->cuda_graph_state = 0;
    }

    /* CUDA Graph capture is disabled because all kernel launches use stream 0
     * (legacy stream), which cannot be captured. To enable, all operator
     * dispatches would need to accept and use a common non-default stream.
     * For now, skip capture entirely. */
    int cuda_capturing = 0;
    (void)cuda_capturing;
#endif

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
                        || next_n->type == OP_GELU
                        || next_n->type == OP_SILU
                        || next_n->type == OP_EXP)) {

                    /* Do not fuse if the compute node's output has other
                     * consumers (e.g. SiLU = Conv→Sigmoid→Mul: the Conv
                     * output feeds both Sigmoid and Mul). */
                    int out_tid = n->output_tensors[0];
                    int consumers = 0;
                    for (int j = 0; j < g->num_nodes; j++) {
                        graph_node_t* other = &g->nodes[j];
                        for (int k = 0; k < other->num_inputs && consumers <= 1; k++) {
                            if (other->input_tensors[k] == out_tid) consumers++;
                        }
                    }
                    if (consumers > 1) continue;

                    if (n->type == OP_CONV2D && n->params
                        && n->params_size >= sizeof(conv_params_t)) {
                        conv_params_t* cp = (conv_params_t*)n->params;
                        cp->fuse_activation = (int64_t)(next_n->type + 1);
                        fused_skip[next_nid] = 1;
                        fused_count++;
                    }
                }
            }

            /* MHA fusion: detect full BERT self-attention subgraph */
            /* Only save restore info for non-permanent fusion */
            if (!g->permanent_fusion) {
                fused_count += detect_and_fuse_mha(g, fused_skip, &mha_restore);
            } else {
                fused_count += detect_and_fuse_mha(g, fused_skip, NULL);
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

        /* Build op name — try dtype-aware name first, fall back to f32 */
        char name_buf[64];
        const char* base = op_name(n->type);
        if (!base) { ret = -1; goto cleanup; }

        /* Check output tensor dtype for FP16 dispatch */
        int output_is_f16 = 0;
        if (n->num_outputs > 0 && n->output_tensors && n->output_tensors[0] >= 0) {
            int tid = n->output_tensors[0];
            if (tid < g->num_tensors && g->tensors[tid].tensor) {
                if (g->tensors[tid].tensor->dtype == DATA_TYPE_F16) output_is_f16 = 1;
            }
        }

        const operator_registry_t* op = NULL;
        if (output_is_f16 && n->type != OP_CAST && n->type != OP_RESHAPE
            && n->type != OP_TRANSPOSE && n->type != OP_SLICE
            && n->type != OP_SPLIT && n->type != OP_SQUEEZE_UNSQUEEZE) {
            /* Try FP16 variant first */
            char base_f16[64];
            /* Replace "_f32" suffix with "_f16" */
            size_t blen = strlen(base);
            if (blen > 3 && strcmp(base + blen - 3, "f32") == 0) {
                snprintf(base_f16, sizeof(base_f16), "%.*sf16", (int)(blen - 3), base);
            } else {
                snprintf(base_f16, sizeof(base_f16), "%s", base);
            }
            if (use_cuda) {
                snprintf(name_buf, sizeof(name_buf), "%s_cuda", base_f16);
            } else {
                snprintf(name_buf, sizeof(name_buf), "%s", base_f16);
            }
            op = operator_find(name_buf);
        }

        if (!op) {
            /* Fall back to f32 variant */
            if (use_cuda) {
                snprintf(name_buf, sizeof(name_buf), "%s_cuda", base);
            } else {
                snprintf(name_buf, sizeof(name_buf), "%s", base);
            }
            op = operator_find(name_buf);
        }
        if (!op) {
            /* Fall back to CPU version if CUDA variant not found */
            if (use_cuda) {
                op = operator_find(base);
            }
        }
        if (!op) { ret = -2; goto cleanup; }

        /* Input slots needed: tensor inputs + weights + extra metadata slots */
        int total_inputs = n->num_inputs + n->num_weights;
        int max_input_idx = total_inputs > 0 ? total_inputs - 1 : 0;
        if (n->type == OP_RELU || n->type == OP_SIGMOID || n->type == OP_GELU
            || n->type == OP_SILU || n->type == OP_EXP || n->type == OP_TANH)
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
            if (tid >= 0 && tid < g->num_tensors) {
                op_outputs[i] = g->tensors[tid].tensor->data;
            } else {
                op_outputs[i] = NULL;
            }
        }

        /* Handle ops that need extra metadata via inputs[] slot */
        /* relu/sigmoid/gelu/tanh: need numel as inputs[1] */
        if (n->type == OP_RELU || n->type == OP_SIGMOID || n->type == OP_GELU
            || n->type == OP_SILU || n->type == OP_EXP || n->type == OP_TANH) {
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
                        if (bytes > 0)
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
        /* Skip D2H for KV-cache tensors (they persist on device between calls) */
        if (use_cuda) {
            for (int i = 0; i < n->num_outputs; i++) {
                int tid = effective_output_tids[i] ? effective_output_tids[i]
                                                   : n->output_tensors[i];
                if (tid >= 0 && tid != g->kv_cache_K_tid && tid != g->kv_cache_V_tid) {
                    tensor_t* ot = g->tensors[tid].tensor;
                    if (ot && ot->numel > 0) {
                        tensor_copy_to_host(ot);
                    }
                }
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
                if (src && outputs[i]->data != src->data) {
                    const data_type_info_t* info = data_type_get_info(src->dtype);
                    size_t bytes = (size_t)src->numel * info->size;
                    memcpy(outputs[i]->data, src->data, bytes);
                }
            }
        }
    }

    ret = 0;

#ifdef USE_CUDA
    /* CUDA Graph capture is skipped — see comment above.
     * Mark state as 2 to avoid repeated attempts. */
    if (use_cuda && g->cuda_graph_state == 0 && ret == 0) {
        g->cuda_graph_state = 2;  /* skip capture, go straight to "done" */
    }
#endif

cleanup:
    /* Restore MHA-fused node to its original type and configuration */
    if (mha_restore.node_id >= 0) {
        graph_node_t* n = &g->nodes[mha_restore.node_id];
        free(n->input_tensors);
        free(n->output_tensors);
        free(n->weights);
        free(n->params);
        n->type         = mha_restore.saved_type;
        n->num_inputs   = mha_restore.saved_num_inputs;
        n->input_tensors = mha_restore.saved_input_tensors;
        n->num_outputs  = mha_restore.saved_num_outputs;
        n->output_tensors = mha_restore.saved_output_tensors;
        n->num_weights  = mha_restore.saved_num_weights;
        n->weights      = mha_restore.saved_weights;
        n->params       = mha_restore.saved_params;
        n->params_size  = mha_restore.saved_params_size;
    }
    free(fused_skip);
    return ret;
}
