#ifndef INFERENCE_GRAPH_H_
#define INFERENCE_GRAPH_H_

#include "platform.h"
#include "operator.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Supported operator types */
typedef enum {
    OP_RELU,
    OP_SIGMOID,
    OP_GELU,
    OP_MATMUL,
    OP_CONV2D,
    OP_MAXPOOL2D,
    OP_AVGPOOL2D,
    OP_BATCHNORM,
    OP_ADD,
    OP_RESHAPE,
    OP_GLOBALAVGPOOL,
    OP_SOFTMAX,
    OP_SILU,
    OP_MUL,
    OP_CONCAT,
    OP_RESIZE,
    OP_TRANSPOSE,
    OP_SUB,
    OP_DIV,
    OP_SLICE,
    OP_SPLIT,
    OP_LAYERNORM,
    OP_GATHER,
    OP_SQUEEZE_UNSQUEEZE,
    OP_EXP,
    OP_REDUCE,
    OP_CAST,
    OP_ARGMAX,
    OP_MHA_FUSED,
    OP_MHA_DECODE,
    OP_CAUSAL_MASK,
    OP_ROPE,
    OP_PAD,
    OP_INPUT,
    OP_OUTPUT
} op_type_t;

/* Node in the computation graph */
typedef struct {
    op_type_t       type;
    int             num_inputs;
    int             num_outputs;
    int*            input_tensors;   /* indices into graph->tensors[] */
    int*            output_tensors;  /* indices into graph->tensors[] */
    int             num_weights;
    tensor_t**      weights;         /* weight/bias tensors */
    void*           params;          /* operator-specific params (e.g. matmul_params_t) */
    size_t          params_size;     /* size of params for copy */
} graph_node_t;

/* Tensor slot (edge between nodes) */
typedef struct {
    tensor_t* tensor;
    int       producer;  /* node index that writes this, -1 if graph input */
    int       consumer;  /* node index that reads this, -1 if graph output */
} graph_tensor_t;

/* Full graph */
typedef struct {
    int             num_nodes;
    graph_node_t*   nodes;
    int             num_tensors;
    graph_tensor_t* tensors;
    int*            topo_order;    /* indices into nodes[], computed */
    int             num_inputs;
    int             num_outputs;
    int*            input_node_ids;
    int*            output_node_ids;
    /* KV-cache support for autoregressive decode */
    int             kv_cache_K_tid;  /* tensor index for K cache, -1 if unused */
    int             kv_cache_V_tid;  /* tensor index for V cache, -1 if unused */
    int             permanent_fusion; /* if 1, skip MHA restore after execution */
} inference_graph_t;

/* Graph construction */
inference_graph_t* graph_create(void);
int   graph_add_node(inference_graph_t* g, op_type_t type,
                     int num_inputs, const int* input_tensors,
                     int num_outputs, const int* output_tensors,
                     int num_weights, tensor_t** weights,
                     void* params, size_t params_size);
int   graph_add_tensor(inference_graph_t* g, tensor_t* t);
int   graph_set_input(inference_graph_t* g, int node_id);
int   graph_set_output(inference_graph_t* g, int node_id);
int   graph_build(inference_graph_t* g);
void  graph_destroy(inference_graph_t* g);

/* KV-cache: mark tensors as persistent (skip D2H copy between calls) */
void  graph_set_kv_cache(inference_graph_t* g, int K_tensor_id, int V_tensor_id);

/* Permanent fusion: skip MHA restore after execution (for decode loops) */
void  graph_set_permanent_fusion(inference_graph_t* g, int enable);

/* Execution */
int   graph_execute(inference_graph_t* g, tensor_t* inputs[],
                    tensor_t* outputs[], bool use_cuda);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_GRAPH_H_ */
