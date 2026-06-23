#ifndef ONNX_LOADER_H_
#define ONNX_LOADER_H_

#include "platform.h"
#include "graph.h"

/* Result of loading an ONNX model */
typedef struct onnx_model_t onnx_model_t;

struct onnx_model_t {
    inference_graph_t* graph;
    int     num_inputs;
    char**  input_names;
    int64_t** input_shapes;
    int*    input_ndims;
    int     num_outputs;
    char**  output_names;
};

/* Load an ONNX model from a file. Returns NULL on failure. */
onnx_model_t* onnx_load_from_file(const char* path);

/* Free all resources */
void onnx_model_destroy(onnx_model_t* model);

/* Post-load: quantize MatMul weight tensors to INT8 block format.
 * Returns number of quantized weights. Requires quantize_int.h. */
int onnx_quantize_weights(onnx_model_t* model);

#endif /* ONNX_LOADER_H_ */
