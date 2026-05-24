#include "inference_engine.h"
#include "onnx_loader.h"
#include "graph.h"
#include <stdlib.h>

struct inference_session_t {
    onnx_model_t* model;
};

int inference_engine_init(void) {
    return 0;
}

void inference_engine_finalize(void) {
}

inference_session_t* inference_session_load(const char* onnx_path) {
    onnx_model_t* model = onnx_load_from_file(onnx_path);
    if (!model) return NULL;

    inference_session_t* session =
        (inference_session_t*)calloc(1, sizeof(inference_session_t));
    if (!session) {
        onnx_model_destroy(model);
        return NULL;
    }
    session->model = model;
    return session;
}

int inference_session_run(inference_session_t* session,
                           tensor_t* inputs[], tensor_t* outputs[],
                           int use_cuda) {
    if (!session || !session->model || !session->model->graph)
        return -1;
    return graph_execute(session->model->graph, inputs, outputs, use_cuda != 0);
}

int inference_session_num_inputs(inference_session_t* session) {
    if (!session || !session->model) return 0;
    return session->model->num_inputs;
}

int inference_session_num_outputs(inference_session_t* session) {
    if (!session || !session->model) return 0;
    return session->model->num_outputs;
}

void inference_session_destroy(inference_session_t* session) {
    if (!session) return;
    onnx_model_destroy(session->model);
    free(session);
}
