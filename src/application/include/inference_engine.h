#ifndef INFERENCE_ENGINE_H_
#define INFERENCE_ENGINE_H_

#include "platform.h"

typedef struct inference_session_t inference_session_t;

int  inference_engine_init(void);
void inference_engine_finalize(void);

inference_session_t* inference_session_load(const char* onnx_path);
int  inference_session_run(inference_session_t* session,
                            tensor_t* inputs[], tensor_t* outputs[],
                            int use_cuda);
int  inference_session_num_inputs(inference_session_t* session);
int  inference_session_num_outputs(inference_session_t* session);
void inference_session_destroy(inference_session_t* session);

#endif /* INFERENCE_ENGINE_H_ */
