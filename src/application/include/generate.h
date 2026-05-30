#ifndef GENERATE_H_
#define GENERATE_H_

#include "graph.h"
#include "platform.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Generation configuration */
typedef struct {
    int64_t  max_new_tokens;   /* maximum tokens to generate */
    int64_t  eos_token_id;    /* stop generation when this token is produced (-1 = no stop) */
    int      temperature;     /* 0 = greedy argmax, >0 = sample with temperature */
    int      verbose;         /* print each generated token */
} generate_config_t;

/* Default config */
static inline generate_config_t generate_default_config(void) {
    generate_config_t cfg;
    cfg.max_new_tokens = 64;
    cfg.eos_token_id = -1;
    cfg.temperature = 0;
    cfg.verbose = 1;
    return cfg;
}

/**
 * Autoregressive text generation.
 *
 * Given a graph with embedding + transformer + LM head, generates
 * tokens one at a time using KV-cache for efficient decode.
 *
 * The graph must have:
 *   - Input node 0: token IDs (int64, shape [B, S])
 *   - Output node 0: logits (float32, shape [B, S, vocab])
 *
 * For the prefill pass, S = prompt_length.
 * For decode passes, S = 1 (single new token).
 *
 * @param g          The inference graph
 * @param prompt     Input token IDs (int64, shape [prompt_len])
 * @param prompt_len Number of prompt tokens
 * @param output     Buffer for generated token IDs (int64, shape [max_new_tokens])
 * @param cfg        Generation config
 * @return           Number of tokens generated (including EOS if hit), or <0 on error
 */
int generate_tokens(inference_graph_t* g,
                    const int64_t* prompt, int64_t prompt_len,
                    int64_t* output, const generate_config_t* cfg);

#ifdef __cplusplus
}
#endif

#endif /* GENERATE_H_ */
