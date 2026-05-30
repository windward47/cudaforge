/**
 * Autoregressive text generation loop.
 *
 * Strategy:
 *   1. Prefill: run full prompt through graph → get logits
 *   2. Decode: for each new token, update input tensor and re-run graph
 *   3. Greedy argmax (or sampling) to select next token
 *
 * Note: This implementation re-runs the full model for each token.
 * For production use, a KV-cache-aware decode path using mha_decode
 * kernel would be much more efficient.
 */
#include "generate.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Find argmax of float array, return index */
static int64_t argmax_f32(const float* data, int64_t n) {
    int64_t best_idx = 0;
    float best_val = data[0];
    for (int64_t i = 1; i < n; i++) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int generate_tokens(inference_graph_t* g,
                    const int64_t* prompt, int64_t prompt_len,
                    int64_t* output, const generate_config_t* cfg) {
    if (!g || !prompt || !output || !cfg) return -1;
    if (prompt_len <= 0 || cfg->max_new_tokens <= 0) return -1;

    /* Find input and output tensors */
    if (g->num_inputs < 1 || g->num_outputs < 1) return -1;

    int input_node_id = g->input_node_ids[0];
    int output_node_id = g->output_node_ids[0];
    if (input_node_id < 0 || output_node_id < 0) return -1;

    graph_node_t* input_node = &g->nodes[input_node_id];
    graph_node_t* output_node = &g->nodes[output_node_id];
    if (input_node->num_outputs < 1 || output_node->num_inputs < 1) return -1;

    int input_tid = input_node->output_tensors[0];
    int output_tid = output_node->input_tensors[0];
    if (input_tid < 0 || input_tid >= g->num_tensors) return -1;
    if (output_tid < 0 || output_tid >= g->num_tensors) return -1;

    tensor_t* input_tensor = g->tensors[input_tid].tensor;
    tensor_t* output_tensor = g->tensors[output_tid].tensor;
    if (!input_tensor || !output_tensor) return -1;

    /* Determine vocab size from output tensor shape */
    int64_t vocab_size = output_tensor->shape[output_tensor->ndim - 1];
    int64_t seq_len = input_tensor->shape[1];  /* fixed seq_len from model */
    int64_t hidden = output_tensor->shape[output_tensor->ndim - 1];  /* vocab or hidden */

    /* For the test model: input is (1, 8) int64, output is (1, 8, 256) float */
    /* We'll use only the last position's logits for next token prediction */
    int64_t batch = input_tensor->shape[0];
    int64_t prompt_tokens_used = (prompt_len < seq_len) ? prompt_len : seq_len;

    /* Fill input with prompt tokens (pad with 0 if prompt < seq_len) */
    int64_t* input_data = (int64_t*)input_tensor->data;
    memset(input_data, 0, (size_t)(batch * seq_len) * sizeof(int64_t));
    for (int64_t i = 0; i < prompt_tokens_used; i++) {
        input_data[i] = prompt[i];
    }

    /* Buffer for generated tokens */
    int64_t* all_tokens = (int64_t*)malloc((size_t)(prompt_len + cfg->max_new_tokens) * sizeof(int64_t));
    if (!all_tokens) return -1;
    memcpy(all_tokens, prompt, (size_t)prompt_len * sizeof(int64_t));
    int64_t total_tokens = prompt_len;
    int64_t generated = 0;

    /* Prefill: run full prompt through graph */
    tensor_t* inputs[] = {input_tensor};
    tensor_t* outputs[] = {output_tensor};
    int ret = graph_execute(g, inputs, outputs, 1 /* use_cuda */);
    if (ret != 0) {
        fprintf(stderr, "generate: prefill failed (ret=%d)\n", ret);
        free(all_tokens);
        return -1;
    }

    /* Get logits for last position */
    float* logits = (float*)output_tensor->data;
    int64_t last_pos = prompt_tokens_used - 1;
    float* last_logits = logits + last_pos * vocab_size;
    int64_t next_token = argmax_f32(last_logits, vocab_size);

    if (cfg->verbose) {
        fprintf(stderr, "generate: prompt %lld tokens, first predicted token = %lld\n",
                (long long)prompt_len, (long long)next_token);
    }

    /* Generate loop */
    for (int64_t step = 0; step < cfg->max_new_tokens; step++) {
        output[generated] = next_token;
        generated++;
        all_tokens[total_tokens] = next_token;
        total_tokens++;

        /* Check EOS */
        if (cfg->eos_token_id >= 0 && next_token == cfg->eos_token_id) {
            if (cfg->verbose) {
                fprintf(stderr, "generate: EOS token %lld at step %lld\n",
                        (long long)next_token, (long long)step);
            }
            break;
        }

        /* Prepare input for next step: shift window or append */
        /* For simplicity: use last seq_len tokens from all_tokens */
        int64_t start = (total_tokens > seq_len) ? total_tokens - seq_len : 0;
        int64_t window_len = total_tokens - start;
        memset(input_data, 0, (size_t)(batch * seq_len) * sizeof(int64_t));
        for (int64_t i = 0; i < window_len && i < seq_len; i++) {
            input_data[i] = all_tokens[start + i];
        }

        /* Re-run graph */
        ret = graph_execute(g, inputs, outputs, 1);
        if (ret != 0) {
            fprintf(stderr, "generate: decode step %lld failed (ret=%d)\n",
                    (long long)step, ret);
            break;
        }

        /* Get logits for last valid position */
        last_pos = (window_len - 1 < seq_len) ? window_len - 1 : seq_len - 1;
        last_logits = logits + last_pos * vocab_size;
        next_token = argmax_f32(last_logits, vocab_size);

        if (cfg->verbose) {
            fprintf(stderr, "generate: step %lld, token = %lld\n",
                    (long long)step, (long long)next_token);
        }
    }

    free(all_tokens);
    return (int)generated;
}
