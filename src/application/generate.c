/**
 * Autoregressive text generation loop.
 *
 * Two modes:
 *   1. KV-cache decode (preferred): when the graph has KV-cache tensors
 *      marked (g->kv_cache_K_tid >= 0), runs prefill->decode with mha_decode,
 *      advancing cache_len each step via graph_update_cache_len. The graph
 *      is expected to take a single token ID and output [1, vocab] logits.
 *   2. Full-graph fallback (legacy): re-runs the entire model for each token
 *      with a sliding input window. Used by ONNX-driven graphs (OP_MHA_FUSED)
 *      that have no KV-cache ports.
 *
 * Supports greedy argmax (temperature=0) and temperature sampling (temperature>0).
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

/* Temperature sampling: returns a token index sampled from softmax(logits/T).
   Falls back to argmax if temperature <= 0 or on error. */
static int64_t sample_token(const float* logits, int64_t vocab, int temperature) {
    if (temperature <= 0) return argmax_f32(logits, vocab);

    /* softmax(logits / temperature) */
    float* probs = (float*)malloc((size_t)vocab * sizeof(float));
    if (!probs) return argmax_f32(logits, vocab);

    float max_l = logits[0];
    for (int64_t i = 1; i < vocab; i++) if (logits[i] > max_l) max_l = logits[i];

    float inv_t = 1.0f / (float)temperature;
    float sum = 0.0f;
    for (int64_t i = 0; i < vocab; i++) {
        probs[i] = expf((logits[i] - max_l) * inv_t);
        sum += probs[i];
    }
    if (sum < 1e-12f) { free(probs); return argmax_f32(logits, vocab); }

    /* Sample from the distribution */
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0.0f;
    int64_t chosen = vocab - 1;
    for (int64_t i = 0; i < vocab; i++) {
        cum += probs[i];
        if (r <= cum) { chosen = i; break; }
    }
    free(probs);
    return chosen;
}

/* ---- KV-cache decode path ----
   Graph takes single token ID (int64, shape [1]) and outputs logits [1, vocab].
   Prefill: feed each prompt token sequentially, advancing cache_len 0..prompt_len-1.
   Decode: feed generated token, continuing cache_len advancement. */
static int generate_kv_cache(inference_graph_t* g,
                             const int64_t* prompt, int64_t prompt_len,
                             int64_t* output, const generate_config_t* cfg) {
    int input_node_id = g->input_node_ids[0];
    int output_node_id = g->output_node_ids[0];
    graph_node_t* input_node = &g->nodes[input_node_id];
    graph_node_t* output_node = &g->nodes[output_node_id];
    int input_tid = input_node->output_tensors[0];
    int output_tid = output_node->input_tensors[0];
    tensor_t* input_tensor = g->tensors[input_tid].tensor;
    tensor_t* output_tensor = g->tensors[output_tid].tensor;

    int64_t vocab = output_tensor->shape[output_tensor->ndim - 1];
    int64_t* id_data = (int64_t*)input_tensor->data;
    float* logits = (float*)output_tensor->data;

    int64_t generated = 0;
    int64_t next_token = -1;

    /* Prefill: process each prompt token, filling KV-cache.
       The last prompt token's logits produce the first generated token. */
    for (int64_t step = 0; step < prompt_len; step++) {
        id_data[0] = prompt[step];
        graph_update_cache_len(g, step);

        tensor_t* inputs[] = {input_tensor};
        tensor_t* outputs[] = {output_tensor};
        int ret = graph_execute(g, inputs, outputs, cfg->use_cuda);
        if (ret != 0) {
            fprintf(stderr, "generate_kv: prefill step %lld failed (ret=%d)\n",
                    (long long)step, ret);
            return -1;
        }
    }

    /* First generated token from last prefill position */
    next_token = sample_token(logits, vocab, cfg->temperature);
    if (cfg->verbose) {
        fprintf(stderr, "generate_kv: prompt %lld tokens, first token = %lld\n",
                (long long)prompt_len, (long long)next_token);
    }

    /* Decode loop */
    int64_t cache_len = prompt_len;
    for (int64_t step = 0; step < cfg->max_new_tokens; step++) {
        output[generated] = next_token;
        generated++;

        if (cfg->eos_token_id >= 0 && next_token == cfg->eos_token_id) {
            if (cfg->verbose) {
                fprintf(stderr, "generate_kv: EOS %lld at step %lld\n",
                        (long long)next_token, (long long)step);
            }
            break;
        }

        /* Feed the just-generated token, advance cache_len */
        id_data[0] = next_token;
        graph_update_cache_len(g, cache_len);
        cache_len++;

        tensor_t* inputs[] = {input_tensor};
        tensor_t* outputs[] = {output_tensor};
        int ret = graph_execute(g, inputs, outputs, cfg->use_cuda);
        if (ret != 0) {
            fprintf(stderr, "generate_kv: decode step %lld failed (ret=%d)\n",
                    (long long)step, ret);
            break;
        }

        next_token = sample_token(logits, vocab, cfg->temperature);
        if (cfg->verbose) {
            fprintf(stderr, "generate_kv: step %lld, token = %lld\n",
                    (long long)step, (long long)next_token);
        }
    }

    return (int)generated;
}

/* ---- Legacy full-graph path (no KV-cache) ---- */
static int generate_full_graph(inference_graph_t* g,
                               const int64_t* prompt, int64_t prompt_len,
                               int64_t* output, const generate_config_t* cfg) {
    int input_node_id = g->input_node_ids[0];
    int output_node_id = g->output_node_ids[0];
    graph_node_t* input_node = &g->nodes[input_node_id];
    graph_node_t* output_node = &g->nodes[output_node_id];
    int input_tid = input_node->output_tensors[0];
    int output_tid = output_node->input_tensors[0];
    tensor_t* input_tensor = g->tensors[input_tid].tensor;
    tensor_t* output_tensor = g->tensors[output_tid].tensor;

    int64_t vocab_size = output_tensor->shape[output_tensor->ndim - 1];
    int64_t seq_len = input_tensor->shape[1];
    int64_t batch = input_tensor->shape[0];
    int64_t prompt_tokens_used = (prompt_len < seq_len) ? prompt_len : seq_len;

    int64_t* input_data = (int64_t*)input_tensor->data;
    memset(input_data, 0, (size_t)(batch * seq_len) * sizeof(int64_t));
    for (int64_t i = 0; i < prompt_tokens_used; i++) {
        input_data[i] = prompt[i];
    }

    int64_t* all_tokens = (int64_t*)malloc((size_t)(prompt_len + cfg->max_new_tokens) * sizeof(int64_t));
    if (!all_tokens) return -1;
    memcpy(all_tokens, prompt, (size_t)prompt_len * sizeof(int64_t));
    int64_t total_tokens = prompt_len;
    int64_t generated = 0;

    /* Prefill */
    tensor_t* inputs[] = {input_tensor};
    tensor_t* outputs[] = {output_tensor};
    int ret = graph_execute(g, inputs, outputs, cfg->use_cuda);
    if (ret != 0) {
        fprintf(stderr, "generate: prefill failed (ret=%d)\n", ret);
        free(all_tokens);
        return -1;
    }

    float* logits = (float*)output_tensor->data;
    int64_t last_pos = prompt_tokens_used - 1;
    float* last_logits = logits + last_pos * vocab_size;
    int64_t next_token = sample_token(last_logits, vocab_size, cfg->temperature);

    if (cfg->verbose) {
        fprintf(stderr, "generate: prompt %lld tokens, first predicted token = %lld\n",
                (long long)prompt_len, (long long)next_token);
    }

    /* Decode loop (re-runs full graph each step) */
    for (int64_t step = 0; step < cfg->max_new_tokens; step++) {
        output[generated] = next_token;
        generated++;
        all_tokens[total_tokens] = next_token;
        total_tokens++;

        if (cfg->eos_token_id >= 0 && next_token == cfg->eos_token_id) {
            if (cfg->verbose) {
                fprintf(stderr, "generate: EOS token %lld at step %lld\n",
                        (long long)next_token, (long long)step);
            }
            break;
        }

        int64_t start = (total_tokens > seq_len) ? total_tokens - seq_len : 0;
        int64_t window_len = total_tokens - start;
        memset(input_data, 0, (size_t)(batch * seq_len) * sizeof(int64_t));
        for (int64_t i = 0; i < window_len && i < seq_len; i++) {
            input_data[i] = all_tokens[start + i];
        }

        ret = graph_execute(g, inputs, outputs, cfg->use_cuda);
        if (ret != 0) {
            fprintf(stderr, "generate: decode step %lld failed (ret=%d)\n",
                    (long long)step, ret);
            break;
        }

        last_pos = (window_len - 1 < seq_len) ? window_len - 1 : seq_len - 1;
        last_logits = logits + last_pos * vocab_size;
        next_token = sample_token(last_logits, vocab_size, cfg->temperature);

        if (cfg->verbose) {
            fprintf(stderr, "generate: step %lld, token = %lld\n",
                    (long long)step, (long long)next_token);
        }
    }

    free(all_tokens);
    return (int)generated;
}

int generate_tokens(inference_graph_t* g,
                    const int64_t* prompt, int64_t prompt_len,
                    int64_t* output, const generate_config_t* cfg) {
    if (!g || !prompt || !output || !cfg) return -1;
    if (prompt_len <= 0 || cfg->max_new_tokens <= 0) return -1;
    if (g->num_inputs < 1 || g->num_outputs < 1) return -1;

    /* Dispatch: KV-cache decode path if the graph has KV-cache tensors marked,
       otherwise legacy full-graph path. */
    if (g->kv_cache_K_tid >= 0) {
        return generate_kv_cache(g, prompt, prompt_len, output, cfg);
    }
    return generate_full_graph(g, prompt, prompt_len, output, cfg);
}
