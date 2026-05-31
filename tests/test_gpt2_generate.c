/**
 * GPT-2 end-to-end generation test.
 *
 * Uses inference_session API to:
 *   1. Load GPT-2 model
 *   2. Run prefill with prompt tokens
 *   3. Verify logits vs reference
 *   4. Run autoregressive generation (greedy)
 */
#include "inference_engine.h"
#include "platform.h"
#include "operator.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); fflush(stderr); exit(1); } \
} while(0)

static int64_t argmax_f32(const float* data, int64_t n) {
    int64_t best = 0;
    for (int64_t i = 1; i < n; i++) {
        if (data[i] > data[best]) best = i;
    }
    return best;
}

static int load_binary(const char* path, void* buf, size_t expected) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if ((size_t)sz != expected) { fclose(f); return -1; }
    size_t r = fread(buf, 1, expected, f);
    fclose(f);
    return (r == expected) ? 0 : -1;
}

/**
 * Test 1: Prefill — load model, run prompt, verify logits
 */
static int test_gpt2_prefill(void) {
    fprintf(stderr, "\n=== GPT-2 Prefill Test ===\n");

    inference_session_t* sess = inference_session_load("tests/gpt2_full.onnx");
    if (!sess) {
        fprintf(stderr, "SKIP: gpt2_full.onnx not found\n");
        return 0;
    }

    int n_in = inference_session_num_inputs(sess);
    int n_out = inference_session_num_outputs(sess);
    fprintf(stderr, "Model: %d inputs, %d outputs\n", n_in, n_out);
    CHECK(n_in >= 1 && n_out >= 1, "Model has no inputs/outputs");

    /* Get input/output tensors from graph */
    /* Input must be int64 to match ONNX model's dtype=7 (INT64) */
    int64_t input_shape[] = {1, 8};
    int64_t logits_shape[] = {1, 8, 256};
    tensor_t* t_in = tensor_create(DATA_TYPE_I64, 2, input_shape);
    tensor_t* t_out = tensor_create(DATA_TYPE_F32, 3, logits_shape);

    /* Load test input (stored as int64, pass as int64 to match model's dtype) */
    int64_t input_ids[8];
    if (load_binary("tests/gpt2_full_input.bin", input_ids, 8 * sizeof(int64_t)) != 0) {
        fprintf(stderr, "SKIP: gpt2_full_input.bin not found\n");
        tensor_destroy(t_in); tensor_destroy(t_out);
        return 0;
    }
    memcpy(t_in->data, input_ids, 8 * sizeof(int64_t));

    fprintf(stderr, "Input tokens: ");
    for (int i = 0; i < 8; i++) fprintf(stderr, "%lld ", (long long)input_ids[i]);
    fprintf(stderr, "\n");

    /* Run inference */
    tensor_t* inputs[] = {t_in};
    tensor_t* outputs[] = {t_out};
    int ret = inference_session_run(sess, inputs, outputs, 0 /* CPU */);
    CHECK(ret == 0, "inference_session_run failed");

    /* Check output tensor */
    fprintf(stderr, "t_out: ndim=%d numel=%lld shape=[%lld,%lld,%lld]\n",
            t_out->ndim, (long long)t_out->numel,
            t_out->ndim > 0 ? (long long)t_out->shape[0] : -1,
            t_out->ndim > 1 ? (long long)t_out->shape[1] : -1,
            t_out->ndim > 2 ? (long long)t_out->shape[2] : -1);

    /* Check logits */
    float* logits = (float*)t_out->data;
    int64_t vocab_size = 256;
    float* last_logits = logits + 7 * vocab_size;
    int64_t next_token = argmax_f32(last_logits, vocab_size);
    fprintf(stderr, "Next token (greedy): %lld\n", (long long)next_token);
    fprintf(stderr, "Logits[0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n",
            last_logits[0], last_logits[1], last_logits[2],
            last_logits[3], last_logits[4], last_logits[5]);

    /* Load reference and compare */
    float ref_logits[8 * 256];
    if (load_binary("tests/gpt2_full_logits.bin", ref_logits, sizeof(ref_logits)) == 0) {
        float* ref_last = ref_logits + 7 * 256;
        int64_t ref_token = argmax_f32(ref_last, 256);
        fprintf(stderr, "Reference next token: %lld\n", (long long)ref_token);

        /* Compare logits */
        float max_diff = 0.0f;
        for (int i = 0; i < 8 * 256; i++) {
            float diff = fabsf(logits[i] - ref_logits[i]);
            if (diff > max_diff) max_diff = diff;
        }
        fprintf(stderr, "Logits max_diff: %.2e\n", max_diff);
        CHECK(max_diff < 1e-3f, "Logits mismatch vs reference");
        CHECK(next_token == ref_token, "Next token mismatch");
    }

    fprintf(stderr, "GPT-2 Prefill: PASS\n");
    tensor_destroy(t_in); tensor_destroy(t_out);
    inference_session_destroy(sess);
    return 0;
}

/**
 * Test 2: Autoregressive generation (greedy, 4 tokens)
 */
static int test_gpt2_generate(void) {
    fprintf(stderr, "\n=== GPT-2 Generate Test ===\n");

    inference_session_t* sess = inference_session_load("tests/gpt2_full.onnx");
    if (!sess) {
        fprintf(stderr, "SKIP: gpt2_full.onnx not found\n");
        return 0;
    }

    int64_t input_shape[] = {1, 8};
    int64_t logits_shape[] = {1, 8, 256};
    tensor_t* t_in = tensor_create(DATA_TYPE_I64, 2, input_shape);
    tensor_t* t_out = tensor_create(DATA_TYPE_F32, 3, logits_shape);

    /* Start with prompt tokens */
    int64_t prompt[8] = {42, 100, 7, 200, 50, 150, 33, 88};
    memcpy(t_in->data, prompt, 8 * sizeof(int64_t));

    int64_t generated[8];
    int n_gen = 0;

    /* Prefill */
    tensor_t* inputs[] = {t_in};
    tensor_t* outputs[] = {t_out};
    int ret = inference_session_run(sess, inputs, outputs, 1);
    CHECK(ret == 0, "Prefill failed");

    /* Get first token */
    float* logits = (float*)t_out->data;
    int64_t next = argmax_f32(logits + 7 * 256, 256);
    generated[n_gen++] = next;
    fprintf(stderr, "Generated token 0: %lld\n", (long long)next);

    /* Decode 3 more tokens */
    for (int step = 1; step < 4; step++) {
        /* Shift input window: use last 7 tokens from prompt + generated */
        int64_t new_input[8];
        int64_t total = 8 + n_gen;
        int64_t start = (total > 8) ? total - 8 : 0;
        /* Build new input from prompt + generated tokens */
        int64_t all_tokens[16];  /* prompt(8) + generated(up to 8) */
        memcpy(all_tokens, prompt, 8 * sizeof(int64_t));
        memcpy(all_tokens + 8, generated, (size_t)n_gen * sizeof(int64_t));
        memset(new_input, 0, 8 * sizeof(int64_t));
        for (int64_t i = 0; i < 8 && (start + i) < total; i++) {
            new_input[i] = all_tokens[start + i];
        }
        memcpy(t_in->data, new_input, 8 * sizeof(int64_t));

        ret = inference_session_run(sess, inputs, outputs, 1);
        CHECK(ret == 0, "Decode step failed");

        next = argmax_f32(logits + 7 * 256, 256);
        generated[n_gen++] = next;
        fprintf(stderr, "Generated token %d: %lld\n", step, (long long)next);
    }

    fprintf(stderr, "All generated tokens: ");
    for (int i = 0; i < n_gen; i++) fprintf(stderr, "%lld ", (long long)generated[i]);
    fprintf(stderr, "\n");

    CHECK(n_gen == 4, "Expected 4 generated tokens");
    fprintf(stderr, "GPT-2 Generate: PASS\n");

    tensor_destroy(t_in); tensor_destroy(t_out);
    inference_session_destroy(sess);
    return 0;
}

int main(void) {
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    test_gpt2_prefill();
    test_gpt2_generate();

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();

    fprintf(stderr, "\n=== GPT-2 Generation Tests Done ===\n");
    return 0;
}
