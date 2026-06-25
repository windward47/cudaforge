/**
 * GPT-2 end-to-end inference loop benchmark — amplifies kernel signal for nsys.
 * Runs GPT-2 prefill N times, measures avg ms/iter.
 *
 * Usage: bench_e2e [iters]   (default 50)
 */
#include "platform.h"
#include "operator.h"
#include "inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "cuda_platform.h"
#include "cuda_ops.h"
#endif

extern int operator_init_all(void);

#ifdef _WIN32
#include <windows.h>
static double now_ms(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

static int load_binary(const char* path, void* buf, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    size_t r = fread(buf, 1, bytes, f);
    fclose(f);
    return (r == bytes) ? 0 : -1;
}

int main(int argc, char** argv) {
    int iters = (argc > 1) ? atoi(argv[1]) : 50;
    platform_init();
    operator_init_all();
#ifdef USE_CUDA
    cuda_platform_init(0);
#endif

    inference_session_t* sess = inference_session_load("tests/gpt2_full.onnx");
    if (!sess) { fprintf(stderr, "FAIL: gpt2_full.onnx not found\n"); return 1; }

    int64_t input_shape[] = {1, 8};
    int64_t logits_shape[] = {1, 8, 256};
    tensor_t* t_in = tensor_create(DATA_TYPE_I64, 2, input_shape);
    tensor_t* t_out = tensor_create(DATA_TYPE_F32, 3, logits_shape);

    int64_t input_ids[8];
    if (load_binary("tests/gpt2_full_input.bin", input_ids, 8 * sizeof(int64_t)) != 0) {
        fprintf(stderr, "FAIL: gpt2_full_input.bin not found\n"); return 1;
    }
    memcpy(t_in->data, input_ids, 8 * sizeof(int64_t));

    tensor_t* inputs[] = {t_in};
    tensor_t* outputs[] = {t_out};

    /* warmup (CUDA) */
    for (int i = 0; i < 3; i++) inference_session_run(sess, inputs, outputs, 1);
#ifdef USE_CUDA
    g_cuda.stream_synchronize(0);
#endif

    fprintf(stderr, "=== GPT-2 Prefill Benchmark (%d iters, CUDA) ===\n", iters);
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) inference_session_run(sess, inputs, outputs, 1);
#ifdef USE_CUDA
    g_cuda.stream_synchronize(0);
#endif
    double t1 = now_ms();
    fprintf(stderr, "CUDA: %.3f ms/iter (total %.1f ms)\n", (t1-t0)/iters, t1-t0);

    inference_session_destroy(sess);
    tensor_destroy(t_in); tensor_destroy(t_out);
#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();
    return 0;
}
