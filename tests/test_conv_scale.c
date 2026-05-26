/**
 * Targeted test: compare CPU vs CUDA conv at ResNet-18 scale.
 */
#include "platform.h"
#include "operator.h"
#include "conv_int.h"
#include "cuda_ops.h"
#include "cuda_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

int main(void) {
    platform_init();
    operator_init_all();
    cuda_platform_init(0);

    /* ResNet layer1.0.conv1: 64×56×56 input, 3×3 kernel, 64 output, stride=1, pad=1 */
    int64_t N=1, C=64, H=56, W=56, K=64, KH=3, KW=3;
    int64_t pad=1, stride=1, dil=1;
    int64_t OH = (H + 2*pad - dil*(KH-1) - 1)/stride + 1;
    int64_t OW = (W + 2*pad - dil*(KW-1) - 1)/stride + 1;

    size_t in_sz  = N*C*H*W;
    size_t w_sz   = K*C*KH*KW;
    size_t out_sz = N*K*OH*OW;
    size_t bias_sz = K;

    float* h_in   = (float*)malloc(in_sz * sizeof(float));
    float* h_w    = (float*)malloc(w_sz * sizeof(float));
    float* h_bias = (float*)malloc(bias_sz * sizeof(float));
    float* h_out_cpu = (float*)calloc(out_sz, sizeof(float));
    float* h_out_cuda = (float*)calloc(out_sz, sizeof(float));

    /* Random data for reproducibility */
    srand(42);
    for (size_t i = 0; i < in_sz; i++) h_in[i] = (float)(rand() % 2000 - 1000) / 100.0f;
    for (size_t i = 0; i < w_sz; i++)  h_w[i]  = (float)(rand() % 200 - 100) / 100.0f;
    for (size_t i = 0; i < bias_sz; i++) h_bias[i] = (float)(rand() % 200 - 100) / 100.0f;

    /* CPU conv */
    const operator_registry_t* cpu_op = operator_find("conv2d_f32");
    {
        const void* inputs[]  = {h_in, h_w, h_bias};
        void*       outputs[] = {h_out_cpu};
        conv_params_t cp = {.N=N,.C=C,.H=H,.W=W,.K=K,.kernel_h=KH,.kernel_w=KW,
                            .stride_h=stride,.stride_w=stride,.pad_h=pad,.pad_w=pad,
                            .dilation_h=dil,.dilation_w=dil};
        int rc = cpu_op->func(inputs, outputs, (const operator_params_t*)&cp, NULL);
        fprintf(stderr, "CPU conv: rc=%d\n", rc);
    }

    /* CUDA conv */
    const operator_registry_t* cuda_op = operator_find("conv2d_f32_cuda");
    {
        float *d_in, *d_w, *d_bias, *d_out;
        d_in   = (float*)g_cuda.device_alloc(in_sz * sizeof(float));
        d_w    = (float*)g_cuda.device_alloc(w_sz * sizeof(float));
        d_bias = (float*)g_cuda.device_alloc(bias_sz * sizeof(float));
        d_out  = (float*)g_cuda.device_alloc(out_sz * sizeof(float));
        g_cuda.memcpy_h2d(d_in, h_in, in_sz * sizeof(float), 0);
        g_cuda.memcpy_h2d(d_w, h_w, w_sz * sizeof(float), 0);
        g_cuda.memcpy_h2d(d_bias, h_bias, bias_sz * sizeof(float), 0);

        const void* inputs[]  = {d_in, d_w, d_bias};
        void*       outputs[] = {d_out};
        conv_params_t cp = {.N=N,.C=C,.H=H,.W=W,.K=K,.kernel_h=KH,.kernel_w=KW,
                            .stride_h=stride,.stride_w=stride,.pad_h=pad,.pad_w=pad,
                            .dilation_h=dil,.dilation_w=dil};
        stream_t s = {0};
        int rc = cuda_op->func(inputs, outputs, (const operator_params_t*)&cp, &s);
        g_cuda.stream_synchronize(0);
        fprintf(stderr, "CUDA conv: rc=%d\n", rc);

        g_cuda.memcpy_d2h(h_out_cuda, d_out, out_sz * sizeof(float), 0);
        g_cuda.stream_synchronize(0);

        g_cuda.device_free(d_in); g_cuda.device_free(d_w);
        g_cuda.device_free(d_bias); g_cuda.device_free(d_out);
    }

    /* Compare */
    float max_diff = 0.0f;
    int mismatch = 0;
    int max_idx = -1;
    for (size_t i = 0; i < out_sz; i++) {
        float diff = fabsf(h_out_cpu[i] - h_out_cuda[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = (int)i; }
        if (diff > 1e-3f) mismatch++;
    }
    fprintf(stderr, "Max diff: %.6e at idx=%d (CPU=%.6f CUDA=%.6f)\n",
            max_diff, max_idx,
            max_idx >= 0 ? h_out_cpu[max_idx] : 0.0f,
            max_idx >= 0 ? h_out_cuda[max_idx] : 0.0f);
    fprintf(stderr, "Mismatches >1e-3: %d/%zu\n", mismatch, out_sz);
    fprintf(stderr, "First 10 CPU: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", h_out_cpu[i]);
    fprintf(stderr, "\nFirst 10 CUDA: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.4f ", h_out_cuda[i]);
    fprintf(stderr, "\n");

    free(h_in); free(h_w); free(h_bias);
    free(h_out_cpu); free(h_out_cuda);

    cuda_platform_finalize();
    platform_finalize();
    return (mismatch > 0) ? 1 : 0;
}
