/**
 * Targeted test: compare CPU vs CUDA operators at ResNet scale.
 */
#include "platform.h"
#include "operator.h"
#include "conv_int.h"
#include "pooling_int.h"
#include "add_int.h"
#include "globalavgpool_int.h"
#include "matmul_int.h"
#include "cuda_ops.h"
#include "cuda_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

extern int operator_init_all(void);

static int compare(const float* cpu, const float* cuda, size_t n, const char* name) {
    float max_diff = 0.0f;
    int mismatch = 0;
    int max_idx = -1;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - cuda[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = (int)i; }
        if (diff > 1e-4f) mismatch++;
    }
    fprintf(stderr, "%s: max_diff=%.6e mismatches>1e-4=%d/%zu",
            name, max_diff, mismatch, n);
    if (max_idx >= 0) fprintf(stderr, " [idx=%d CPU=%.6f CUDA=%.6f]", max_idx, cpu[max_idx], cuda[max_idx]);
    fprintf(stderr, "\n");
    return mismatch;
}

int main(void) {
    platform_init();
    operator_init_all();
    cuda_platform_init(0);

    /* ---- MaxPool2D: 64×112×112 → 64×56×56, 3×3 kernel stride=2, pad=1 ---- */
    {
        int64_t N=1, C=64, H=112, W=112, KH=3, KW=3, stride=2, pad=1;
        int64_t OH = (H + 2*pad - KH)/stride + 1;
        int64_t OW = (W + 2*pad - KW)/stride + 1;
        size_t in_sz = N*C*H*W, out_sz = N*C*OH*OW;
        float* h_in = (float*)malloc(in_sz * sizeof(float));
        float* h_out_cpu = (float*)calloc(out_sz, sizeof(float));
        float* h_out_cuda = (float*)calloc(out_sz, sizeof(float));
        srand(100);
        for (size_t i = 0; i < in_sz; i++) h_in[i] = (float)(rand()%2000-1000)/100.0f;

        /* CPU */
        const operator_registry_t* cpu_op = operator_find("maxpool2d_f32");
        {
            const void* inputs[] = {h_in};
            void* outputs[] = {h_out_cpu};
            pool_params_t pp = {.N=N,.C=C,.H=H,.W=W,.kernel_h=KH,.kernel_w=KW,
                               .stride_h=stride,.stride_w=stride,.pad_h=pad,.pad_w=pad};
            cpu_op->func(inputs, outputs, (const operator_params_t*)&pp, NULL);
        }
        /* CUDA */
        const operator_registry_t* cuda_op = operator_find("maxpool2d_f32_cuda");
        {
            float *d_in = (float*)g_cuda.device_alloc(in_sz*sizeof(float));
            float *d_out = (float*)g_cuda.device_alloc(out_sz*sizeof(float));
            g_cuda.memcpy_h2d(d_in, h_in, in_sz*sizeof(float), 0);
            const void* inputs[] = {d_in};
            void* outputs[] = {d_out};
            pool_params_t pp = {.N=N,.C=C,.H=H,.W=W,.kernel_h=KH,.kernel_w=KW,
                               .stride_h=stride,.stride_w=stride,.pad_h=pad,.pad_w=pad};
            stream_t s = {0};
            cuda_op->func(inputs, outputs, (const operator_params_t*)&pp, &s);
            g_cuda.stream_synchronize(0);
            g_cuda.memcpy_d2h(h_out_cuda, d_out, out_sz*sizeof(float), 0);
            g_cuda.stream_synchronize(0);
            g_cuda.device_free(d_in); g_cuda.device_free(d_out);
        }
        compare(h_out_cpu, h_out_cuda, out_sz, "MaxPool2D");
        free(h_in); free(h_out_cpu); free(h_out_cuda);
    }

    /* ---- Add (same shape): 64×56×56 ---- */
    {
        int64_t N=64*56*56;
        size_t sz = N;
        float* h_a = (float*)malloc(sz * sizeof(float));
        float* h_b = (float*)malloc(sz * sizeof(float));
        float* h_out_cpu = (float*)calloc(sz, sizeof(float));
        float* h_out_cuda = (float*)calloc(sz, sizeof(float));
        srand(200);
        for (size_t i = 0; i < sz; i++) { h_a[i] = (float)(rand()%2000-1000)/100.0f; h_b[i] = (float)(rand()%2000-1000)/100.0f; }

        /* CPU */
        const operator_registry_t* cpu_op = operator_find("add_f32");
        {
            const void* inputs[] = {h_a, h_b};
            void* outputs[] = {h_out_cpu};
            add_params_t ap = {.numel = N, .B_numel = N};
            cpu_op->func(inputs, outputs, (const operator_params_t*)&ap, NULL);
        }
        /* CUDA */
        const operator_registry_t* cuda_op = operator_find("add_f32_cuda");
        {
            float *d_a = (float*)g_cuda.device_alloc(sz*sizeof(float));
            float *d_b = (float*)g_cuda.device_alloc(sz*sizeof(float));
            float *d_out = (float*)g_cuda.device_alloc(sz*sizeof(float));
            g_cuda.memcpy_h2d(d_a, h_a, sz*sizeof(float), 0);
            g_cuda.memcpy_h2d(d_b, h_b, sz*sizeof(float), 0);
            const void* inputs[] = {d_a, d_b};
            void* outputs[] = {d_out};
            add_params_t ap = {.numel = N, .B_numel = N};
            stream_t s = {0};
            cuda_op->func(inputs, outputs, (const operator_params_t*)&ap, &s);
            g_cuda.stream_synchronize(0);
            g_cuda.memcpy_d2h(h_out_cuda, d_out, sz*sizeof(float), 0);
            g_cuda.stream_synchronize(0);
            g_cuda.device_free(d_a); g_cuda.device_free(d_b); g_cuda.device_free(d_out);
        }
        compare(h_out_cpu, h_out_cuda, sz, "Add");
        free(h_a); free(h_b); free(h_out_cpu); free(h_out_cuda);
    }

    /* ---- GlobalAvgPool: 512×7×7 ---- */
    {
        int64_t N=1, C=512, H=7, W=7;
        size_t in_sz = N*C*H*W, out_sz = N*C;
        float* h_in = (float*)malloc(in_sz * sizeof(float));
        float* h_out_cpu = (float*)calloc(out_sz, sizeof(float));
        float* h_out_cuda = (float*)calloc(out_sz, sizeof(float));
        srand(300);
        for (size_t i = 0; i < in_sz; i++) h_in[i] = (float)(rand()%2000-1000)/100.0f;

        /* CPU */
        const operator_registry_t* cpu_op = operator_find("globalavgpool_f32");
        {
            const void* inputs[] = {h_in};
            void* outputs[] = {h_out_cpu};
            globalavgpool_params_t gp = {.N=N,.C=C,.H=H,.W=W};
            cpu_op->func(inputs, outputs, (const operator_params_t*)&gp, NULL);
        }
        /* CUDA */
        const operator_registry_t* cuda_op = operator_find("globalavgpool_f32_cuda");
        {
            float *d_in = (float*)g_cuda.device_alloc(in_sz*sizeof(float));
            float *d_out = (float*)g_cuda.device_alloc(out_sz*sizeof(float));
            g_cuda.memcpy_h2d(d_in, h_in, in_sz*sizeof(float), 0);
            const void* inputs[] = {d_in};
            void* outputs[] = {d_out};
            globalavgpool_params_t gp = {.N=N,.C=C,.H=H,.W=W};
            stream_t s = {0};
            cuda_op->func(inputs, outputs, (const operator_params_t*)&gp, &s);
            g_cuda.stream_synchronize(0);
            g_cuda.memcpy_d2h(h_out_cuda, d_out, out_sz*sizeof(float), 0);
            g_cuda.stream_synchronize(0);
            g_cuda.device_free(d_in); g_cuda.device_free(d_out);
        }
        compare(h_out_cpu, h_out_cuda, out_sz, "GlobalAvgPool");
        free(h_in); free(h_out_cpu); free(h_out_cuda);
    }

    /* ---- MatMul: 1×512 * 512×1000 (FC layer) ---- */
    {
        int64_t M=1, N=1000, K=512;
        size_t a_sz=M*K, b_sz=K*N, c_sz=M*N, bias_sz=N;
        float* h_a = (float*)malloc(a_sz*sizeof(float));
        float* h_b = (float*)malloc(b_sz*sizeof(float));
        float* h_bias = (float*)malloc(bias_sz*sizeof(float));
        float* h_out_cpu = (float*)calloc(c_sz, sizeof(float));
        float* h_out_cuda = (float*)calloc(c_sz, sizeof(float));
        srand(400);
        for (size_t i = 0; i < a_sz; i++) h_a[i] = (float)(rand()%2000-1000)/100.0f;
        for (size_t i = 0; i < b_sz; i++) h_b[i] = (float)(rand()%200-100)/100.0f;
        for (size_t i = 0; i < bias_sz; i++) h_bias[i] = (float)(rand()%200-100)/100.0f;

        /* CPU */
        const operator_registry_t* cpu_op = operator_find("matmul_f32");
        {
            const void* inputs[] = {h_a, h_b, h_bias};
            void* outputs[] = {h_out_cpu};
            matmul_params_t mp = {.M=M,.N=N,.K=K};
            cpu_op->func(inputs, outputs, (const operator_params_t*)&mp, NULL);
        }
        /* CUDA */
        const operator_registry_t* cuda_op = operator_find("matmul_f32_cuda");
        {
            float *d_a = (float*)g_cuda.device_alloc(a_sz*sizeof(float));
            float *d_b = (float*)g_cuda.device_alloc(b_sz*sizeof(float));
            float *d_bias = (float*)g_cuda.device_alloc(bias_sz*sizeof(float));
            float *d_out = (float*)g_cuda.device_alloc(c_sz*sizeof(float));
            g_cuda.memcpy_h2d(d_a, h_a, a_sz*sizeof(float), 0);
            g_cuda.memcpy_h2d(d_b, h_b, b_sz*sizeof(float), 0);
            g_cuda.memcpy_h2d(d_bias, h_bias, bias_sz*sizeof(float), 0);
            const void* inputs[] = {d_a, d_b, d_bias};
            void* outputs[] = {d_out};
            matmul_params_t mp = {.M=M,.N=N,.K=K};
            stream_t s = {0};
            cuda_op->func(inputs, outputs, (const operator_params_t*)&mp, &s);
            g_cuda.stream_synchronize(0);
            g_cuda.memcpy_d2h(h_out_cuda, d_out, c_sz*sizeof(float), 0);
            g_cuda.stream_synchronize(0);
            g_cuda.device_free(d_a); g_cuda.device_free(d_b); g_cuda.device_free(d_bias); g_cuda.device_free(d_out);
        }
        compare(h_out_cpu, h_out_cuda, c_sz, "MatMul");
        free(h_a); free(h_b); free(h_bias); free(h_out_cpu); free(h_out_cuda);
    }

    cuda_platform_finalize();
    platform_finalize();
    return 0;
}
