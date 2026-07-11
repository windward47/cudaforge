/* FP16 CPU operators — convert to FP32, delegate to optimized FP32 kernel, convert back.
 * Halves memory bandwidth (critical for APU unified memory) while using full-precision compute.
 */

#include "operator.h"
#include "fp16.h"
#include "matmul_int.h"
#include "add_int.h"
#include "rope_int.h"
#include <stdlib.h>
#include <string.h>

/* rope_f32 (CPU) - delegated to by rope_f16_cpu below */
extern int rope_f32(const void* inputs[], void* outputs[],
                    const operator_params_t* params, stream_t* stream);

/* --------------------------------------------------------------------------
 * matmul_f16: FP16 A,B → FP32 matmul → FP16 C
 * -------------------------------------------------------------------------- */
static int matmul_f16_cpu(const void* inputs[], void* outputs[],
                          const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const matmul_params_t* p = (const matmul_params_t*)params;
    const fp16_t* A16    = (const fp16_t*)inputs[0];
    const fp16_t* B16    = (const fp16_t*)inputs[1];
    const fp16_t* bias16 = (const fp16_t*)inputs[2];
    fp16_t* C16          = (fp16_t*)outputs[0];

    int64_t M = p->M, N = p->N, K = p->K;
    int64_t batch_size = p->batch_size > 0 ? p->batch_size : 1;
    int64_t stride_a = p->stride_a ? p->stride_a : M * K;
    int64_t stride_b = p->stride_b ? p->stride_b : K * N;
    int64_t stride_c = p->stride_c ? p->stride_c : M * N;

    float* A32 = (float*)malloc((size_t)M * K * sizeof(float));
    float* B32 = (float*)malloc((size_t)K * N * sizeof(float));
    float* C32 = (float*)malloc((size_t)M * N * sizeof(float));
    float* bias32 = bias16 ? (float*)malloc((size_t)N * sizeof(float)) : NULL;
    if (!A32 || !B32 || !C32 || (bias16 && !bias32)) {
        free(A32); free(B32); free(C32); free(bias32);
        return -1;
    }

    if (bias16) fp16_to_fp32_buf(bias32, bias16, N);

    for (int64_t bi = 0; bi < batch_size; bi++) {
        fp16_to_fp32_buf(A32, A16 + bi * stride_a, M * K);
        fp16_to_fp32_buf(B32, B16 + bi * stride_b, K * N);

        const void* f32_inputs[] = {A32, B32, bias32};
        void* f32_outputs[] = {C32};
        matmul_params_t mp = {
            .M = M, .N = N, .K = K,
            .transpose_a = p->transpose_a,
            .transpose_b = p->transpose_b,
            .batch_size = 1, .stride_a = 0, .stride_b = 0, .stride_c = 0,
        };

        const operator_registry_t* mm = operator_find("matmul_f32");
        int ret = mm ? mm->func(f32_inputs, f32_outputs,
                                (const operator_params_t*)&mp, stream) : -1;
        if (ret != 0) {
            free(A32); free(B32); free(C32); free(bias32);
            return ret;
        }

        fp32_to_fp16_buf(C16 + bi * stride_c, C32, M * N);
    }

    free(A32); free(B32); free(C32); free(bias32);
    return 0;
}

static const operator_registry_t s_matmul_f16_reg = {
    .name      = "matmul_f16",
    .data_type = "f16",
    .func      = matmul_f16_cpu,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

/* --------------------------------------------------------------------------
 * add_f16: FP16 element-wise add
 * -------------------------------------------------------------------------- */
static int add_f16_cpu(const void* inputs[], void* outputs[],
                       const operator_params_t* params, stream_t* stream) {
    (void)stream;
    if (!inputs || !inputs[0] || !inputs[1] || !outputs || !outputs[0])
        return -1;
    if (!params) return -1;

    const add_params_t* p = (const add_params_t*)params;
    const fp16_t* a16 = (const fp16_t*)inputs[0];
    const fp16_t* b16 = (const fp16_t*)inputs[1];
    fp16_t* out16     = (fp16_t*)outputs[0];

    int64_t N  = p->numel;
    int64_t BN = p->B_numel;

    if (BN == 1) {
        float bv = fp16_to_fp32(b16[0]);
        for (int64_t i = 0; i < N; i++)
            out16[i] = fp32_to_fp16(fp16_to_fp32(a16[i]) + bv);
    } else {
        int64_t blocks = N / BN;
        for (int64_t blk = 0; blk < blocks; blk++) {
            for (int64_t i = 0; i < BN; i++) {
                int64_t idx = blk * BN + i;
                out16[idx] = fp32_to_fp16(fp16_to_fp32(a16[idx]) + fp16_to_fp32(b16[i]));
            }
        }
    }
    return 0;
}

static const operator_registry_t s_add_f16_reg = {
    .name      = "add_f16",
    .data_type = "f16",
    .func      = add_f16_cpu,
    .version   = 1,
    .flags     = OP_FLAG_NONE,
};

/* --------------------------------------------------------------------------
 * relu_f16: FP16 ReLU
 * -------------------------------------------------------------------------- */
static int relu_f16_cpu(const void* inputs[], void* outputs[],
                        const operator_params_t* params, stream_t* stream) {
    (void)params;
    (void)stream;
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;

    const fp16_t* in = (const fp16_t*)inputs[0];
    fp16_t* out      = (fp16_t*)outputs[0];
    int64_t n        = *(const int64_t*)inputs[1];

    for (int64_t i = 0; i < n; i++) {
        float v = fp16_to_fp32(in[i]);
        out[i] = fp32_to_fp16(v > 0.0f ? v : 0.0f);
    }
    return 0;
}

static const operator_registry_t s_relu_f16_reg = {
    .name      = "relu_f16",
    .data_type = "f16",
    .func      = relu_f16_cpu,
    .version   = 1,
    .flags     = OP_FLAG_IN_PLACE,
};

/* --------------------------------------------------------------------------
 * rope_f16: FP16 RoPE - convert to FP32, delegate to rope_f32, convert back.
 * Reuses the full-precision rotation (sin/cos/inv_freq logic); FP16 only
 * halves memory bandwidth, compute stays FP32 for numerical stability.
 * -------------------------------------------------------------------------- */
static int rope_f16_cpu(const void* inputs[], void* outputs[],
                        const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const rope_params_t* p = (const rope_params_t*)params;
    int64_t S = p->seq_len, d = p->head_dim, H = p->num_heads;
    int64_t B = (p->batch_size > 0) ? p->batch_size : 1;
    int64_t total = B * S * H * d;

    const fp16_t* in16 = (const fp16_t*)inputs[0];
    fp16_t* out16      = (fp16_t*)outputs[0];

    float* buf32 = (float*)malloc((size_t)total * sizeof(float));
    if (!buf32) return -1;
    fp16_to_fp32_buf(buf32, in16, total);

    /* In-place on the FP32 buffer */
    const void* f32_in[] = { buf32 };
    void* f32_out[] = { buf32 };
    int ret = rope_f32(f32_in, f32_out, params, stream);
    if (ret == 0) {
        fp32_to_fp16_buf(out16, buf32, total);
    }
    free(buf32);
    return ret;
}

static const operator_registry_t s_rope_f16_reg = {
    .name      = "rope_f16",
    .data_type = "f16",
    .func      = rope_f16_cpu,
    .version   = 1,
    .flags     = OP_FLAG_IN_PLACE,
};

/* --------------------------------------------------------------------------
 * Registration
 * -------------------------------------------------------------------------- */
int register_matmul_f16_cpu(void) { return operator_register(&s_matmul_f16_reg); }
int register_add_f16_cpu(void)    { return operator_register(&s_add_f16_reg); }
int register_relu_f16_cpu(void)   { return operator_register(&s_relu_f16_reg); }
int register_rope_f16_cpu(void)   { return operator_register(&s_rope_f16_reg); }

int register_fp16_cpu_ops(void) {
    int ret = 0;
    ret += register_matmul_f16_cpu();
    ret += register_add_f16_cpu();
    ret += register_relu_f16_cpu();
    ret += register_rope_f16_cpu();
    return ret;
}
