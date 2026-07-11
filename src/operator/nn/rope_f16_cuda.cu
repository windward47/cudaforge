/* FP16 RoPE CUDA kernel.
   FP16 input/output, FP32 compute for sin/cos/rotation (numerical stability).
   Registered as "rope_f16_cuda". Mirrors rope_f32_kernel but with __half I/O.
   Reuses rope_params_t (same layout/inv_freq/batch/pos_offset semantics). */
#include "operator.h"
#include "cuda_ops.h"
#include "rope_int.h"
#include <cuda_fp16.h>
#include <math.h>

/* Max head_dim for shared-memory inv_freq cache (must match rope_cuda.cu). */
#define ROPE_MAX_HALF_D 256

__global__ void rope_f16_kernel(__half* data, int64_t B, int64_t S, int64_t H, int64_t d,
                                 float base, const float* inv_freq, int is_halfsplit,
                                 int64_t pos_offset) {
    int64_t half_d = d / 2;
    int64_t total = B * S * H * half_d;

    /* Cooperatively precompute inv_freq into shared memory (eliminates per-thread powf). */
    __shared__ float s_inv_freq[ROPE_MAX_HALF_D];
    if (half_d <= ROPE_MAX_HALF_D) {
        for (int64_t i = threadIdx.x; i < half_d; i += blockDim.x) {
            s_inv_freq[i] = inv_freq ? inv_freq[i]
                                     : (1.0f / powf(base, (float)(2 * i) / (float)d));
        }
        __syncthreads();
    }

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    /* Decode (i, h, pos, b) from linear idx */
    int64_t i = idx % half_d;
    int64_t tmp = idx / half_d;
    int64_t h = tmp % H;
    tmp = tmp / H;
    int64_t pos = tmp % S;
    int64_t b = tmp / S;

    float freq = (half_d <= ROPE_MAX_HALF_D) ? s_inv_freq[i]
                : (inv_freq ? inv_freq[i]
                            : (1.0f / powf(base, (float)(2 * i) / (float)d)));
    float angle = (float)(pos + pos_offset) * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    __half* q = data + ((b * S + pos) * H + h) * d;
    if (!is_halfsplit) {
        /* INTERLEAVED: adjacent pair (2i, 2i+1) */
        float x0 = __half2float(q[2 * i]);
        float x1 = __half2float(q[2 * i + 1]);
        q[2 * i]     = __float2half(x0 * c - x1 * s);
        q[2 * i + 1] = __float2half(x0 * s + x1 * c);
    } else {
        /* HALFSPLIT: (i, i+d/2) - GPT-NeoX/LLaMA layout */
        float x0 = __half2float(q[i]);
        float x1 = __half2float(q[i + half_d]);
        q[i]            = __float2half(x0 * c - x1 * s);
        q[i + half_d]   = __float2half(x0 * s + x1 * c);
    }
}

int rope_f16_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const rope_params_t* p = (const rope_params_t*)params;
    int64_t B = (p->batch_size > 0) ? p->batch_size : 1;
    int64_t S = p->seq_len, H = p->num_heads, d = p->head_dim;
    int64_t pos_off = (p->pos_offset > 0) ? p->pos_offset : 0;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    /* Copy input to output if not in-place */
    if (inputs[0] != outputs[0]) {
        size_t bytes = (size_t)B * S * H * d * sizeof(__half);
        (void)cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice, s);
    }

    int64_t total = B * S * H * (d / 2);
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(rope_f16_kernel, grid, block, 0, s,
                              (__half*)outputs[0], B, S, H, d, p->base, p->inv_freq,
                              (int)(p->layout == ROPE_LAYOUT_HALFSPLIT), pos_off);
}

extern "C" int register_rope_f16_cuda(void) {
    static operator_registry_t reg = {
        .name = "rope_f16_cuda", .data_type = "f16",
        .func = rope_f16_cuda, .version = 1, .flags = OP_FLAG_IN_PLACE,
    };
    return operator_register(&reg);
}
