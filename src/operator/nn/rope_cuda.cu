/* CUDA RoPE - Rotary Position Encoding kernel.
   Applies rotation to Q/K tensor in-place.
   Grid: 1D over (B * S * H * d/2) elements.

   Layout (p->layout):
     INTERLEAVED: adjacent pair (2i, 2i+1)        - GPT-J
     HALFSPLIT:   (i, i+d/2)                      - GPT-NeoX / LLaMA
   Frequency (p->inv_freq):
     non-NULL: use precomputed table [d/2] (host-side scaling variants)
     NULL:     compute from base on the fly (backward compatible)
   Position (p->pos_offset):
     real_pos = pos + pos_offset, so KV-cache decode (offset=cache_len) works. */
#include "operator.h"
#include "cuda_ops.h"
#include "rope_int.h"
#include <math.h>

/* Max head_dim we support via shared-memory inv_freq cache. head_dim in modern
   LLMs is 64-128 (half_d 32-64); 256 is a safe upper bound. */
#define ROPE_MAX_HALF_D 256

__global__ void rope_f32_kernel(float* data, int64_t B, int64_t S, int64_t H, int64_t d,
                                 float base, const float* inv_freq, int is_halfsplit,
                                 int64_t pos_offset) {
    int64_t half_d = d / 2;
    int64_t total = B * S * H * half_d;

    /* Cooperatively precompute inv_freq into shared memory.
       This eliminates the per-thread powf() (expensive) when inv_freq==NULL,
       and consolidates the table lookup for all threads in the block. */
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

    /* Decode (i, h, pos, b) from linear idx: i fastest, then h, then pos, then b */
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

    float* q = data + ((b * S + pos) * H + h) * d;
    if (!is_halfsplit) {
        /* INTERLEAVED: adjacent pair (2i, 2i+1) */
        float x0 = q[2 * i];
        float x1 = q[2 * i + 1];
        q[2 * i]     = x0 * c - x1 * s;
        q[2 * i + 1] = x0 * s + x1 * c;
    } else {
        /* HALFSPLIT: (i, i+d/2) - GPT-NeoX/LLaMA layout */
        float x0 = q[i];
        float x1 = q[i + half_d];
        q[i]            = x0 * c - x1 * s;
        q[i + half_d]   = x0 * s + x1 * c;
    }
}

int rope_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0] || !params) return -1;

    const rope_params_t* p = (const rope_params_t*)params;
    int64_t B = (p->batch_size > 0) ? p->batch_size : 1;
    int64_t S = p->seq_len, H = p->num_heads, d = p->head_dim;
    int64_t pos_off = (p->pos_offset > 0) ? p->pos_offset : 0;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    /* Copy input to output if not in-place */
    if (inputs[0] != outputs[0]) {
        size_t bytes = (size_t)B * S * H * d * sizeof(float);
        (void)cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice, s);
    }

    int64_t total = B * S * H * (d / 2);
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(rope_f32_kernel, grid, block, 0, s,
                              (float*)outputs[0], B, S, H, d, p->base, p->inv_freq,
                              (int)(p->layout == ROPE_LAYOUT_HALFSPLIT), pos_off);
}

extern "C" int register_rope_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "rope_f32_cuda", .data_type = "f32",
        .func = rope_f32_cuda, .version = 2, .flags = OP_FLAG_IN_PLACE,
    };
    return operator_register(&reg);
}
