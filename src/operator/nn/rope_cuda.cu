/* CUDA RoPE - Rotary Position Encoding kernel.
   Applies rotation to Q/K tensor in-place.
   Grid: 1D over (S * H * d/2) elements.

   Layout (p->layout):
     INTERLEAVED: adjacent pair (2i, 2i+1)        - GPT-J
     HALFSPLIT:   (i, i+d/2)                      - GPT-NeoX / LLaMA
   Frequency (p->inv_freq):
     non-NULL: use precomputed table [d/2] (host-side scaling variants)
     NULL:     compute from base on the fly (backward compatible) */
#include "operator.h"
#include "cuda_ops.h"
#include "rope_int.h"
#include <math.h>

__global__ void rope_f32_kernel(float* data, int64_t S, int64_t H, int64_t d,
                                 float base, const float* inv_freq, int is_halfsplit) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t half_d = d / 2;
    int64_t total = S * H * half_d;
    if (idx >= total) return;

    int64_t i = idx % half_d;
    int64_t tmp = idx / half_d;
    int64_t h = tmp % H;
    int64_t pos = tmp / H;

    float freq = inv_freq ? inv_freq[i]
                          : (1.0f / powf(base, (float)(2 * i) / (float)d));
    float angle = (float)pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);

    float* q = data + (pos * H + h) * d;
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
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    /* Copy input to output if not in-place */
    if (inputs[0] != outputs[0]) {
        size_t bytes = (size_t)p->seq_len * p->num_heads * p->head_dim * sizeof(float);
        (void)cudaMemcpyAsync(outputs[0], inputs[0], bytes, cudaMemcpyDeviceToDevice, s);
    }

    int64_t total = p->seq_len * p->num_heads * (p->head_dim / 2);
    dim3 block(OPS_THREADS_PER_BLOCK, 1, 1);
    dim3 grid((unsigned int)((total + OPS_THREADS_PER_BLOCK - 1) / OPS_THREADS_PER_BLOCK), 1, 1);

    return CUDA_KERNEL_LAUNCH(rope_f32_kernel, grid, block, 0, s,
                              (float*)outputs[0], p->seq_len, p->num_heads,
                              p->head_dim, p->base, p->inv_freq,
                              (int)(p->layout == ROPE_LAYOUT_HALFSPLIT));
}

extern "C" int register_rope_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "rope_f32_cuda", .data_type = "f32",
        .func = rope_f32_cuda, .version = 2, .flags = OP_FLAG_IN_PLACE,
    };
    return operator_register(&reg);
}
