/* MHA decode CUDA kernel — single-token attention with KV-cache.
   grid = (B, 1, 1), one block per batch element.
   Each block: compute Q/K/V for new token, write K/V to cache,
   compute attention over cache, output projection. */
#include "operator.h"
#include "cuda_ops.h"
#include "mha_decode_int.h"
#include <cuda_fp16.h>

#define DECODE_MAX_D 64

__global__ void mha_decode_f32_kernel(
    const float* __restrict__ X_new,    /* (B, 1, D) */
    const float* __restrict__ K_cache,  /* (B, max_seq, H, d) */
    const float* __restrict__ V_cache,  /* (B, max_seq, H, d) */
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ WK, const float* __restrict__ bK,
    const float* __restrict__ WV, const float* __restrict__ bV,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y,              /* (B, 1, D) */
    float* __restrict__ K_cache_out,    /* (B, max_seq, H, d) — updated */
    float* __restrict__ V_cache_out,    /* (B, max_seq, H, d) — updated */
    int64_t B, int64_t D, int64_t H, int64_t d,
    float scale, int64_t cache_len, int64_t max_seq)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int D_int = (int)D, H_int = (int)H, d_int = (int)d;
    int total_len = (int)(cache_len + 1);

    const float* x = X_new + b * D;
    float* y = Y + b * D;

    /* Initialize Y with output bias */
    for (int j = tid; j < D_int; j += nthreads) {
        y[j] = bO ? bO[j] : 0.0f;
    }
    __syncthreads();

    /* Per-head scratch in shared memory */
    extern __shared__ float smem[];
    float* scores_smem = smem;       /* max_seq floats */
    float* reduce_buf = smem + max_seq; /* nthreads floats for reductions */

    /* Process each head */
    for (int h = 0; h < H_int; h++) {
        int ho = h * d_int;

        /* 1. Compute Q = x · WQ + bQ */
        float Q_reg[DECODE_MAX_D];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) acc += x[j] * WQ[j * D_int + ho + di];
            Q_reg[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
        }

        /* 2. Compute K_new, V_new and write to cache at position cache_len */
        for (int di = tid; di < d_int; di += nthreads) {
            float k_acc = 0.0f, v_acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                k_acc += x[j] * WK[j * D_int + ho + di];
                v_acc += x[j] * WV[j * D_int + ho + di];
            }
            int64_t idx = (b * max_seq + cache_len) * H_int * d_int + ho + di;
            K_cache_out[idx] = k_acc + (bK ? bK[ho + di] : 0.0f);
            V_cache_out[idx] = v_acc + (bV ? bV[ho + di] : 0.0f);
        }
        __syncthreads();

        /* 3. Compute attention scores for positions 0..cache_len */
        float local_max = -1e38f;
        for (int t = tid; t < total_len; t += nthreads) {
            float dot = 0.0f;
            int64_t k_off = (b * max_seq + t) * H_int * d_int + ho;
            for (int di = 0; di < d_int; di++) dot += Q_reg[di] * K_cache_out[k_off + di];
            scores_smem[t] = dot * scale;
            if (scores_smem[t] > local_max) local_max = scores_smem[t];
        }
        __syncthreads();

        /* Reduce max: each thread stores local_max, then tree reduce */
        reduce_buf[tid] = local_max;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (reduce_buf[tid + stride] > reduce_buf[tid])
                    reduce_buf[tid] = reduce_buf[tid + stride];
            }
            __syncthreads();
        }
        /* Broadcast result from reduce_buf[0] to all threads */
        float max_score = reduce_buf[0];
        __syncthreads();

        /* 4. Softmax: exp(score - max) and sum */
        float local_sum = 0.0f;
        for (int t = tid; t < total_len; t += nthreads) {
            scores_smem[t] = expf(scores_smem[t] - max_score);
            local_sum += scores_smem[t];
        }
        __syncthreads();

        /* Reduce sum — reset reduce_buf first to avoid stale max values */
        reduce_buf[tid] = 0.0f;
        __syncthreads();
        reduce_buf[tid] = local_sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }
        /* Broadcast result from reduce_buf[0] to all threads */
        float sum_exp = reduce_buf[0];
        __syncthreads();
        if (sum_exp < 1e-12f) sum_exp = 1e-12f;

        /* 5. Weighted V sum → merged (use shared memory for broadcast) */
        /* Store merged in scores_smem (reused after softmax is done) */
        float merged_local[DECODE_MAX_D] = {0.0f};
        for (int t = tid; t < total_len; t += nthreads) {
            float w = scores_smem[t] / sum_exp;
            int64_t v_off = (b * max_seq + t) * H_int * d_int + ho;
            for (int di = 0; di < d_int; di++) merged_local[di] += w * V_cache_out[v_off + di];
        }
        __syncthreads();

        /* Broadcast merged from thread 0 to all threads via shared memory */
        if (tid == 0) {
            for (int di = 0; di < d_int; di++) scores_smem[di] = merged_local[di];
        }
        __syncthreads();
        float merged[DECODE_MAX_D];
        for (int di = 0; di < d_int; di++) merged[di] = scores_smem[di];
        __syncthreads();

        /* 6. Output projection: Y += merged · WO */
        for (int j = tid; j < D_int; j += nthreads) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++) contrib += merged[di] * WO[(ho + di) * D_int + j];
            atomicAdd(&y[j], contrib);
        }
        __syncthreads();
    }
}

int mha_decode_f32_cuda(const void* inputs[], void* outputs[],
                         const operator_params_t* params, stream_t* stream) {
    if (!params) return -1;
    for (int i = 0; i < 11; i++) if (!inputs[i]) return -1;
    for (int i = 0; i < 3; i++)  if (!outputs[i]) return -1;

    const mha_decode_params_t* p = (const mha_decode_params_t*)params;
    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    dim3 grid((unsigned int)p->batch_size, 1, 1);
    dim3 block(256, 1, 1);

    /* Shared memory: scores(max_seq) + reduce_buf(nthreads) */
    size_t smem_bytes = (size_t)(p->max_seq + 256) * sizeof(float);

    return CUDA_KERNEL_LAUNCH(mha_decode_f32_kernel, grid, block, smem_bytes, s,
        (const float*)inputs[0],   /* X_new */
        (const float*)inputs[1],   /* K_cache */
        (const float*)inputs[2],   /* V_cache */
        (const float*)inputs[3],   /* WQ */
        (const float*)inputs[4],   /* bQ */
        (const float*)inputs[5],   /* WK */
        (const float*)inputs[6],   /* bK */
        (const float*)inputs[7],   /* WV */
        (const float*)inputs[8],   /* bV */
        (const float*)inputs[9],   /* WO */
        (const float*)inputs[10],  /* bO */
        (float*)outputs[0],        /* Y */
        (float*)outputs[1],        /* K_cache_out */
        (float*)outputs[2],        /* V_cache_out */
        p->batch_size, p->hidden_size, p->num_heads, p->head_dim,
        p->scale, p->cache_len, p->max_seq);
}

extern "C" int register_mha_decode_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_decode_f32_cuda", .data_type = "f32",
        .func = mha_decode_f32_cuda, .version = 1, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
