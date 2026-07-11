/* MHA decode CUDA kernel - single-token attention with KV-cache.
   Supports GQA (Grouped-Query Attention): H_kv <= H_q.
   grid = (B, 1, 1), one block per batch element.

   RoPE fusion (R9-h): when rope_base > 0, Q and K_new are rotated after
   projection using pos = cache_len. K_new is staged in shared memory,
   rotated pairwise, then written to cache (so cache stores rotated K). */
#include "operator.h"
#include "cuda_ops.h"
#include "mha_decode_int.h"
#include <cuda_fp16.h>
#include <math.h>

#define DECODE_MAX_D 64

/* Apply RoPE in-place to a (d,) vector at position pos (device helper). */
static __device__ void rope_apply_vec_dev(float* q, int d, int pos,
                                          float rope_base, int rope_layout,
                                          const float* inv_freq) {
    int half_d = d / 2;
    int is_halfsplit = (rope_layout == ROPE_LAYOUT_HALFSPLIT);
    for (int i = 0; i < half_d; i++) {
        float freq = inv_freq ? inv_freq[i]
                              : (1.0f / powf(rope_base, (float)(2 * i) / (float)d));
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        if (!is_halfsplit) {
            float x0 = q[2 * i], x1 = q[2 * i + 1];
            q[2 * i]     = x0 * c - x1 * s;
            q[2 * i + 1] = x0 * s + x1 * c;
        } else {
            float x0 = q[i], x1 = q[i + half_d];
            q[i]          = x0 * c - x1 * s;
            q[i + half_d] = x0 * s + x1 * c;
        }
    }
}

__global__ void mha_decode_f32_kernel(
    const float* __restrict__ X_new,    /* (B, 1, D) */
    const float* __restrict__ K_cache,  /* (B, max_seq, H_kv, d) */
    const float* __restrict__ V_cache,  /* (B, max_seq, H_kv, d) */
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ WK, const float* __restrict__ bK,
    const float* __restrict__ WV, const float* __restrict__ bV,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y,              /* (B, 1, D) */
    float* __restrict__ K_cache_out,    /* (B, max_seq, H_kv, d) */
    float* __restrict__ V_cache_out,    /* (B, max_seq, H_kv, d) */
    int64_t B, int64_t D, int64_t H, int64_t H_kv, int64_t d,
    float scale, int64_t cache_len, int64_t max_seq,
    float rope_base, int rope_layout, const float* rope_inv_freq)
{
    int b = blockIdx.x;
    if (b >= B) return;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int D_int = (int)D, H_int = (int)H, d_int = (int)d;
    int H_kv_int = (int)H_kv;
    int kv_dim = H_kv_int * d_int;
    int group_size = H_int / H_kv_int;
    int total_len = (int)(cache_len + 1);

    const float* x = X_new + b * D;
    float* y = Y + b * D;

    int use_rope = (rope_base > 0.0f);

    /* Initialize Y with output bias */
    for (int j = tid; j < D_int; j += nthreads) {
        y[j] = bO ? bO[j] : 0.0f;
    }

    /* Copy input KV-cache to output (in-place update semantics).
       CPU does memcpy; CUDA must do the same so attention at cache_len>0
       reads the previously-cached K/V at positions 0..cache_len-1. */
    {
        int64_t cache_elems = (int64_t)B * max_seq * H_kv_int * d_int;
        for (int64_t i = tid; i < cache_elems; i += nthreads) {
            K_cache_out[i] = K_cache[i];
            V_cache_out[i] = V_cache[i];
        }
    }
    __syncthreads();

    /* Shared memory layout:
       scores_smem[max_seq]  | reduce_buf[nthreads] | K_new_smem[H_kv*d] (RoPE staging) */
    extern __shared__ float smem[];
    float* scores_smem = smem;                          /* max_seq floats */
    float* reduce_buf  = smem + max_seq;                /* nthreads floats */
    float* K_new_smem  = smem + max_seq + nthreads;     /* H_kv*d floats (RoPE) */

    /* Compute K_new, V_new for all KV heads - all threads cooperatively.
       When RoPE is enabled, stage K_new into shared memory first so it can be
       rotated pairwise (each thread only holds one di otherwise). */
    {
        int total_kv = H_kv_int * d_int;
        for (int i = tid; i < total_kv; i += nthreads) {
            int kv_h = i / d_int;
            int di = i % d_int;
            int kv_ho = kv_h * d_int;
            float k_acc = 0.0f, v_acc = 0.0f;
            for (int j = 0; j < D_int; j++) {
                float xj = x[j];
                k_acc += xj * WK[j * kv_dim + kv_ho + di];
                v_acc += xj * WV[j * kv_dim + kv_ho + di];
            }
            float k_val = k_acc + (bK ? bK[kv_ho + di] : 0.0f);
            float v_val = v_acc + (bV ? bV[kv_ho + di] : 0.0f);

            if (use_rope) {
                K_new_smem[i] = k_val;   /* stage for pairwise rotation */
                /* V doesn't get rotated; write directly */
                int64_t idx = (b * max_seq + cache_len) * H_kv_int * d_int + kv_ho + di;
                V_cache_out[idx] = v_val;
            } else {
                int64_t idx = (b * max_seq + cache_len) * H_kv_int * d_int + kv_ho + di;
                K_cache_out[idx] = k_val;
                V_cache_out[idx] = v_val;
            }
        }
    }
    if (use_rope) {
        __syncthreads();
        /* Rotate each KV head's K_new in shared memory, then write to cache.
           One thread per (kv_h) handles the whole d vector (d<=64, fits regs).
           pos = cache_len (the new token's position). */
        for (int kv_h = tid; kv_h < H_kv_int; kv_h += nthreads) {
            int kv_ho = kv_h * d_int;
            rope_apply_vec_dev(K_new_smem + kv_ho, d_int, (int)cache_len,
                               rope_base, rope_layout, rope_inv_freq);
            int64_t cache_idx = (b * max_seq + cache_len) * H_kv_int * d_int + kv_ho;
            for (int di = 0; di < d_int; di++) {
                K_cache_out[cache_idx + di] = K_new_smem[kv_ho + di];
            }
        }
    }
    __syncthreads();

    /* Process each query head */
    for (int h = 0; h < H_int; h++) {
        int ho = h * d_int;
        int kv_h = h / group_size;  /* GQA: map query head -> KV head */
        int kv_ho = kv_h * d_int;

        /* 1. Compute Q = x · WQ + bQ (serial per head, single thread) */
        float Q_reg[DECODE_MAX_D];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++) acc += x[j] * WQ[j * D_int + ho + di];
            Q_reg[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
        }
        /* R9-h: rotate Q before attention (pos = cache_len) */
        if (use_rope) {
            rope_apply_vec_dev(Q_reg, d_int, (int)cache_len,
                               rope_base, rope_layout, rope_inv_freq);
        }

        /* 2. Compute attention scores */
        float local_max = -1e38f;
        for (int t = tid; t < total_len; t += nthreads) {
            float dot = 0.0f;
            int64_t k_off = (b * max_seq + t) * H_kv_int * d_int + kv_ho;
            for (int di = 0; di < d_int; di++) dot += Q_reg[di] * K_cache_out[k_off + di];
            scores_smem[t] = dot * scale;
            if (scores_smem[t] > local_max) local_max = scores_smem[t];
        }
        __syncthreads();

        /* Reduce max */
        reduce_buf[tid] = local_max;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (reduce_buf[tid + stride] > reduce_buf[tid])
                    reduce_buf[tid] = reduce_buf[tid + stride];
            }
            __syncthreads();
        }
        float max_score = reduce_buf[0];
        __syncthreads();

        /* 3. Softmax: exp(score - max) and sum */
        float local_sum = 0.0f;
        for (int t = tid; t < total_len; t += nthreads) {
            scores_smem[t] = expf(scores_smem[t] - max_score);
            local_sum += scores_smem[t];
        }
        __syncthreads();

        reduce_buf[tid] = 0.0f;
        __syncthreads();
        reduce_buf[tid] = local_sum;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce_buf[tid] += reduce_buf[tid + stride];
            __syncthreads();
        }
        float sum_exp = reduce_buf[0];
        __syncthreads();
        if (sum_exp < 1e-12f) sum_exp = 1e-12f;

        /* 4. Weighted V sum -> merged (reduce across threads).
           Each thread accumulates its own merged_local for the t values it
           processed; these partial sums must be combined across threads
           (when total_len > nthreads_per_token, multiple threads contribute).
           Use scores_smem (no longer needed) as a d-sized accumulator. */
        float merged_local[DECODE_MAX_D] = {0.0f};
        for (int t = tid; t < total_len; t += nthreads) {
            float w = scores_smem[t] / sum_exp;
            int64_t v_off = (b * max_seq + t) * H_kv_int * d_int + kv_ho;
            for (int di = 0; di < d_int; di++) merged_local[di] += w * V_cache_out[v_off + di];
        }
        __syncthreads();

        /* Accumulate partial sums from all threads into scores_smem[0..d-1] */
        if (tid < d_int) scores_smem[tid] = 0.0f;
        __syncthreads();
        for (int di = 0; di < d_int; di++) {
            atomicAdd(&scores_smem[di], merged_local[di]);
        }
        __syncthreads();

        float merged[DECODE_MAX_D];
        for (int di = 0; di < d_int; di++) merged[di] = scores_smem[di];
        __syncthreads();

        /* 5. Output projection: Y += merged · WO
           No atomicAdd needed: threads handle disjoint j values,
           and __syncthreads() between heads ensures sequential accumulation. */
        for (int j = tid; j < D_int; j += nthreads) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++) contrib += merged[di] * WO[(ho + di) * D_int + j];
            y[j] += contrib;
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

    /* Shared memory: scores[max_seq] + reduce[256] + K_new_staging[H_kv*d] (RoPE) */
    int64_t H_kv = p->num_kv_heads > 0 ? p->num_kv_heads : p->num_heads;
    size_t k_staging = (p->rope_base > 0.0f) ? (size_t)H_kv * p->head_dim * sizeof(float) : 0;
    size_t smem_bytes = (size_t)(p->max_seq + 256) * sizeof(float) + k_staging;

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
        p->batch_size, p->hidden_size, p->num_heads, p->num_kv_heads, p->head_dim,
        p->scale, p->cache_len, p->max_seq,
        p->rope_base, p->rope_layout, p->rope_inv_freq);
}

extern "C" int register_mha_decode_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_decode_f32_cuda", .data_type = "f32",
        .func = mha_decode_f32_cuda, .version = 2, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
