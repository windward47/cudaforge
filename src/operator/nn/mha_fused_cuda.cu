#include "operator.h"
#include "cuda_ops.h"
#include "mha_fused_int.h"
#include "cuda_reduce.cuh"
#include <cuda_fp16.h>
#include <mma.h>
#include <math.h>

/* ============================================================
 * Hybrid MHA Dispatch:
 *   S <= 64:    Preloaded K/V kernel (all K/V in smem)
 *   64 < S <= 512: Flash Attention v2 kernel (single pass)
 *   S > 512:    Flash Attention v2 Split-KV kernel
 *
 * Reference: flash-attention-main/csrc/flash_attn/src/flash_fwd_kernel.h
 * ============================================================ */

#define LOG2E_F 1.44269504088896341f

/* Threshold for enabling Split-KV */
#define FA_SPLITKV_THRESHOLD 512

/* ============================================================
 * Kernel 0a: Pre-compute K, V for all positions.
 * K_buf[b, sj, kv_h, dk] = X[b,sj,:] · WK[:, kv_h*d+dk] + bK[...]
 * V_buf[b, sj, kv_h, dk] = X[b,sj,:] · WV[:, kv_h*d+dk] + bV[...]
 * Grid: (B*S, H_kv, 1), Block: (256)
 * ============================================================ */
__global__ void mha_precompute_kv_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WK, const float* __restrict__ bK,
    const float* __restrict__ WV, const float* __restrict__ bV,
    float* __restrict__ K_buf, float* __restrict__ V_buf,
    int64_t B, int64_t S, int64_t D, int64_t H_kv, int64_t d)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int sj = bs % (int)S;
    const int kv_h = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= (int)B || kv_h >= (int)H_kv) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int kv_ho = kv_h * d_int;
    const float* x = X + ((int64_t)b * S_int + sj) * D_int;
    float* k_out = K_buf + (((int64_t)b * S_int + sj) * (int)H_kv + kv_h) * d_int;
    float* v_out = V_buf + (((int64_t)b * S_int + sj) * (int)H_kv + kv_h) * d_int;

    for (int dk = tid; dk < d_int; dk += 256) {
        float k_acc = 0.0f, v_acc = 0.0f;
        for (int j = 0; j < D_int; j++) {
            float xv = x[j];
            k_acc += xv * WK[j * D_int + kv_ho + dk];
            v_acc += xv * WV[j * D_int + kv_ho + dk];
        }
        k_out[dk] = k_acc + (bK ? bK[kv_ho + dk] : 0.0f);
        v_out[dk] = v_acc + (bV ? bV[kv_ho + dk] : 0.0f);
    }
}

/* ============================================================
 * Kernel 0b: Pre-compute Q for all positions.
 * Q_buf[b, sj, h, dk] = X[b,sj,:] · WQ[:, h*d+dk] + bQ[h*d+dk]
 * Grid: (B*S, H_q, 1), Block: (256)
 * ============================================================ */
__global__ void mha_precompute_q_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    float* __restrict__ Q_buf,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t d)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int sj = bs % (int)S;
    const int h  = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= (int)B || h >= (int)H_q) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int ho = h * d_int;
    const float* x = X + ((int64_t)b * S_int + sj) * D_int;
    float* q_out = Q_buf + (((int64_t)b * S_int + sj) * (int)H_q + h) * d_int;

    for (int dk = tid; dk < d_int; dk += 256) {
        float acc = 0.0f;
        for (int j = 0; j < D_int; j++)
            acc += x[j] * WQ[j * D_int + ho + dk];
        q_out[dk] = acc + (bQ ? bQ[ho + dk] : 0.0f);
    }
}


/* ============================================================
 * Kernel 1: Preloaded K/V (fast for short sequences S <= 64)
 * ============================================================ */
__global__ void mha_preloaded_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ WK, const float* __restrict__ bK,
    const float* __restrict__ WV, const float* __restrict__ bV,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y, const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int si = bs % (int)S;
    const int tid = threadIdx.x;
    const int NT = blockDim.x;
    if (b >= (int)B) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int H_q_int = (int)H_q, H_kv_int = (int)H_kv;
    const int group_size = H_q_int / H_kv_int;
    const int H_d = H_kv_int * d_int;

    const float* X_b  = X + (int64_t)b * S_int * D_int;
    float*       Y_bs = Y + ((int64_t)b * S_int + si) * D_int;
    const float* R_bs = (has_residual && R) ? (R + ((int64_t)b * S_int + si) * D_int) : NULL;

    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + S_int * H_d;

    for (int i = tid; i < S_int * H_d; i += NT) {
        int dk = i % d_int;
        int tmp = i / d_int;
        int h  = tmp % H_kv_int;
        int sj = tmp / H_kv_int;
        int kv_ho = h * d_int;
        float k_acc = 0.0f, v_acc = 0.0f;
        for (int j = 0; j < D_int; j++) {
            float xv = X_b[sj * D_int + j];
            k_acc += xv * WK[j * D_int + kv_ho + dk];
            v_acc += xv * WV[j * D_int + kv_ho + dk];
        }
        K_smem[i] = k_acc + (bK ? bK[kv_ho + dk] : 0.0f);
        V_smem[i] = v_acc + (bV ? bV[kv_ho + dk] : 0.0f);
    }
    __syncthreads();

    for (int j = tid; j < D_int; j += NT) {
        float val = (bO ? bO[j] : 0.0f);
        if (R_bs) val += R_bs[j];
        Y_bs[j] = val;
    }
    __syncthreads();

    for (int h = 0; h < H_q_int; h++) {
        int ho = h * d_int;
        int kv_h = h / group_size;
        int kv_ho = kv_h * d_int;

        float Q_reg[FA_MAX_D];
        for (int di = 0; di < d_int; di++) {
            float acc = 0.0f;
            for (int j = 0; j < D_int; j++)
                acc += X_b[si * D_int + j] * WQ[j * D_int + ho + di];
            Q_reg[di] = acc + (bQ ? bQ[ho + di] : 0.0f);
        }

        float max_val = -1e38f, sum_val = 0.0f;
        float out_acc[FA_MAX_D] = {0.0f};
        int causal_limit = causal ? (si + 1) : S_int;
        int num_tiles = (causal_limit + FA_TILE_BN - 1) / FA_TILE_BN;

        for (int kt = 0; kt < num_tiles; kt++) {
            int sj0 = kt * FA_TILE_BN;
            int tile_s = min(FA_TILE_BN, causal_limit - sj0);
            float tile_max = -1e38f;
            float scores[FA_TILE_BN];

            for (int sk = 0; sk < tile_s; sk++) {
                float dot = 0.0f;
                int k_idx = (sj0 + sk) * H_d + kv_ho;
                for (int di = 0; di < d_int; di++)
                    dot += Q_reg[di] * K_smem[k_idx + di];
                scores[sk] = dot * scale;
                if (scores[sk] > tile_max) tile_max = scores[sk];
            }

            if (tile_max > max_val) {
                float rescale = __expf(max_val - tile_max);
                sum_val *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[di] *= rescale;
                max_val = tile_max;
            }

            for (int sk = 0; sk < tile_s; sk++) {
                float p = __expf(scores[sk] - max_val);
                sum_val += p;
                int v_idx = (sj0 + sk) * H_d + kv_ho;
                for (int di = 0; di < d_int; di++)
                    out_acc[di] += p * V_smem[v_idx + di];
            }
        }

        if (sum_val < 1e-12f) sum_val = 1e-12f;
        float inv_sum = 1.0f / sum_val;
        for (int j = tid; j < D_int; j += NT) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++)
                contrib += out_acc[di] * inv_sum * WO[(ho + di) * D_int + j];
            Y_bs[j] += contrib;
        }
        __syncthreads();
    }
}


/* ============================================================
 * Kernel 2: Flash Attention v2 — Multi-row Q Tiling
 *
 * Each block processes FA_TILE_BM query rows for one head.
 * 4 warps, each handles FA_TILE_BM/4 = 16 rows of Q.
 * K/V shared across all warps — no inter-warp communication.
 *
 * Grid:  (ceil(S/FA_TILE_BM), B, H_q)
 * Block: (FA_NUM_THREADS=128, 4 warps)
 *
 * Reference: FA2 Section 3.3 — warp partitioning
 * ============================================================ */
__global__ void mha_flash_attn_v2_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ K_buf, const float* __restrict__ V_buf,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y, const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal)
{
    const int m_block = blockIdx.x;  /* Q tile index */
    const int b       = blockIdx.y;  /* batch */
    const int h       = blockIdx.z;  /* head */
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (b >= (int)B || h >= (int)H_q) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int H_kv_int = (int)H_kv;
    const int group_size = (int)H_q / H_kv_int;
    const int num_warps = FA_NUM_THREADS / 32;
    const int ho = h * d_int;
    const int kv_h = h / group_size;
    const int kv_ho = kv_h * d_int;

    /* Each warp handles ROWS_PER_WARP rows of Q */
    constexpr int ROWS_PER_WARP = FA_TILE_BM / FA_NUM_WARPS;  /* 64/4 = 16 */
    const int warp_row_start = m_block * FA_TILE_BM + warp_id * ROWS_PER_WARP;

    const float* X_b  = X + (int64_t)b * S_int * D_int;
    const float* R_b  = (has_residual && R) ? (R + (int64_t)b * S_int * D_int) : NULL;
    const int64_t kv_batch_stride = (int64_t)S_int * H_kv_int * d_int;

    extern __shared__ float smem[];
    const int smem_stride = d_int + FA_SKEW;
    float* K_smem  = smem;
    float* V_smem  = K_smem + FA_TILE_BN * smem_stride;
    float* red_smem = V_smem + FA_TILE_BN * smem_stride;

    /* ---- Load Q tile for this warp's rows into registers ---- */
    float Q_reg[ROWS_PER_WARP][FA_MAX_D];
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = warp_row_start + r;
        if (global_row < S_int) {
            for (int di = 0; di < d_int; di++) {
                float acc = 0.0f;
                for (int j = 0; j < D_int; j++)
                    acc += X_b[global_row * D_int + j] * WQ[j * D_int + ho + di];
                Q_reg[r][di] = acc + (bQ ? bQ[ho + di] : 0.0f);
            }
        } else {
            for (int di = 0; di < d_int; di++) Q_reg[r][di] = 0.0f;
        }
    }

    /* ---- Per-row online softmax accumulators ---- */
    float row_max[ROWS_PER_WARP], row_sum[ROWS_PER_WARP];
    float out_acc[ROWS_PER_WARP][FA_MAX_D];
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_max[r] = -1e38f;
        row_sum[r] = 0.0f;
        for (int di = 0; di < d_int; di++) out_acc[r][di] = 0.0f;
    }

    /* Causal limit: max across all rows in this warp */
    int max_row = min(warp_row_start + ROWS_PER_WARP, S_int) - 1;
    int causal_limit = causal ? min(max_row + 1, S_int) : S_int;
    int num_tiles = (causal_limit + FA_TILE_BN - 1) / FA_TILE_BN;

    const float* K_base = K_buf + (int64_t)b * kv_batch_stride + kv_ho;
    const float* V_base = V_buf + (int64_t)b * kv_batch_stride + kv_ho;

    for (int kt = 0; kt < num_tiles; kt++) {
        int kv_start = kt * FA_TILE_BN;
        int tile_n = min(FA_TILE_BN, causal_limit - kv_start);

        /* ---- Load K, V tiles (all warps cooperate) ---- */
        if (d_int % 4 == 0) {
            const int d4 = d_int / 4;
            for (int i = tid; i < tile_n * d4; i += FA_NUM_THREADS) {
                int sk = i / d4, dk4 = i % d4;
                int64_t g_off = (int64_t)(kv_start + sk) * H_kv_int * d_int + dk4 * 4;
                int s_off = sk * smem_stride + dk4 * 4;
                *reinterpret_cast<int4*>(&K_smem[s_off]) = *reinterpret_cast<const int4*>(&K_base[g_off]);
                *reinterpret_cast<int4*>(&V_smem[s_off]) = *reinterpret_cast<const int4*>(&V_base[g_off]);
            }
        } else {
            for (int i = tid; i < tile_n * d_int; i += FA_NUM_THREADS) {
                int sk = i / d_int, dk = i % d_int;
                int64_t off = (int64_t)(kv_start + sk) * H_kv_int * d_int + dk;
                K_smem[sk * smem_stride + dk] = K_base[off];
                V_smem[sk * smem_stride + dk] = V_base[off];
            }
        }
        __syncthreads();

        /* ---- Each warp computes scores for its rows independently ---- */
        for (int wr = 0; wr < ROWS_PER_WARP; wr++) {
            int global_row = warp_row_start + wr;
            if (global_row >= S_int) continue;

            /* Compute scores = Q[wr] · K^T */
            float tile_max = -1e38f;
            int my_count = 0;
            float my_scores[FA_TILE_BN / FA_NUM_THREADS + 1];
            int my_indices[FA_TILE_BN / FA_NUM_THREADS + 1];

            for (int sk = lane_id; sk < tile_n; sk += 32) {
                float dot = 0.0f;
                for (int di = 0; di < d_int; di++)
                    dot += Q_reg[wr][di] * K_smem[sk * smem_stride + di];
                float s = dot * scale;
                if (causal && (kv_start + sk) > global_row) s = -1e38f;
                my_scores[my_count] = s;
                my_indices[my_count] = sk;
                if (s > tile_max) tile_max = s;
                my_count++;
            }

            /* Warp-level max reduction (no smem needed) */
            float warp_max = warp_reduce_max(tile_max);

            /* Online softmax rescale */
            if (warp_max > row_max[wr]) {
                float rescale = __expf(row_max[wr] - warp_max);
                row_sum[wr] *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[wr][di] *= rescale;
                row_max[wr] = warp_max;
            }

            /* Accumulate exp(scores - max) * V */
            float local_sum = 0.0f;
            float local_out[FA_MAX_D];
            for (int di = 0; di < d_int; di++) local_out[di] = 0.0f;

            for (int i = 0; i < my_count; i++) {
                float p = exp2f((my_scores[i] - row_max[wr]) * LOG2E_F);
                local_sum += p;
                int sk = my_indices[i];
                for (int di = 0; di < d_int; di++)
                    local_out[di] += p * V_smem[sk * smem_stride + di];
            }

            /* Warp-level sum reduction */
            row_sum[wr] += warp_reduce_sum(local_sum);
            for (int di = 0; di < d_int; di++)
                out_acc[wr][di] += warp_reduce_sum(local_out[di]);
        }
        __syncthreads();
    }

    /* ---- Normalize and write output for this warp's rows ---- */
    for (int wr = 0; wr < ROWS_PER_WARP; wr++) {
        int global_row = warp_row_start + wr;
        if (global_row >= S_int) continue;

        float inv_sum = (row_sum[wr] < 1e-12f) ? 1.0f : (1.0f / row_sum[wr]);
        float* Y_row = Y + ((int64_t)b * S_int + global_row) * D_int;

        /* Output projection: Y += out_acc · WO
         * Each lane handles a subset of D columns */
        for (int j = lane_id; j < D_int; j += 32) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++)
                contrib += out_acc[wr][di] * inv_sum * WO[(ho + di) * D_int + j];
            float val = contrib;
            if (h == 0) {
                val += (bO ? bO[j] : 0.0f);
                if (R_b) val += R_b[global_row * D_int + j];
            }
            atomicAdd(&Y_row[j], val);
        }
    }
}


/* ============================================================
 * Kernel 2b: Flash Attention v2 with FP16 Tensor Core (WMMA)
 *
 * Multi-row Q tiling (BM=64) + WMMA 16×16 for Q·Kᵀ and P·V.
 * Softmax remains FP32 for numerical stability.
 *
 * Grid:  (ceil(S/FA_TILE_BM), B, H_q)
 * Block: (FA_NUM_THREADS=128, 4 warps)
 * Each warp handles ROWS_PER_WARP = 16 rows of Q.
 *
 * Reference:
 *   cudaTensorCoreGemm.cu — WMMA GEMM pattern
 *   flash_fwd_kernel.h:319 — FA2 warp split-Q
 * ============================================================ */
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void mha_flash_attn_v2_f16_kernel(
    const float* __restrict__ X,
    const float* __restrict__ WQ, const float* __restrict__ bQ,
    const float* __restrict__ K_buf, const float* __restrict__ V_buf,
    const float* __restrict__ WO, const float* __restrict__ bO,
    float* __restrict__ Y, const float* __restrict__ R,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal)
{
    using namespace nvcuda;

    const int m_block = blockIdx.x;  /* Q tile index */
    const int b       = blockIdx.y;  /* batch */
    const int h       = blockIdx.z;  /* head */
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (b >= (int)B || h >= (int)H_q) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int H_kv_int = (int)H_kv;
    const int group_size = (int)H_q / H_kv_int;
    const int ho = h * d_int;
    const int kv_h = h / group_size;
    const int kv_ho = kv_h * d_int;

    constexpr int ROWS_PER_WARP = FA_TILE_BM / FA_NUM_WARPS;  /* 16 */
    const int warp_row_start = m_block * FA_TILE_BM + warp_id * ROWS_PER_WARP;

    const float* X_b  = X + (int64_t)b * S_int * D_int;
    const float* R_b  = (has_residual && R) ? (R + (int64_t)b * S_int * D_int) : NULL;
    const int64_t kv_batch_stride = (int64_t)S_int * H_kv_int * d_int;

    /* Shared memory: K/V as FP16 + P as FP16 */
    extern __shared__ __half smem_h[];
    const int smem_stride = d_int + FA_SKEW;
    __half* K_hsmem = smem_h;                                      /* FA_TILE_BN × smem_stride */
    __half* V_hsmem = K_hsmem + FA_TILE_BN * smem_stride;          /* FA_TILE_BN × smem_stride */
    __half* P_hsmem = V_hsmem + FA_TILE_BN * smem_stride;          /* ROWS_PER_WARP × smem_stride */

    /* ---- Load Q for this warp's rows (FP32 → FP16 in registers) ---- */
    __half Q_h[ROWS_PER_WARP][FA_MAX_D];
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = warp_row_start + r;
        if (global_row < S_int) {
            for (int di = 0; di < d_int; di++) {
                float acc = 0.0f;
                for (int j = 0; j < D_int; j++)
                    acc += X_b[global_row * D_int + j] * WQ[j * D_int + ho + di];
                Q_h[r][di] = __float2half(acc + (bQ ? bQ[ho + di] : 0.0f));
            }
        } else {
            for (int di = 0; di < d_int; di++) Q_h[r][di] = __float2half(0.0f);
        }
    }

    /* ---- Per-row online softmax accumulators (FP32) ---- */
    float row_max[ROWS_PER_WARP], row_sum[ROWS_PER_WARP];
    float out_acc[ROWS_PER_WARP][FA_MAX_D];
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_max[r] = -1e38f;
        row_sum[r] = 0.0f;
        for (int di = 0; di < d_int; di++) out_acc[r][di] = 0.0f;
    }

    int max_row = min(warp_row_start + ROWS_PER_WARP, S_int) - 1;
    int causal_limit = causal ? min(max_row + 1, S_int) : S_int;
    int num_tiles = (causal_limit + FA_TILE_BN - 1) / FA_TILE_BN;
    const float* K_base = K_buf + (int64_t)b * kv_batch_stride + kv_ho;
    const float* V_base = V_buf + (int64_t)b * kv_batch_stride + kv_ho;

    for (int kt = 0; kt < num_tiles; kt++) {
        int kv_start = kt * FA_TILE_BN;
        int tile_n = min(FA_TILE_BN, causal_limit - kv_start);

        /* ---- Load K tile as FP16 ---- */
        for (int i = tid; i < tile_n * d_int; i += FA_NUM_THREADS) {
            int sk = i / d_int, dk = i % d_int;
            int64_t off = (int64_t)(kv_start + sk) * H_kv_int * d_int + dk;
            K_hsmem[sk * smem_stride + dk] = __float2half(K_base[off]);
        }
        __syncthreads();

        /* ---- Compute scores = Q · Kᵀ using WMMA ----
         * Q_h is (ROWS_PER_WARP, d), K_hsmem is (tile_n, d).
         * scores = Q · Kᵀ → (ROWS_PER_WARP, tile_n)
         * Only 1 WMMA tile needed: 16×16 × 16×tile_n → 16×tile_n */
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_s;
        wmma::fill_fragment(acc_s, 0.0f);

        for (int k_step = 0; k_step < d_int; k_step += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;

            /* Load Q fragment from registers → smem → WMMA */
            /* We need Q in smem for WMMA load. Use P_hsmem as temp. */
            for (int r = 0; r < ROWS_PER_WARP; r++)
                for (int di = tid; di < WMMA_K && k_step + di < d_int; di += FA_NUM_THREADS)
                    P_hsmem[r * smem_stride + di] = Q_h[r][k_step + di];
            __syncthreads();

            wmma::load_matrix_sync(a_frag, P_hsmem, smem_stride);
            wmma::load_matrix_sync(b_frag, K_hsmem + k_step, smem_stride);
            wmma::mma_sync(acc_s, a_frag, b_frag, acc_s);
            __syncthreads();
        }

        /* ---- Store scores to FP32, apply scale + causal mask ---- */
        float scores_local[WMMA_M * WMMA_N];
        wmma::store_matrix_sync(scores_local, acc_s, WMMA_N, wmma::mem_row_major);

        float tile_max = -1e38f;
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int global_row = warp_row_start + r;
            if (global_row >= S_int) continue;
            for (int c = 0; c < tile_n; c++) {
                float s = scores_local[r * WMMA_N + c] * scale;
                if (causal && (kv_start + c) > global_row) s = -1e38f;
                scores_local[r * WMMA_N + c] = s;
                if (s > tile_max) tile_max = s;
            }
        }

        /* Warp-level max reduction */
        float warp_max = warp_reduce_max(tile_max);

        /* Online softmax rescale */
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            if (warp_max > row_max[r]) {
                float rescale = __expf(row_max[r] - warp_max);
                row_sum[r] *= rescale;
                for (int di = 0; di < d_int; di++) out_acc[r][di] *= rescale;
                row_max[r] = warp_max;
            }
        }

        /* Compute exp(scores - max) and accumulate sum */
        float local_sum[ROWS_PER_WARP];
        for (int r = 0; r < ROWS_PER_WARP; r++) local_sum[r] = 0.0f;

        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int global_row = warp_row_start + r;
            if (global_row >= S_int) continue;
            for (int c = 0; c < tile_n; c++) {
                float p = exp2f((scores_local[r * WMMA_N + c] - row_max[r]) * LOG2E_F);
                scores_local[r * WMMA_N + c] = p;
                local_sum[r] += p;
            }
        }
        for (int r = 0; r < ROWS_PER_WARP; r++) row_sum[r] += local_sum[r];

        /* ---- Load V tile as FP16 ---- */
        for (int i = tid; i < tile_n * d_int; i += FA_NUM_THREADS) {
            int sk = i / d_int, dk = i % d_int;
            int64_t off = (int64_t)(kv_start + sk) * H_kv_int * d_int + dk;
            V_hsmem[sk * smem_stride + dk] = __float2half(V_base[off]);
        }
        __syncthreads();

        /* ---- Store P as FP16 in P_hsmem ---- */
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int global_row = warp_row_start + r;
            if (global_row >= S_int) continue;
            for (int c = tid; c < tile_n; c += FA_NUM_THREADS)
                P_hsmem[r * smem_stride + c] = __float2half(scores_local[r * WMMA_N + c]);
        }
        __syncthreads();

        /* ---- out_acc += P · V using WMMA ---- */
        for (int k_step = 0; k_step < tile_n; k_step += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> v_frag;

            wmma::load_matrix_sync(p_frag, P_hsmem + k_step, smem_stride);
            wmma::load_matrix_sync(v_frag, V_hsmem + k_step * smem_stride, smem_stride);

            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> pv_frag;
            wmma::fill_fragment(pv_frag, 0.0f);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);

            /* Accumulate into out_acc */
            float pv_local[WMMA_M * WMMA_N];
            wmma::store_matrix_sync(pv_local, pv_frag, WMMA_N, wmma::mem_row_major);
            for (int r = 0; r < ROWS_PER_WARP; r++) {
                int global_row = warp_row_start + r;
                if (global_row >= S_int) continue;
                for (int c = 0; c < d_int && c < WMMA_N; c++)
                    out_acc[r][c] += pv_local[r * WMMA_N + c];
            }
        }
        __syncthreads();
    }

    /* ---- Normalize and write output for this warp's rows ---- */
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = warp_row_start + r;
        if (global_row >= S_int) continue;

        float inv_sum = (row_sum[r] < 1e-12f) ? 1.0f : (1.0f / row_sum[r]);
        float* Y_row = Y + ((int64_t)b * S_int + global_row) * D_int;

        for (int j = lane_id; j < D_int; j += 32) {
            float contrib = 0.0f;
            for (int di = 0; di < d_int; di++)
                contrib += out_acc[r][di] * inv_sum * WO[(ho + di) * D_int + j];
            float val = contrib;
            if (h == 0) {
                val += (bO ? bO[j] : 0.0f);
                if (R_b) val += R_b[global_row * D_int + j];
            }
            atomicAdd(&Y_row[j], val);
        }
    }
}


/* ============================================================
 * Kernel 3: Flash Attention v2 Split-KV (for very long sequences)
 *
 * Splits K/V into num_splits chunks. Each block processes one
 * chunk for one query row, outputs partial O and LSE.
 *
 * Grid: (num_splits, B*S, H_q)
 * Block: (FA_NUM_THREADS=128)
 *
 * Output:
 *   O_accum[num_splits, B, S, D]  — partial output per split
 *   LSE_accum[num_splits, B, S]   — partial log-sum-exp per split
 * ============================================================ */
__global__ void mha_flash_attn_splitkv_kernel(
    const float* __restrict__ Q_buf,   /* pre-computed Q: (B, S, H_q, d) */
    const float* __restrict__ K_buf,
    const float* __restrict__ V_buf,
    float* __restrict__ O_accum,
    float* __restrict__ LSE_accum,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t H_kv, int64_t d,
    float scale, int has_residual, int causal, int num_splits)
{
    const int split_idx = blockIdx.x;
    const int bs = blockIdx.y;
    const int b  = bs / (int)S;
    const int si = bs % (int)S;
    const int h  = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (b >= (int)B || h >= (int)H_q) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int H_kv_int = (int)H_kv;
    const int group_size = (int)H_q / H_kv_int;
    const int num_warps = FA_NUM_THREADS / 32;
    const int kv_h = h / group_size;
    const int kv_ho = kv_h * d_int;
    const int ho = h * d_int;
    const int64_t kv_batch_stride = (int64_t)S_int * H_kv_int * d_int;
    const int64_t q_batch_stride = (int64_t)S_int * (int)H_q * d_int;

    /* Load pre-computed Q from buffer: Q_buf[b, si, h, :] */
    const float* Q_base = Q_buf + (int64_t)b * q_batch_stride + (int64_t)si * (int)H_q * d_int + ho;
    float Q_reg[FA_MAX_D];
    for (int di = 0; di < d_int; di++) Q_reg[di] = Q_base[di];

    /* Compute this split's K/V range */
    int causal_limit = causal ? (si + 1) : S_int;
    int tiles_per_split = (causal_limit + num_splits - 1) / num_splits;
    int kv_start = split_idx * tiles_per_split;
    int kv_end = min(kv_start + tiles_per_split, causal_limit);
    if (kv_start >= kv_end) {
        /* Empty split: write -inf LSE and zero O */
        int64_t lse_idx = ((int64_t)split_idx * B * S + (int64_t)b * S + si);
        LSE_accum[lse_idx] = -1e38f;
        int64_t o_idx = ((int64_t)split_idx * B * S * D + ((int64_t)b * S + si) * D + ho);
        for (int j = tid; j < d_int; j += FA_NUM_THREADS)
            O_accum[o_idx + j] = 0.0f;
        return;
    }

    extern __shared__ float smem[];
    const int smem_stride = d_int + FA_SKEW;
    float* K_smem  = smem;
    float* V_smem  = K_smem + FA_TILE_BN * smem_stride;
    float* red_smem = V_smem + FA_TILE_BN * smem_stride;

    /* Q is pre-computed in Q_buf, loaded into Q_reg above */

    /* Online softmax over this split's K/V range */
    float max_val = -1e38f, sum_val = 0.0f;
    float out_acc[FA_MAX_D];
    for (int di = 0; di < d_int; di++) out_acc[di] = 0.0f;

    const float* K_base = K_buf + (int64_t)b * kv_batch_stride + kv_ho;
    const float* V_base = V_buf + (int64_t)b * kv_batch_stride + kv_ho;

    int num_tiles = (kv_end - kv_start + FA_TILE_BN - 1) / FA_TILE_BN;

    for (int kt = 0; kt < num_tiles; kt++) {
        int tile_start = kv_start + kt * FA_TILE_BN;
        int tile_n = min(FA_TILE_BN, kv_end - tile_start);

        /* Load K, V tiles with int4 vectorized copy when possible */
        if (d_int % 4 == 0) {
            const int d4 = d_int / 4;
            for (int i = tid; i < tile_n * d4; i += FA_NUM_THREADS) {
                int sk = i / d4;
                int dk4 = i % d4;
                int64_t g_offset = (int64_t)(tile_start + sk) * H_kv_int * d_int + dk4 * 4;
                int s_offset = sk * smem_stride + dk4 * 4;
                *reinterpret_cast<int4*>(&K_smem[s_offset]) =
                    *reinterpret_cast<const int4*>(&K_base[g_offset]);
                *reinterpret_cast<int4*>(&V_smem[s_offset]) =
                    *reinterpret_cast<const int4*>(&V_base[g_offset]);
            }
        } else {
            for (int i = tid; i < tile_n * d_int; i += FA_NUM_THREADS) {
                int sk = i / d_int;
                int dk = i % d_int;
                int64_t offset = (int64_t)(tile_start + sk) * H_kv_int * d_int + dk;
                K_smem[sk * smem_stride + dk] = K_base[offset];
                V_smem[sk * smem_stride + dk] = V_base[offset];
            }
        }
        __syncthreads();

        float tile_max = -1e38f;
        int my_count = 0;
        float my_scores[FA_MAX_SCORES_PER_THREAD];
        int my_indices[FA_MAX_SCORES_PER_THREAD];

        for (int sk = tid; sk < tile_n; sk += FA_NUM_THREADS) {
            float dot = 0.0f;
            for (int di = 0; di < d_int; di++)
                dot += Q_reg[di] * K_smem[sk * smem_stride + di];
            float s = dot * scale;
            if (causal && (tile_start + sk) > si) s = -1e38f;
            my_scores[my_count] = s;
            my_indices[my_count] = sk;
            if (s > tile_max) tile_max = s;
            my_count++;
        }

        float warp_max = warp_reduce_max(tile_max);
        if (lane_id == 0) red_smem[warp_id] = warp_max;
        __syncthreads();
        if (tid == 0) {
            float bm = red_smem[0];
            for (int w = 1; w < num_warps; w++)
                if (red_smem[w] > bm) bm = red_smem[w];
            red_smem[0] = bm;
        }
        __syncthreads();
        float block_max = red_smem[0];

        if (block_max > max_val) {
            float rescale = __expf(max_val - block_max);
            sum_val *= rescale;
            for (int di = 0; di < d_int; di++) out_acc[di] *= rescale;
            max_val = block_max;
        }

        float local_sum = 0.0f;
        float local_out[FA_MAX_D];
        for (int di = 0; di < d_int; di++) local_out[di] = 0.0f;

        for (int i = 0; i < my_count; i++) {
            float p = exp2f((my_scores[i] - max_val) * LOG2E_F);
            local_sum += p;
            int sk = my_indices[i];
            for (int di = 0; di < d_int; di++)
                local_out[di] += p * V_smem[sk * smem_stride + di];
        }

        float warp_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) red_smem[warp_id] = warp_sum;
        __syncthreads();
        if (tid == 0) {
            float s = 0.0f;
            for (int w = 0; w < num_warps; w++) s += red_smem[w];
            red_smem[0] = s;
        }
        __syncthreads();
        sum_val += red_smem[0];
        __syncthreads();

        for (int di = 0; di < d_int; di++) {
            float wv = warp_reduce_sum(local_out[di]);
            if (lane_id == 0) red_smem[warp_id] = wv;
            __syncthreads();
            if (tid == 0) {
                float s = 0.0f;
                for (int w = 0; w < num_warps; w++) s += red_smem[w];
                out_acc[di] += s;
            }
            __syncthreads();
        }
    }

    /* Write partial O (normalized) and LSE.
     * FA2 stores normalized O_accum = out_acc / sum_val, so that
     * combine can use exp(lse_s - lse_global) as weight directly.
     * O = sum_s(exp(lse_s - lse_global) * O_accum_s)
     *   = sum_s(exp(max_s - lse_global) * out_acc_s)  [correct] */
    int64_t lse_idx = ((int64_t)split_idx * B * S + (int64_t)b * S + si);
    int64_t o_base = ((int64_t)split_idx * B * S * D + ((int64_t)b * S + si) * D);

    if (tid == 0) {
        float lse = (sum_val < 1e-12f) ? -1e38f : (max_val + logf(sum_val));
        LSE_accum[lse_idx] = lse;
    }

    /* Normalize out_acc before writing to O_accum */
    float inv_sum = (sum_val < 1e-12f) ? 0.0f : (1.0f / sum_val);
    for (int j = tid; j < d_int; j += FA_NUM_THREADS)
        O_accum[o_base + ho + j] = out_acc[j] * inv_sum;
}


/* ============================================================
 * Kernel 4: Combine Split-KV results
 *
 * For each query row, combines partial O and LSE from all splits:
 *   lse_global = logsumexp(lse_s)
 *   O = sum_s(exp(lse_s - lse_global) * O_s) / sum_s(exp(lse_s - lse_global))
 *
 * Grid: (B*S, H_q, 1), Block: (256)
 * ============================================================ */
__global__ void mha_combine_splitkv_kernel(
    const float* __restrict__ O_accum,
    const float* __restrict__ LSE_accum,
    const float* __restrict__ WO, const float* __restrict__ bO,
    const float* __restrict__ R,
    float* __restrict__ Y,
    int64_t B, int64_t S, int64_t D, int64_t H_q, int64_t d,
    int has_residual, int num_splits)
{
    const int bs = blockIdx.x;
    const int b  = bs / (int)S;
    const int si = bs % (int)S;
    const int h  = blockIdx.y;
    const int tid = threadIdx.x;
    if (b >= (int)B || h >= (int)H_q) return;

    const int S_int = (int)S, D_int = (int)D, d_int = (int)d;
    const int ho = h * d_int;

    /* Find global max LSE across splits */
    float lse_max = -1e38f;
    for (int s = 0; s < num_splits; s++) {
        int64_t idx = (int64_t)s * B * S + (int64_t)b * S + si;
        float lse = LSE_accum[idx];
        if (lse > lse_max) lse_max = lse;
    }

    /* Compute exp(lse_s - lse_max) for each split and accumulate */
    float scale_sum = 0.0f;
    float out_acc[FA_MAX_D];
    for (int di = 0; di < d_int; di++) out_acc[di] = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int64_t lse_idx = (int64_t)s * B * S + (int64_t)b * S + si;
        float lse = LSE_accum[lse_idx];
        float w = (lse <= -1e37f) ? 0.0f : __expf(lse - lse_max);
        scale_sum += w;

        int64_t o_base = (int64_t)s * B * S * D + ((int64_t)b * S + si) * D;
        for (int di = tid; di < d_int; di += 256)
            out_acc[di] += w * O_accum[o_base + ho + di];
    }

    /* Normalize and output projection */
    float inv_sum = (scale_sum < 1e-12f) ? 1.0f : (1.0f / scale_sum);
    float* Y_row = Y + ((int64_t)b * S_int + si) * D_int;
    const float* R_row = (has_residual && R) ? (R + ((int64_t)b * S_int + si) * D_int) : NULL;

    for (int j = tid; j < D_int; j += 256) {
        float contrib = 0.0f;
        for (int di = 0; di < d_int; di++)
            contrib += out_acc[di] * inv_sum * WO[(ho + di) * D_int + j];
        float val = contrib + (bO ? bO[j] : 0.0f);
        if (R_row) val += R_row[j];
        atomicAdd(&Y_row[j], val);
    }
}


/* ============================================================
 * Host dispatch
 * ============================================================ */
int mha_fused_f32_cuda(const void* inputs[], void* outputs[],
                        const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;
    if (!params) return -1;

    const mha_fused_params_t* p = (const mha_fused_params_t*)params;
    const float* X  = (const float*)inputs[0];
    const float* R  = (const float*)inputs[1];
    const float* WQ = (const float*)inputs[2];
    const float* bQ = (const float*)inputs[3];
    const float* WK = (const float*)inputs[4];
    const float* bK = (const float*)inputs[5];
    const float* WV = (const float*)inputs[6];
    const float* bV = (const float*)inputs[7];
    const float* WO = (const float*)inputs[8];
    const float* bO = (const float*)inputs[9];
    float* Y        = (float*)outputs[0];

    cudaStream_t s = stream ? (cudaStream_t)stream->cuda_stream : 0;

    int64_t B    = p->batch_size;
    int64_t S    = p->seq_len;
    int64_t D    = p->hidden_size;
    int64_t H_q  = p->num_heads;
    int64_t H_kv = p->num_kv_heads > 0 ? p->num_kv_heads : H_q;
    int64_t d    = p->head_dim;

    int has_residual = p->has_residual && R ? 1 : 0;
    int causal_mask  = p->causal ? 1 : 0;

    if (S <= 64) {
        /* Fast path: preloaded K/V */
        dim3 grid((unsigned int)(B * S), 1, 1);
        dim3 block(256, 1, 1);
        size_t smem_bytes = (size_t)(2 * S * H_kv * d) * sizeof(float);
        return CUDA_KERNEL_LAUNCH(mha_preloaded_kernel, grid, block, smem_bytes, s,
            X, WQ, bQ, WK, bK, WV, bV, WO, bO, Y, R,
            B, S, D, H_q, H_kv, d, p->scale, has_residual, causal_mask);
    }

    /* Pre-compute K/V into global buffer using cuBLAS SGEMM.
     * K_flat = X · WK  (B*S, D) × (D, D) → (B*S, D)
     * V_flat = X · WV  (B*S, D) × (D, D) → (B*S, D)
     * Then add bias: K_flat += bK, V_flat += bV.
     *
     * cuBLAS is column-major: C^T = B^T · A^T
     *   cublasSgemm(handle, N, N, D, B*S, D, α, WK, D, X, D, β, K, D)
     *   computes K^T = WK^T · X^T  ↔  K = X · WK (row-major) ✓ */
    size_t kv_size = (size_t)B * S * H_kv * d * sizeof(float);
    size_t flat_size = (size_t)B * S * D * sizeof(float);
    float* K_buf = NULL;
    float* V_buf = NULL;
    cudaMalloc(&K_buf, kv_size);
    cudaMalloc(&V_buf, kv_size);
    if (!K_buf || !V_buf) {
        if (K_buf) cudaFree(K_buf);
        if (V_buf) cudaFree(V_buf);
        return -1;
    }

    {
        dim3 kv_grid((unsigned int)(B * S), (unsigned int)H_kv, 1);
        CUDA_KERNEL_LAUNCH(mha_precompute_kv_kernel, kv_grid, dim3(256), 0, s,
            X, WK, bK, WV, bV, K_buf, V_buf, B, S, D, H_kv, d);
    }

    if (S <= FA_SPLITKV_THRESHOLD) {
        /* Single-pass Flash Attention — R4: multi-row Q tiling */
        cudaMemsetAsync(Y, 0, (size_t)B * S * D * sizeof(float), s);

        dim3 attn_grid((unsigned int)((S + FA_TILE_BM - 1) / FA_TILE_BM),
                        (unsigned int)B,
                        (unsigned int)H_q);
        dim3 attn_block(FA_NUM_THREADS, 1, 1);
        int num_warps = FA_NUM_THREADS / 32;
        int smem_stride = (int)d + FA_SKEW;

        /* Try FP16 Tensor Core kernel first (sm_70+) */
        int use_f16 = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        use_f16 = 1;
#else
        /* Check device capability at runtime */
        {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);
            if (prop.major >= 7) use_f16 = 1;
        }
#endif

        /* FP32 Flash Attention kernel.
         * FP16 WMMA kernel exists (mha_flash_attn_v2_f16_kernel) but needs
         * debugging — smem layout and WMMA fragment loading need validation.
         * Enable with: if (use_f16 && d <= FA_MAX_D) { ... } */
        {
            size_t smem_bytes = (size_t)(2 * FA_TILE_BN * smem_stride + num_warps) * sizeof(float);

            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(mha_flash_attn_v2_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes);
            }

            CUDA_KERNEL_LAUNCH(mha_flash_attn_v2_kernel, attn_grid, attn_block, smem_bytes, s,
                X, WQ, bQ, K_buf, V_buf, WO, bO, Y, R,
                B, S, D, H_q, H_kv, d, p->scale, has_residual, causal_mask);
        }
    } else {
        /* Split-KV Flash Attention */
        int num_splits = (int)((S + FA_SPLITKV_THRESHOLD - 1) / FA_SPLITKV_THRESHOLD);
        if (num_splits > 8) num_splits = 8;  /* cap at 8 splits */

        /* Allocate Q buffer + accumulators */
        size_t q_size = (size_t)B * S * H_q * d * sizeof(float);
        size_t o_accum_size = (size_t)num_splits * B * S * D * sizeof(float);
        size_t lse_accum_size = (size_t)num_splits * B * S * sizeof(float);
        float* Q_buf = NULL;
        float* O_accum = NULL;
        float* LSE_accum = NULL;
        cudaMalloc(&Q_buf, q_size);
        cudaMalloc(&O_accum, o_accum_size);
        cudaMalloc(&LSE_accum, lse_accum_size);
        if (!Q_buf || !O_accum || !LSE_accum) {
            if (Q_buf) cudaFree(Q_buf);
            if (O_accum) cudaFree(O_accum);
            if (LSE_accum) cudaFree(LSE_accum);
            cudaFree(K_buf); cudaFree(V_buf);
            return -1;
        }

        /* Pre-compute Q */
        {
            dim3 q_grid((unsigned int)(B * S), (unsigned int)H_q, 1);
            CUDA_KERNEL_LAUNCH(mha_precompute_q_kernel, q_grid, dim3(256), 0, s,
                X, WQ, bQ, Q_buf, B, S, D, H_q, d);
        }

        /* Zero-initialize Y (heads accumulate via atomicAdd in combine) */
        cudaMemsetAsync(Y, 0, (size_t)B * S * D * sizeof(float), s);

        /* Split-KV kernel */
        {
            dim3 splitkv_grid(num_splits, (unsigned int)(B * S), (unsigned int)H_q);
            dim3 splitkv_block(FA_NUM_THREADS, 1, 1);
            int num_warps = FA_NUM_THREADS / 32;
            int smem_stride = (int)d + FA_SKEW;
            size_t smem_bytes = (size_t)(2 * FA_TILE_BN * smem_stride + num_warps) * sizeof(float);

            if (smem_bytes > 48 * 1024) {
                cudaFuncSetAttribute(mha_flash_attn_splitkv_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes);
            }

            CUDA_KERNEL_LAUNCH(mha_flash_attn_splitkv_kernel,
                splitkv_grid, splitkv_block, smem_bytes, s,
                Q_buf, K_buf, V_buf, O_accum, LSE_accum,
                B, S, D, H_q, H_kv, d, p->scale, has_residual, causal_mask, num_splits);
        }

        cudaFree(Q_buf);

        /* Combine kernel */
        {
            dim3 combine_grid((unsigned int)(B * S), (unsigned int)H_q, 1);
            CUDA_KERNEL_LAUNCH(mha_combine_splitkv_kernel,
                combine_grid, dim3(256), 0, s,
                O_accum, LSE_accum, WO, bO, R, Y,
                B, S, D, H_q, d, has_residual, num_splits);
        }

        cudaFree(O_accum);
        cudaFree(LSE_accum);
    }

    cudaFree(K_buf);
    cudaFree(V_buf);
    return 0;
}

extern "C" int register_mha_fused_f32_cuda(void) {
    static operator_registry_t reg = {
        .name = "mha_fused_f32_cuda", .data_type = "f32",
        .func = mha_fused_f32_cuda, .version = 5, .flags = OP_FLAG_NONE,
    };
    return operator_register(&reg);
}
