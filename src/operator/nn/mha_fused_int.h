#ifndef MHA_FUSED_INT_H_
#define MHA_FUSED_INT_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t batch_size;       /* B */
    int64_t seq_len;          /* S */
    int64_t hidden_size;      /* D */
    int64_t num_heads;        /* H_q (query heads) */
    int64_t num_kv_heads;     /* H_kv (key/value heads, GQA: H_kv <= H_q) */
    int64_t head_dim;         /* d = D/H_q */
    float   scale;            /* 1/sqrt(d) */
    bool    has_residual;     /* true if residual input is provided */
    bool    causal;           /* true = decoder causal mask (upper triangle masked) */
} mha_fused_params_t;

/* Flash Attention v2 tile sizes (reference: flash-attention-main/csrc/flash_attn/src/kernel_traits.h)
 *
 * FA_TILE_BM: Q tile rows per block (each block processes this many query rows)
 * FA_TILE_BN: K/V tile columns per block (each iteration loads this many K/V rows)
 * FA_MAX_D:   maximum head dimension (limited by register/smem pressure)
 *
 * Grid layout: (ceil(S / FA_TILE_BM), B, H_q)
 * Each block handles FA_TILE_BM query rows against all K/V tiles.
 */
#define FA_TILE_BM  64   /* query tile rows per block */
#define FA_TILE_BN  64   /* K/V tile rows per iteration */
#define FA_MAX_D    64   /* maximum head dimension */

/* Warp/block config for Flash Attention kernel */
#define FA_NUM_WARPS  4   /* number of warps per block (4 × 32 = 128 threads) */
#define FA_NUM_THREADS (FA_NUM_WARPS * 32)  /* 128 threads per block */

/* Max K/V positions a single thread can handle per tile:
 *   ceil(FA_TILE_BN / FA_NUM_THREADS) — currently ceil(64/128) = 1
 *   +1 for safety when tile_n is not a multiple of FA_NUM_THREADS */
#define FA_MAX_SCORES_PER_THREAD (FA_TILE_BN / FA_NUM_THREADS + 1)

#endif /* MHA_FUSED_INT_H_ */
