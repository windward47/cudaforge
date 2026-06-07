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

/* Flash Attention tile size: K/V tile rows loaded from global memory */
#define FA_TILE_BR  64   /* query tile rows (not used in current 1-query-per-block design) */
#define FA_TILE_BC  64   /* K/V tile columns (tile size for K/V loading) */
#define FA_MAX_D    64   /* maximum head dimension */

#endif /* MHA_FUSED_INT_H_ */
