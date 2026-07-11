#ifndef MHA_DECODE_INT_H_
#define MHA_DECODE_INT_H_

#include <stdint.h>
#include "rope_int.h"   /* ROPE_LAYOUT_* constants */

typedef struct {
    int64_t batch_size;   /* B */
    int64_t hidden_size;  /* D */
    int64_t num_heads;    /* H_q - query heads */
    int64_t num_kv_heads; /* H_kv - key/value heads (GQA: H_kv < H_q, MQA: H_kv=1) */
    int64_t head_dim;     /* d = D/H_q */
    float   scale;        /* 1/sqrt(d) */
    int64_t cache_len;    /* number of cached positions (current token index) */
    int64_t max_seq;      /* max sequence length (cache buffer size) */

    /* RoPE fusion (R9-h). When rope_base > 0, Q and K_new are rotated after
       projection, using pos = cache_len (the new token's position). This keeps
       the KV-cache storing rotated K, so prefill/decode cache semantics align.
       rope_base == 0 disables RoPE (backward-compatible: memset->0 skips it). */
    float   rope_base;        /* theta base (e.g. 10000.0); 0 = RoPE disabled */
    int32_t rope_layout;      /* ROPE_LAYOUT_INTERLEAVED or ROPE_LAYOUT_HALFSPLIT */
    const float* rope_inv_freq; /* precomputed freq table [d/2]; NULL = derive from rope_base */
} mha_decode_params_t;

#endif /* MHA_DECODE_INT_H_ */
