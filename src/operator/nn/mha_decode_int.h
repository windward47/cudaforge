#ifndef MHA_DECODE_INT_H_
#define MHA_DECODE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t batch_size;   /* B */
    int64_t hidden_size;  /* D */
    int64_t num_heads;    /* H */
    int64_t head_dim;     /* d = D/H */
    float   scale;        /* 1/sqrt(d) */
    int64_t cache_len;    /* number of cached positions (current token index) */
    int64_t max_seq;      /* max sequence length (cache buffer size) */
} mha_decode_params_t;

#endif /* MHA_DECODE_INT_H_ */
