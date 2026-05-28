#ifndef MHA_FUSED_INT_H_
#define MHA_FUSED_INT_H_

#include <stdint.h>
#include <stdbool.h>

typedef struct {
    int64_t batch_size;       /* B */
    int64_t seq_len;          /* S */
    int64_t hidden_size;      /* D */
    int64_t num_heads;        /* H */
    int64_t head_dim;         /* d = D/H */
    float   scale;            /* 1/sqrt(d) */
    bool    has_residual;     /* true if residual input is provided */
} mha_fused_params_t;

#endif /* MHA_FUSED_INT_H_ */
