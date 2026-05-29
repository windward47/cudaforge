#ifndef ROPE_INT_H_
#define ROPE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t seq_len;    /* S */
    int64_t head_dim;   /* d (must be even) */
    int64_t num_heads;  /* H */
    float   base;       /* theta base, typically 10000.0 */
} rope_params_t;

#endif /* ROPE_INT_H_ */
