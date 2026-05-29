#ifndef CAUSAL_MASK_INT_H_
#define CAUSAL_MASK_INT_H_

#include <stdint.h>

typedef struct {
    int64_t seq_len;  /* S — output is (S, S) */
} causal_mask_params_t;

#endif /* CAUSAL_MASK_INT_H_ */
