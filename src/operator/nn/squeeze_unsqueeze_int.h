#ifndef SQUEEZE_UNSQUEEZE_INT_H_
#define SQUEEZE_UNSQUEEZE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;     /* total elements (unchanged through squeeze/unsqueeze) */
} squeeze_unsqueeze_params_t;

#endif /* SQUEEZE_UNSQUEEZE_INT_H_ */
