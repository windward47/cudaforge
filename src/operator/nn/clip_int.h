#ifndef CLIP_INT_H_
#define CLIP_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;
    float   min_val;
    float   max_val;
} clip_params_t;

#endif /* CLIP_INT_H_ */
