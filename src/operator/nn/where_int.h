#ifndef WHERE_INT_H_
#define WHERE_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;       /* output elements */
    int64_t cond_numel;  /* condition elements (for broadcast) */
    int64_t x_numel;     /* X elements (for broadcast) */
    int64_t y_numel;     /* Y elements (for broadcast) */
} where_params_t;

#endif /* WHERE_INT_H_ */
