#ifndef ADD_INT_H_
#define ADD_INT_H_

#include <stdint.h>

typedef struct {
    int64_t numel;       /* total elements in output (= input A numel) */
    int64_t B_numel;     /* total elements in input B (1 = scalar, C = per-channel) */
} add_params_t;

#endif /* ADD_INT_H_ */
