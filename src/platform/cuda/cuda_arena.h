#ifndef CUDA_ARENA_H_
#define CUDA_ARENA_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int    cuda_arena_init(size_t total_bytes);
void*  cuda_arena_alloc(size_t bytes);
void   cuda_arena_reset(void);
void   cuda_arena_destroy(void);
size_t cuda_arena_used(void);
size_t cuda_arena_capacity(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_ARENA_H_ */
