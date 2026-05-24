#ifndef CUDA_PLATFORM_H_
#define CUDA_PLATFORM_H_

#ifdef __cplusplus
extern "C" {
#endif

int  cuda_platform_init(int device_id);
void cuda_platform_finalize(void);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PLATFORM_H_ */
