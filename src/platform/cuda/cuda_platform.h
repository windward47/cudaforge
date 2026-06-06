#ifndef CUDA_PLATFORM_H_
#define CUDA_PLATFORM_H_

#include "cuda_ops.h"

#ifdef __cplusplus
extern "C" {
#endif

int  cuda_platform_init(int device_id);
void cuda_platform_finalize(void);

/**
 * @brief 获取当前 GPU 硬件能力快照
 * @param caps  输出参数，填充 gpu_caps_t 结构
 * @param device_id  设备 ID（通常为 0）
 * @return 0 成功，负值失败
 */
int cuda_get_gpu_caps(gpu_caps_t* caps, int device_id);

/**
 * @brief 打印 GPU 硬件能力摘要到 stderr
 * @param device_id  设备 ID（通常为 0）
 */
void cuda_print_gpu_caps(int device_id);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_PLATFORM_H_ */
