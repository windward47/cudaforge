#include "platform.h"
#include <stdio.h>

#ifdef USE_CUDA
#include "cuda_ops.h"
#include "cuda_platform.h"
#endif

/* Forward declaration from operator library */
extern int operator_init_all(void);

int main(void) {
    printf("CudaForge v0.1.0\n");

    if (platform_init() != 0) {
        fprintf(stderr, "platform_init() failed\n");
        return 1;
    }

    /* Register all built-in operators */
    operator_init_all();

    printf("Platform: %s (%d cores, %d B cache line)\n",
           g_platform ? g_platform->name : "?",
           platform_get_core_count(),
           platform_get_cache_line_size());

#ifdef USE_CUDA
    if (cuda_platform_init(0) == 0) {
        int dev_count = 0;
        g_cuda.get_device_count(&dev_count);
        printf("CUDA: enabled (%d device(s))\n", dev_count);
    } else {
        printf("CUDA: init failed (no GPU?)\n");
    }
#else
    printf("CUDA: disabled\n");
#endif

#ifdef USE_CUDA
    cuda_platform_finalize();
#endif
    platform_finalize();
    return 0;
}
