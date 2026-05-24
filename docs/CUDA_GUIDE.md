# CUDA 算子开发指南

## 0. 编译环境

### .cu 文件编译为 C++20

nvcc 将 `.cu` 文件编译为 **C++20**（`CMAKE_CUDA_STANDARD=20`）。宿主 `.c` 文件由 MSVC 编译为 C11。

### cuda_ops.h 的双编译模式

`src/platform/include/cuda_ops.h` 是整个项目的 CUDA 入口。它通过 `__CUDACC__` 宏在两个模式间切换：

| 宏定义 | 编译场景 | `<cuda_runtime.h>` | CUDA 类型 |
| --- | --- | --- | --- |
| `__CUDACC__` 已定义 | nvcc 编译 `.cu` | 真实 include | `dim3`, `cudaStream_t`, `cudaDeviceProp` 等 |
| `__CUDACC__` 未定义 | MSVC 编译 `.c` | 不 include | 前向声明（`cudaStream_t` → `void*`） |

`.c` 文件在 `USE_CUDA` 宏定义时可以安全地 include `cuda_ops.h`，并访问 `g_cuda` 全局单例的函数指针表（`device_alloc`、`memcpy_h2d` 等）。

### 算子参数结构体的内部头文件模式

算子专用参数类型（如 `conv_params_t`、`pool_params_t`）定义在独立的 `*_int.h` 头文件中，`.c` 和 `.cu` 共用：

```
src/operator/nn/
├── conv.c             # CPU fallback (include "conv_int.h")
├── conv_cuda.cu       # CUDA kernel (include "conv_int.h")
├── conv_int.h         # conv_params_t 定义
├── pooling.c
├── pooling_cuda.cu
├── pooling_int.h      # pool_params_t 定义
├── batchnorm.c
├── batchnorm_cuda.cu
├── batchnorm_int.h    # batchnorm_params_t 定义
└── relu.c / relu_cuda.cu   # relu 参数简单，无需 *_int.h
```

## 1. Kernel 模板

### 通用模板

```cuda
/* kernel 定义 */
__global__ void relu_f32_kernel(const float* input, float* output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}

/* launch wrapper */
int relu_f32_cuda(const void* inputs[], void* outputs[],
                  const operator_params_t* params, stream_t* stream) {
    if (!inputs || !inputs[0] || !outputs || !outputs[0]) return -1;

    const float* in  = (const float*)inputs[0];
    float* out       = (float*)outputs[0];
    int64_t n        = ((const int64_t*)inputs[1])[0];  /* numel via meta */
    cudaStream_t s   = stream ? (cudaStream_t)stream->cuda_stream : 0;
    dim3 block(256, 1, 1);
    dim3 grid((unsigned int)((n + 255) / 256), 1, 1);

    CUDA_KERNEL_LAUNCH(relu_f32_kernel, grid, block, 0, s, in, out, n);
    return 0;
}
```

### 错误检查宏

`CUDA_CHECK` 宏定义在 `cuda_ops.h` 中（已包含 `<stdio.h>` 和 `<stdlib.h>`）：

```cuda
/* src/platform/include/cuda_ops.h */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
```

### Kernel launch 辅助宏

使用 **C++ variadic template**（非 C99 compound literal）。模板自动对每个参数取 `&`，正确生成 `cudaLaunchKernel` 所需的 `void**` 数组：

```cuda
/* src/platform/include/cuda_ops.h */

/* C++ variadic template — 仅 __CUDACC__ 时可见 */
template<typename... Args>
static inline void _cuda_kernel_call(const void* kernel, dim3 grid, dim3 block,
                                      size_t shared_mem, cudaStream_t stream,
                                      Args&&... args) {
    void* params[] = { (void*)&args... };
    CUDA_CHECK(cudaLaunchKernel(kernel, grid, block, params, shared_mem, stream));
}

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    _cuda_kernel_call((const void*)(kernel), grid, block, shared_mem, stream, __VA_ARGS__)
```

> **为什么不是 C99 compound literal `(const void*[]){...}`？** `.cu` 文件编译为 C++20，MSVC 不支持该语法。variadic template 是等价的标准 C++ 方案。

---

## 2. 优化策略（按优先级）

### Level 0: Naive 实现

目标：功能正确，可验证
- 直接全局内存访问
- 每个线程处理一个元素
- 不做任何 block / tile 优化

### Level 1: 共享内存（Shared Memory）

适用场景：数据复用（conv、matmul）
- 将 tile 从 global memory 加载到 `__shared__`
- 块内同步：`__syncthreads()`
- 注意 bank conflict

```cuda
__global__ void matmul_f32_kernel(const float* A, const float* B, float* C,
                                   int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    /* tile 循环... */
}
```

### Level 2: Warp-level 原语

适用场景：reduce、scan、softmax
- `__shfl_down_sync` / `__shfl_xor_sync`
- `__warp_reduce` / `__warp_scan`
- warp 内无需 `__syncthreads()`

```cuda
/* Warp Reduce */
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}
```

### Level 3: Tensor Core

适用场景：`matmul`、`conv`（计算能力 >= 7.0）
- `wmma::fragment` / `wmma::load_matrix_sync`
- 要求维度对齐（通常是 16 的倍数）

```cuda
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

__global__ void matmul_f16_tc(const half* A, const half* B, float* C,
                               int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    /* ... */
    wmma::load_matrix_sync(a_frag, A + row * K, K);
    wmma::load_matrix_sync(b_frag, B + col * K, K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
```

---

## 3. 性能 checklist

| 检查项 | 说明 |
| --- | --- |
| Grid/Block 配置 | block 通常取 128~256 threads，grid 足够大以掩盖延迟 |
| 合并访问 | 相邻线程访问相邻地址（thread `i` 访问 `data[i]`） |
| 共享内存 bank conflict | 避免同一 warp 访问同一 bank 的不同地址 |
| 分支发散 | 同一 warp 内避免 if-else 不同分支 |
| `__syncthreads` 位置 | 只在共享内存读写之间使用，避免不必要的同步 |
| Occupancy | 用 `cudaOccupancyMaxPotentialBlockSize` 检查 |
| 数据传输 | 尽量异步（stream）+ pinned memory，最小化 H2D/D2H |

---

## 4. 数值验证

### CPU vs GPU 对比模板

```c
/* 测试用例 */
void test_relu_f32() {
    /* 1. 准备输入数据 */
    tensor_t* input  = tensor_rand(DATA_TYPE_F32, 2, (int64_t[]){4, 256});
    tensor_t* output_cpu = tensor_like(input);
    tensor_t* output_gpu = tensor_like(input);

    /* 2. CPU fallback */
    relu_f32(input, output_cpu, input->numel);

    /* 3. GPU kernel */
    tensor_copy_to_device(input);
    relu_f32_cuda(input, output_gpu, NULL);
    tensor_copy_to_host(output_gpu);

    /* 4. 对比 */
    assert_tensor_allclose(output_cpu, output_gpu, 1e-5f, 1e-5f);
    /* 清理... */
}
```

### allclose 实现

```c
bool tensor_allclose(const tensor_t* a, const tensor_t* b,
                     float rtol, float atol) {
    for (int64_t i = 0; i < a->numel; i++) {
        float diff = fabsf(a->data_f32[i] - b->data_f32[i]);
        float max  = fmaxf(fabsf(a->data_f32[i]), fabsf(b->data_f32[i]));
        if (diff > atol && diff > rtol * max) {
            return false;
        }
    }
    return true;
}
```
