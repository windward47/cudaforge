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

### Level 1.5: 共享内存归约（Shared-Memory Reduction）

适用场景：LayerNorm、Softmax 沿轴计算均值/方差/最大值
- pow2 对齐 block 尺寸，stride-based 并行归约
- 动态共享内存分配（`extern __shared__`）
- 与 warp-level 不同，在 block 级做数据聚合

```cuda
/* LayerNorm CUDA kernel — shared-memory mean/variance reduction */
__global__ void layernorm_f32_kernel(const float* input, float* output,
                                       const float* gamma, const float* beta,
                                       int64_t N, int64_t normalized_size,
                                       float epsilon) {
    extern __shared__ float s_buf[];
    float* s_mean = s_buf;
    float* s_var  = s_buf + blockDim.x;

    int64_t row = blockIdx.x;
    if (row >= N) return;
    const float* in_row = input + row * normalized_size;
    float* out_row       = output + row * normalized_size;

    /* 1. 并行归约计算均值 */
    float sum = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x)
        sum += in_row[i];
    s_mean[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_mean[threadIdx.x] += s_mean[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = s_mean[0] / (float)normalized_size;

    /* 2. 并行归约计算方差 */
    float sq = 0.0f;
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float d = in_row[i] - mean;
        sq += d * d;
    }
    s_var[threadIdx.x] = sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_var[threadIdx.x] += s_var[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_std = rsqrtf(s_var[0] / (float)normalized_size + epsilon);

    /* 3. 归一化 + 仿射变换 */
    for (int i = threadIdx.x; i < normalized_size; i += blockDim.x) {
        float val = (in_row[i] - mean) * inv_std;
        if (gamma) val = val * gamma[i] + beta[i];
        out_row[i] = val;
    }
}
```

**关键点**：
- block 尺寸取 2 的幂（256 / 512），保证 stride 归约无分支发散
- `__syncthreads()` 在每次归约迭代时必须调用（共享内存读写之间）
- `extern __shared__` 通过 kernel launch 第三个参数动态分配大小

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

### Level 4: Kernel Fusion（算子融合）

适用场景：Conv/MatMul → ReLU/Sigmoid/GELU 连续模式，或 Multi-Head Attention 子图融合，消除中间张量的显存往返。

**简单融合 (Conv+Activation)**: 在 conv/matmul kernel 末尾内联激活函数，避免单独的激活 kernel launch 和中间结果的 global memory 读写。

**复杂融合 (MHA_Fused)**: 将 ~14 节点的 self-attention 子图（3×QKV MatMul + 3×Reshape + 3×Transpose + Q·K^T MatMul + Mul + Softmax + ×V MatMul + Transpose + Reshape + Output MatMul + Residual Add）融合为单个 kernel。核心设计：

```cuda
/* MHA Fused Kernel — grid=(B*S), 每 block 处理一个 query position
   内部循环所有 head，避免 block 间输出竞争 */
__global__ void mha_fused_kernel(
    const float* X,      /* (B, S, D) — LayerNorm 输出 */
    const float* WQ, const float* bQ,
    const float* WK, const float* bK,
    const float* WV, const float* bV,
    const float* WO, const float* bO,
    float* Y,            /* (B, S, D) — 输出 */
    const float* R,      /* (B, S, D) — 残差，可为 NULL */
    int64_t B, S, D, H, d, float scale, int has_residual)
{
    int bs = blockIdx.x;         /* flat: b * S + si */
    int b  = bs / S;
    int si = bs % S;

    __shared__ float K_smem[64 * 64];  /* K/V tile 缓存 */
    __shared__ float V_smem[64 * 64];

    /* 初始化输出 = bias + residual */
    Y_bs[j] = bO[j] + (R_bs ? R_bs[j] : 0);

    /* 逐 head 循环（避免 block 间竞争） */
    for (int h = 0; h < H; h++) {
        /* 1. 计算 Q_h[si, :] */
        float Q_reg[64];  /* ≤ head_dim */
        for (di) Q_reg[di] = sum_j X[si,j] * WQ[j, h*d+di] + bQ[h*d+di];

        /* 2. 协作加载 K_h, V_h → shared memory */
        for (i = tid; i < S*d; i += blockDim.x) {
            int sk = i / d, dk = i % d;
            K_smem[sk*d+dk] = sum_j X[sk,j] * WK[j, h*d+dk] + bK[h*d+dk];
            V_smem[sk*d+dk] = sum_j X[sk,j] * WV[j, h*d+dk] + bV[h*d+dk];
        }
        __syncthreads();

        /* 3. Q·K^T → scale → softmax */
        float scores[S];
        for (sj) scores[sj] = dot(Q_reg, K_smem[sj,:]) * scale;
        for (sj) scores[sj] = expf(scores[sj] - max_val); sum_val += scores[sj];

        /* 4. Weighted V sum + output projection */
        float merged[d];
        for (di) merged[di] = sum_j scores[sj] * V_smem[sj,di] / sum_val;

        /* 5. 累加到输出: Y += merged · WO_slice */
        for (j) {
            float contrib = 0;
            for (di) contrib += merged[di] * WO[h*d+di, j];
            Y_bs[j] += contrib;
        }
        __syncthreads();
    }
}
```

**关键设计决策**:
- **grid=(B×S)** 而非 (B×H): 每 block 写唯一输出位置，避免 block 间用 atomicAdd 竞争
- **Heads 串联**而非并联: 多 block 写同一输出用 atomicAdd 会产生大量全局内存竞争，不如在单 block 内顺序处理
- **scores/merged 用寄存器数组**: 避免共享内存竞态（曾导致 max_diff=28.7 的错误）
- **仅 S≤64 时加载完整 K/V 到 shared memory**: 长序列用 tiled 在线 softmax

**Application 层检测**: `detect_and_fuse_mha()` 在 `graph_execute()` 的 fusion pre-pass 中检测完整 self-attention 子图模式（QKV MatMul→Reshape→Transpose→Q·K^T→Mul→Softmax→×V→Transpose→Reshape→Output MatMul→Residual Add），转换为 MHA_Fused 节点。融合后保存原始节点配置，每次执行后恢复（非破坏性融合）。

```cuda
/* conv kernel 末尾 — 融合激活 */
float sum = ...;  /* conv 累加结果 */

/* 1. 先加 bias（如果融合了） */
if (bias) sum += bias[k];

/* 2. 再应用激活函数 */
if (fuse_activation == 1) {
    sum = sum > 0.0f ? sum : 0.0f;  /* ReLU */
} else if (fuse_activation == 2) {
    sum = 1.0f / (1.0f + expf(-sum));  /* Sigmoid */
}
```

**关键注意**：bias 必须在激活函数**之前**应用。错误顺序 `ReLU(conv(x)) + bias` 与正确顺序 `ReLU(conv(x) + bias)` 的语义完全不同。当 bias ≠ 0 时，错误顺序产生静默的数值错误。

**多消费者安全检查**：融合前必须验证 Conv/MatMul 输出张量仅被激活节点消费。若存在额外消费者（如 SiLU 模式 `Conv→Sigmoid→Mul` 中 Conv 输出同时喂给 Sigmoid 和 Mul），融合会导致未写入的陈旧数据被下游节点读取。`graph_execute` 的 fusion pre-pass 扫描所有节点计数消费者数量，`consumers > 1` 时跳过融合。

**Application 层配合**: `graph_execute` 的 fusion pre-pass 检测拓扑序相邻的 Conv→ReLU/Sigmoid/GELU 模式，设置 `fuse_activation` 标记并跳过被融合的激活节点。输出张量通过 `effective_output_tids` 重定向到激活节点的输出，保证下游消费者拿到后激活结果。

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
    tensor_t* input  = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){4, 256});
    /* fill input->data with random values via manual loop or helper */
    tensor_t* output_cpu = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){4, 256});
    tensor_t* output_gpu = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){4, 256});

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
