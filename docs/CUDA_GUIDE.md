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

#### WMMA 注意事项（RTX 2050 / sm_86 实战经验）

1. **smem 对齐**：`wmma::load_matrix_sync` 要求 256-bit（32 字节）对齐。共享内存指针起始地址需对齐，每行 leading dimension 应为 16 的倍数。
2. **smem skew padding**：朴素行优先布局会引发 bank conflict。参考 `cudaTensorCoreGemm.cu` 的 `SKEW_HALF`，每行末尾加 16 个 half 偏移（本项目 `FA_SKEW`）。
3. **行/列主序与转置**：要计算 `Q · Kᵀ`，K 在 smem 中是 `(rows, d)` 行主序，用 `wmma::col_major` 加载即得到 `Kᵀ`（标准技巧，无需手动转置）。
4. **寄存器压力**：accumulator fragment 每个 16×16 占 8 个 float/线程。**切勿**用 `float out[ROWS][D]` 这样的 per-thread 大数组（如 16×64=1024 floats）做 WMMA 输出累积——会导致寄存器溢出 / kernel 无法启动（"unspecified launch failure"）。正确做法是用多个 fragment（4 个 16×16 fragment 覆盖 64 列，共 32 floats/线程）。
5. **partial tile 处理**：K/V tile 不足 16 行时，WMMA 仍读取 16 行，超出部分必须零填充，否则读到 garbage。

#### Flash Attention 2 优化路径

参考 `flash-attention-main/` 官方实现 + Tri Dao 论文。本项目 `mha_fused_cuda.cu` 的演进（FP32 路径）：

| 阶段 | 优化 | S=512 耗时 | 关键改动 |
| --- | --- | --- | --- |
| 初始 | 标量循环 | 391 ms | 每 block 1 行 Q，smem 全局归约 |
| R2-7 | warp 归约 + exp2f | 391 ms | `warp_reduce_sum/max` + `exp2f(x*log2e)` |
| R3-a | grid H_q 维度 | 310 ms | `blockIdx.z = h`，消除 head 串行 |
| R4-a/b | 多行 Q tiling | **93 ms** | BM=64，4 warps 各 16 行，warp 级归约无通信 |

**FA2 对齐要点**（论文 Section 3.1-3.3）：
- **Online softmax 不缩放中间 O**：每轮只 rescale 一次 `exp(m_old - m_new)`，最后才 `O = diag(ℓ)^{-1} · Õ`
- **外层 Q / 内层 K-V 循环**：Q blocks 天然并行（grid 维度）
- **Warp 分 Q 不分 K/V**（Fig 3b）：每个 warp 独立处理 Q 的不同行，K/V 共享，**无跨 warp 通信**
- **non-matmul FLOP 贵 16×**：尽量把 Q·Kᵀ 和 P·V 走 Tensor Core（FP16 路径，见下）

**未完成项**：FP16 WMMA Flash Attention（`mha_flash_attn_v2_f16_kernel`）需正确的列分块 accumulator fragment 设计（见上文第 4 点），当前为 experimental。

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

    /* 动态共享内存: 2*S*d floats (16KB for BERT-base S=8,d=64)
       比静态 64*64*2 (64KB) 节省 4x，SM 占用率翻倍 */
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = smem + S * d;

    /* 初始化输出 = bias + residual */
    Y_bs[j] = bO[j] + (R_bs ? R_bs[j] : 0);

    /* 逐 head 循环（避免 block 间竞争） */
    for (int h = 0; h < H; h++) {
        /* 1. 计算 Q_h[si, :] 到寄存器 */
        float Q_reg[64];
        for (di) Q_reg[di] = sum_j X[si,j] * WQ[j, h*d+di] + bQ[h*d+di];

        /* 2. Tiled K/V 处理 + Online Softmax */
        float max_val = -1e38f, sum_val = 0.0f;
        float out_acc[64] = {0};

        for (kt = 0; kt < num_kv_tiles; kt++) {
            /* 2a. 协作加载 K/V tile → shared memory */
            for (i = tid; i < tile_s * d; i += blockDim.x) {
                int sk = i / d, dk = i % d;
                K_smem[sk*d+dk] = sum_j X[sk,j] * WK[j, h*d+dk] + bK[h*d+dk];
                V_smem[sk*d+dk] = sum_j X[sk,j] * WV[j, h*d+dk] + bV[h*d+dk];
            }
            __syncthreads();

            /* 2b. 计算 tile 内 scores */
            for (sk) tile_scores[sk] = dot(Q_reg, K_smem[sk,:]) * scale;

            /* 2c. Online softmax: rescale 累积器 */
            if (tile_max > max_val) {
                sum_val *= expf(max_val - tile_max);
                out_acc *= expf(max_val - tile_max);
                max_val = tile_max;
            }

            /* 2d. 累加 exp(scores) · V */
            for (sk) {
                float p = expf(tile_scores[sk] - max_val);
                sum_val += p;
                for (di) out_acc[di] += p * V_smem[sk*d+di];
            }
            __syncthreads();
        }

        /* 3. 归一化 + 输出投影: Y += (out_acc/sum_val) · WO */
        for (j) Y_bs[j] += out_acc[di] / sum_val * WO[h*d+di, j];
    }
}
```

**关键设计决策**:
- **grid=(B×S)** 而非 (B×H): 每 block 写唯一输出位置，避免 block 间用 atomicAdd 竞争
- **Heads 串联**而非并联: 多 block 写同一输出用 atomicAdd 会产生大量全局内存竞争，不如在单 block 内顺序处理
- **scores/out_acc 用寄存器数组**: 避免共享内存竞态（曾导致 max_diff=28.7 的错误）
- **动态共享内存** (`extern __shared__`): 按 `2*S*d` 分配，比静态 64×64 节省 4x 显存，SM 占用率翻倍
- **Tiled K/V + Online Softmax**: K/V 按 S 维度分 tile 加载到 shared memory，tile 间通过 rescale 累积器保持数值一致性，支持任意序列长度（S>64）

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

---

## 5. 按算子类型的优化路径

> 参考 CUDAForge 的两阶段 Judge 模式：先诊断瓶颈类型，再选择对应优化策略。

| 算子类型 | 代表算子 | 瓶颈特征 | 优先优化方向 | 现状 |
| --- | --- | --- | --- | --- |
| Element-wise | ReLU, GELU, SiLU, Add, Mul | 带宽受限 | Vectorized loads (float4) | 待优化 |
| Reduction | Softmax, LayerNorm, BatchNorm | 带宽+计算混合 | Shared memory reduction | 已实现 |
| GEMM | MatMul, Conv | 计算受限 | Tiling -> Warp -> Tensor Core | 4 级全覆盖 |
| Attention | MHA Fused | 带宽+计算混合 | 融合 + tiled online softmax | 已实现 |
| Pooling | MaxPool, AvgPool | 带宽受限 | Shared memory 窗口复用 | 待优化 |
| Permute | Transpose, Reshape, Slice | 带宽受限 | 合并访问 + 零拷贝 | 基础实现 |

---

## 6. NCU 诊断 Checklist

> 参考 CUDAForge 收集的 24 项 Nsight Compute 硬件指标。使用 `compute-sanitizer` 和 `nsight-compute` 进行算子级性能分析。

### 6.1 关键指标

| 指标类别 | NCU Metric | 含义 | 健康范围 |
| --- | --- | --- | --- |
| SM 占用率 | `sm__warps_active.avg.pct_of_peak_sustained_active` | 实际活跃 warp 占峰值比例 | >60% |
| 寄存器压力 | `launch__registers_per_thread` | 每线程寄存器数 | <64（避免 spilling） |
| 显存带宽 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | DRAM 带宽利用率 | >70%（带宽受限算子） |
| L1 命中率 | `l1tex__t_sector_hit_rate.pct` | L1 缓存命中率 | >80% |
| L2 命中率 | `lts__t_sector_hit_rate.pct` | L2 缓存命中率 | >50% |
| Warp 停顿 | `smsp__warps_issue_stalled_long_scoreboard_per_issue_active` | 长延迟停顿（显存访问） | <20% |
| 分支发散 | `smsp__sass_average_branch_targets_threads_uniform.pct` | 分支目标一致性 | >90% |

### 6.2 瓶颈诊断流程

```text
1. 运行 NCU profile:
   ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,
                 launch__registers_per_thread,
                 dram__throughput.avg.pct_of_peak_sustained_elapsed
       ./build/Release/bench_xxx.exe

2. 判断瓶颈类型:
   IF dram_throughput > 70% AND sm_occupancy < 50%:
     → 带宽受限：优化内存访问模式（合并访问、vectorized loads）
   IF dram_throughput < 30% AND sm_occupancy > 60%:
     → 计算受限：优化计算密度（tiling、Tensor Core）
   IF registers_per_thread > 64:
     → 寄存器溢出：减少中间变量、循环展开
   IF l1_hit_rate < 50%:
     → 局部性差：使用 shared memory 缓存
   IF branch_divergence < 80%:
     → 分支发散：重构条件逻辑、用算术替代分支
```

### 6.3 常用 NCU 命令

```bash
# 基本 profile（延迟 + SM 占用率）
ncu --set basic ./build/Release/bench_matmul.exe

# 详细 profile（内存 + 缓存 + warp 停顿）
ncu --set full ./build/Release/bench_matmul.exe

# 只看特定指标
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread ./build/Release/bench_relu.exe

# 输出为 CSV（便于脚本处理）
ncu --csv --set basic ./build/Release/bench_matmul.exe > profile_matmul.csv
```

---

## 7. CPU SIMD (AVX2) 优化

> 参考 CUDA-Agent 的向量化加载策略，为 CPU 算子添加 AVX2 优化路径。

### 7.1 启用方式

```bash
# 构建 AVX2 版本
cmake -B build-avx2 -DENABLE_AVX2=ON -DENABLE_TESTS=ON
cmake --build build-avx2 -j$(nproc)
```

CMake 会自动为 `operator` 库添加 `-mavx2 -mfma`（GCC/Clang）或 `/arch:AVX2`（MSVC）编译标志，并定义 `USE_AVX2` 宏。

### 7.2 编码模式

```c
#if defined(USE_AVX2)
#include <immintrin.h>

static int relu_f32_avx2(const float* in, float* out, int64_t n) {
    __m256 zero = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(in + i);
        _mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
    }
    for (; i < n; i++) out[i] = fmaxf(in[i], 0.0f);  // 标量尾部
    return 0;
}
#endif

int relu_f32(...) {
    // ... 参数校验 ...
#if defined(USE_AVX2)
    return relu_f32_avx2(in, out, n);
#else
    for (int64_t i = 0; i < n; i++) out[i] = fmaxf(in[i], 0.0f);
    return 0;
#endif
}
```

### 7.3 已优化算子及性能

| 算子 | 标量 (ms) | AVX2 (ms) | 加速比 |
| --- | --- | --- | --- |
| ReLU (1M) | 12.30 | 0.30 | **41x** |
| GELU (1M) | 16.90 | 1.00 | **17x** |
| Sigmoid (1M) | 4.50 | 1.30 | **3.5x** |
| Softmax | ✓ | ✓ | 向量化 max/exp/sum |
| LayerNorm | ✓ | ✓ | 向量化 mean/var/normalize |
| Reduce | ✓ | ✓ | 向量化 sum/max |

### 7.4 关键 AVX2 内联函数

| 指令 | 用途 | 示例 |
| --- | --- | --- |
| `_mm256_loadu_ps` | 加载 8 个 float | `__m256 v = _mm256_loadu_ps(ptr)` |
| `_mm256_storeu_ps` | 存储 8 个 float | `_mm256_storeu_ps(ptr, v)` |
| `_mm256_add_ps` | 8 路并行加法 | `_mm256_add_ps(a, b)` |
| `_mm256_mul_ps` | 8 路并行乘法 | `_mm256_mul_ps(a, b)` |
| `_mm256_fmadd_ps` | 融合乘加 (FMA) | `_mm256_fmadd_ps(a, b, c)` → a*b+c |
| `_mm256_max_ps` | 8 路并行取最大 | `_mm256_max_ps(a, b)` |
| `_mm256_set1_ps` | 广播标量到 8 路 | `_mm256_set1_ps(3.14f)` |
| `_mm256_setzero_ps` | 8 路零向量 | `_mm256_setzero_ps()` |
