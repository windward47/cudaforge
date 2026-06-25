# CudaForge → HIP 移植计划

> **目标**：将 CudaForge 从 CUDA 移植到 HIP，使其在 AMD Radeon 8060S (gfx1151, RDNA3) + ROCm 7.1 上运行。
>
> **策略**：最小化代码改动——通过 `cuda2hip.h` 兼容头文件将 CUDA API 映射到 HIP，保留全部 `.cu` 文件结构和算子命名。

---

## 1. 现状分析

### 1.1 本机硬件

| 项 | 值 |
|---|---|
| CPU | AMD RYZEN AI MAX+ 395, 16C/32T, 3GHz, AVX-512 |
| RAM | 32GB (APU 统一内存，与 GPU 共享) |
| GPU | AMD Radeon 8060S, gfx1151, 40 CU, RDNA3, 96GB 共享内存 |
| NPU | XDNA2 AIE-ML |
| GPU 运行时 | ROCm 7.1.0 + HIP 7.1 |

### 1.2 代码规模

| 类别 | 数量 | 说明 |
|---|---|---|
| `.cu` 文件 (算子) | 37 | `src/operator/` 下的 kernel 实现 |
| `.cu` 文件 (平台) | 4 | `src/platform/cuda/` 下的设备/内存管理 |
| `.c` 文件 | 63 | CPU fallback + 测试 + 应用层 |
| `__global__` kernel | 74 | CUDA kernel 函数 |
| CUDA API 调用点 | 163 | 14 种不同 CUDA Runtime API |
| FP16 intrinsic | 147 | 8 种不同的 half-precision 内置函数 |
| WMMA 调用 | 24 | 2 个文件使用 Tensor Core |

### 1.3 需要移植的 CUDA 特性

| 特性 | 出现次数 | HIP 等价物 | 难度 |
|---|---|---|---|
| CUDA Runtime API (cudaMalloc 等) | 163 处 | hip 前缀替换 | ✅ 自动 |
| `CUDA_KERNEL_LAUNCH` 宏 | 74 处 | 替换为 `hipLaunchKernelGGL` | ✅ 自动 |
| `__CUDACC__` 宏守卫 | ~20 处 | 替换为 `__HIPCC__` | ✅ 自动 |
| `<cuda_runtime.h>` | 5 处 | `<hip/hip_runtime.h>` | ✅ 自动 |
| `<cuda_fp16.h>` | 9 处 | `<hip/hip_fp16.h>` | ✅ 自动 |
| FP16 intrinsics (`__half` 等) | 147 处 | 完全相同 | ✅ 无需改 |
| `nvcuda::wmma` | 24 处 (2 文件) | `rocwmma` 或 fallback | ⚠️ 需处理 |
| `__int_as_float` | 1 处 | 完全相同 | ✅ 无需改 |
| `cudaHostAlloc` | 1 处 | `hipHostMalloc` | ✅ 自动 |
| `cudaLaunchKernel` | 1 处 (宏内) | `hipModuleLaunchKernel` | ✅ 自动 |

**结论**：97% 的代码可通过 `hipify-perl` 或兼容头文件自动映射。唯二的难点是 WMMA Tensor Core（2 个文件）和构建系统。

---

## 2. 移植方案

### 2.1 总体策略：兼容头文件 + hipify

不重命名文件，不改变目录结构。创建一个 `cuda2hip.h` 头文件，在 hipcc 编译时自动将所有 CUDA API 符号映射到 HIP。代码层面只做最小改动。

```
编译流程：
  hipcc 编译 .cu 文件
    → 包含 cuda2hip.h（自动重命名 CUDA API → HIP）
    → 生成 HIP kernel
    → 链接到 ROCm 运行时
```

### 2.2 受影响文件清单

```
需修改的文件（共 8 个）：
  CMakeLists.txt                              # 构建系统
  src/platform/include/cuda_ops.h             # 核心头文件
  src/platform/cuda/cuda_memory.cu            # 内存管理
  src/platform/cuda/cuda_device.cu            # 设备管理
  src/platform/cuda/cuda_arena.cu             # Arena 分配器
  src/platform/cuda/cuda_platform_init.cu     # 平台初始化
  src/operator/blas/matmul_cuda.cu            # WMMA Tensor Core
  src/operator/nn/mha_fused_f16_cuda.cu       # WMMA Tensor Core

需新建的文件（共 1 个）：
  src/platform/include/cuda2hip.h             # CUDA → HIP 兼容头文件
```

---

## 3. 实施步骤

### Phase 1：创建兼容头文件与 CMake 改造

#### 步骤 1.1：创建 `src/platform/include/cuda2hip.h`

```c
/* cuda2hip.h — CUDA-to-HIP translation layer
 * When compiled with hipcc, this header transparently maps
 * all CUDA runtime API calls to their HIP equivalents.
 */
#ifndef CUDA2HIP_H_
#define CUDA2HIP_H_

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/* ---- 类型映射 ---- */
#define cudaStream_t         hipStream_t
#define cudaError_t          hipError_t
#define cudaDeviceProp       hipDeviceProp_t
#define cudaSuccess          hipSuccess

/* ---- 常量映射 ---- */
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

/* ---- API 映射（14 个 Runtime API） ---- */
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaHostAlloc            hipHostMalloc
#define cudaFreeHost             hipHostFree
#define cudaMemcpyAsync          hipMemcpyAsync
#define cudaSetDevice            hipSetDevice
#define cudaDeviceReset          hipDeviceReset
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamSynchronize    hipStreamSynchronize
#define cudaStreamDestroy        hipStreamDestroy
#define cudaGetErrorString       hipGetErrorString
#define cudaLaunchKernel         hipModuleLaunchKernel

#endif /* __HIPCC__ */
#endif /* CUDA2HIP_H_ */
```

#### 步骤 1.2：修改 `CMakeLists.txt`

- 将 `enable_language(CUDA)` 替换为 HIP 检测逻辑
- 新增 `ENABLE_HIP` 选项（默认 ON）
- 使用 `hip_add_library()` 或 `find_package(hip REQUIRED)` 编译 `.cu` 文件
- 目标架构从 `sm_86` 改为 `gfx1151`
- 链接 `hip::device` 代替 CUDA runtime
- 保留 `ENABLE_CUDA` 和 `ENABLE_TESTS` 选项兼容性

关键 CMake 改动：

```cmake
option(ENABLE_HIP "Enable HIP/ROCm backend" ON)

if(ENABLE_HIP)
    find_package(hip REQUIRED)
    set(CMAKE_HIP_ARCHITECTURES "gfx1151")
    # .cu 文件自动由 hipcc 编译
    enable_language(HIP)
    add_compile_definitions(USE_HIP)
endif()
```

#### 步骤 1.3：修改 `cuda_ops.h`

- 在文件顶部 `#include "cuda2hip.h"`
- 将 `#ifdef __CUDACC__` 守卫改为 `#if defined(__CUDACC__) || defined(__HIPCC__)`
- `CUDA_KERNEL_LAUNCH` 宏内部：当 `__HIPCC__` 时使用 `hipLaunchKernelGGL` 替代 `cudaLaunchKernel`

### Phase 2：平台层移植

#### 步骤 2.1：移植 4 个平台层 `.cu` 文件

这些文件通过 `cuda2hip.h` 自动映射，改动极小：

| 文件 | 改动 |
|---|---|
| `cuda_memory.cu` | `cudaHostAlloc` 的 flag 参数从 `cudaHostAllocDefault` → `hipHostMallocDefault`（已被宏覆盖，无需改代码） |
| `cuda_device.cu` | 无代码改动（全部通过宏映射） |
| `cuda_arena.cu` | 无代码改动 |
| `cuda_platform_init.cu` | 无代码改动 |

#### 步骤 2.2：验证平台层编译

```bash
cmake -B build -DENABLE_HIP=ON -DENABLE_CUDA=OFF -DENABLE_TESTS=ON
cmake --build build 2>&1 | head -50
```

预期：平台层 4 个文件编译通过，链接到 `libamdhip64.so`。

### Phase 3：算子 Kernel 移植

#### 步骤 3.1：批量编译验证（35 个纯 Runtime API kernel 文件）

这 35 个文件只使用标准 CUDA Runtime API + FP16 intrinsics，通过 `cuda2hip.h` 可直接编译：

```
relu_cuda.cu          add_cuda.cu           sub_cuda.cu
mul_cuda.cu           div_cuda.cu           softmax_cuda.cu
pooling_cuda.cu       batchnorm_cuda.cu     conv_cuda.cu
concat_cuda.cu        reshape_cuda.cu       transpose_cuda.cu
resize_cuda.cu        slice_cuda.cu         split_cuda.cu
layernorm_cuda.cu     gather_cuda.cu        squeeze_unsqueeze_cuda.cu
reduce_cuda.cu        cast_cuda.cu          argmax_cuda.cu
globalavgpool_cuda.cu causal_mask_cuda.cu   rope_cuda.cu
pad_cuda.cu           clip_cuda.cu          where_cuda.cu
activations_cuda.cu   mha_decode_cuda.cu    mha_fused_cuda.cu
conv_f16_cuda.cu      matmul_f16_cuda.cu    elementwise_f16_cuda.cu
softmax_f16_cuda.cu   norm_f16_cuda.cu
```

**预期**：全部通过 `hipcc` 编译，零代码改动。

#### 步骤 3.2：处理 WMMA Tensor Core（2 个文件）

这是唯一的难点。涉及文件：
- `src/operator/blas/matmul_cuda.cu` — 8 处 WMMA 调用
- `src/operator/nn/mha_fused_f16_cuda.cu` — 16 处 WMMA 调用

**方案 A（推荐）：条件编译 + Fallback**

在 WMMA 相关代码段用 `#ifdef __HIPCC__` 包裹，HIP 路径使用已有的 tiled matmul kernel（不使用 Tensor Core）作为替代。这两个文件已经有非 WMMA 的实现路径，只需在 HIP 编译时跳过 WMMA 分支。

```c
#ifdef __CUDACC__  /* CUDA: 使用 WMMA Tensor Core */
    // ... wmma:: fragment, load, mma, store ...
#elif defined(__HIPCC__)  /* HIP: 使用 tiled matmul fallback */
    // ... 已有的 tiled_shared_matmul_kernel ...
#endif
```

**方案 B（可选，后续优化）：使用 rocWMMA**

安装 `apt install rocwmma-dev`，用 `rocwmma` 命名空间替代 `nvcuda::wmma`。API 几乎 1:1 映射：

| CUDA WMMA | rocWMMA |
|---|---|
| `nvcuda::wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>` | `rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major>` |
| `wmma::load_matrix_sync` | `rocwmma::load_matrix_sync` |
| `wmma::mma_sync` | `rocwmma::mma_sync` |
| `wmma::store_matrix_sync` | `rocwmma::store_matrix_sync` |
| `wmma::fill_fragment` | `rocwmma::fill_fragment` |

但 RDNA3 (gfx1151) 的 wave32 架构下，rocWMMA 的 m16n16k16 支持需要验证。建议 Phase 3 先走方案 A，Phase 5 再优化。

### Phase 4：测试验证

#### 步骤 4.1：编译全部测试

```bash
cmake -B build -DENABLE_HIP=ON -DENABLE_CUDA=OFF -DENABLE_TESTS=ON \
      -DENABLE_AVX512=ON -DENABLE_OPENMP=ON
cmake --build build -j$(nproc)
```

#### 步骤 4.2：运行 CPU 测试（基线）

```bash
ctest --test-dir build -R "test_" --output-on-failure
```

预期：所有 CPU-only 测试通过（与 CUDA 完全无关）。

#### 步骤 4.3：运行 GPU 测试

```bash
# 平台层
./build/test_cuda_kernels

# CPU vs GPU 精度对比
./build/test_ops_scale
./build/test_conv_scale
./build/test_reduce
./build/test_rope
./build/test_fp16_ops
```

精度容差（沿用项目规范）：
- FP32 CPU vs HIP：`1e-4` 以内
- FP16 CPU vs HIP：`0.02` 以内
- Conv2D scale test：`1e-3` 以内

#### 步骤 4.4：端到端推理验证

```bash
# ONNX 模型加载 + 推理
./build/test_onnx
./build/test_resnet
./build/test_yolo
./build/test_gpt2_generate
```

### Phase 5：性能优化（后续）

| 优化项 | 预期收益 | 复杂度 |
|---|---|---|
| 启用 AVX-512 CPU 路径 | CPU 推理 2-4x | 低（CMake 选项） |
| rocWMMA Tensor Core | FP16 矩阵乘 2-4x | 中（安装 + API 适配） |
| Shared memory bank conflict 优化 | 减少 bank conflict 延迟 | 中 |
| Wave-level reduction（`__shfl_xor`） | reduce/softmax 更快 | 低（HIP 已支持） |
| ROCm rocBLAS 集成 | 矩阵乘接近峰值 | 中（API 替换） |

---

## 4. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| gfx1151 是较新 RDNA3 变体，ROCm 兼容性待验证 | 编译/运行失败 | ROCm 7.1 已声明支持 gfx11*；如遇问题可降级到 gfx1100 通用目标 |
| WMMA 在 RDNA3 wave32 下行为差异 | Tensor Core 精度/性能异常 | Phase 3 用 tiled fallback 先跑通，Phase 5 再启用 WMMA |
| 96GB 共享内存是 APU 统一内存，带宽低于独立显卡 | 大模型推理较慢 | 这是硬件限制，无法软件解决；中等模型足够 |
| `hipModuleLaunchKernel` vs `hipLaunchKernelGGL` 宏行为差异 | kernel 启动失败 | `cuda2hip.h` 中优先使用 `hipLaunchKernelGGL`，需要调整 `CUDA_KERNEL_LAUNCH` 宏实现 |

---

## 5. 执行清单

- [ ] Phase 1.1：创建 `src/platform/include/cuda2hip.h`
- [ ] Phase 1.2：修改 `CMakeLists.txt`（HIP 构建支持）
- [ ] Phase 1.3：修改 `cuda_ops.h`（条件编译守卫 + kernel launch 宏）
- [ ] Phase 2.1：验证平台层 4 个文件编译通过
- [ ] Phase 2.2：运行 `test_cuda_kernels` 验证平台层功能
- [ ] Phase 3.1：验证 35 个算子文件编译通过（零改动预期）
- [ ] Phase 3.2：处理 WMMA（matmul_cuda.cu + mha_fused_f16_cuda.cu 的 fallback 路径）
- [ ] Phase 4.1：全部测试编译通过
- [ ] Phase 4.2：CPU 测试全部通过
- [ ] Phase 4.3：GPU 测试全部通过（精度在容差内）
- [ ] Phase 4.4：端到端推理验证（ONNX / ResNet / YOLO / GPT-2）
- [ ] Phase 5：性能优化（按需）
