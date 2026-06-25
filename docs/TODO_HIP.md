# CudaForge — HIP 移植与多平台集成任务跟踪

> 基于 [PLAN.md](../PLAN.md) 的详细步骤。历史任务记录见 [TODO_LIST.md](TODO_LIST.md)。

---

## 当前状态

**v0.9.0** — AVX512 + ROCm/HIP 多平台集成完成。

| 指标 | 数值 |
| --- | --- |
| 目标 GPU | AMD Radeon 8060S (gfx1151, RDNA3) |
| 运行时 | ROCm 7.1 + HIP 7.1 |
| 编译器 | hipcc (Clang 21) + GCC 15 |
| HIP 测试通过 | **35/36**（1 个 RDNA3 已知问题） |
| CPU-only 测试通过 | **32/32** |
| 编译错误 | 0 |

---

## Phase 1：创建兼容头文件与 CMake 改造 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 1.1 | 创建 `cuda2hip.h` 兼容头文件 | `src/platform/include/cuda2hip.h` | ✅ | 双模式：hipcc 全量宏映射 + host-GCC 类型/函数映射 |
| 1.2 | 修改 `CMakeLists.txt` | `CMakeLists.txt` | ✅ | `ENABLE_HIP` + `ENABLE_CUDA` + `ENABLE_AVX512` + `ENABLE_AVX2` + `ENABLE_OPENMP` |
| 1.3 | 修改 `cuda_ops.h` | `src/platform/include/cuda_ops.h` | ✅ | `__HIPCC__`/`USE_CUDA` 守卫 + HIP kernel launch |

## Phase 2：平台层移植 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 2.1 | 编译平台层 .cu 文件 | 4 个 `src/platform/cuda/*.cu` | ✅ | 通过 cuda2hip.h + cuda_runtime.h shim |
| 2.2 | 验证平台层链接 ROCm | — | ✅ | 链接到 libamdhip64.so |

## Phase 3：算子 Kernel 移植 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 3.1 | 编译 35 个标准算子 .cu 文件 | 35 个 `src/operator/**/*_cuda.cu` | ✅ | 通过 cuda2hip.h 自动映射 |
| 3.2 | 处理 WMMA（matmul_cuda.cu） | `src/operator/blas/matmul_cuda.cu` | ✅ | `#ifdef __CUDACC__` 条件编译，HIP 用 tiled/rocBLAS fallback |
| 3.3 | 处理 WMMA（mha_fused_cuda.cu） | `src/operator/nn/mha_fused_cuda.cu` | ✅ | `#ifdef __CUDACC__` 守卫 WMMA kernel，HIP 用 FP32 flash fallback |

## Phase 4：测试验证 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 4.1 | 编译全部测试 | — | ✅ | 44 个目标全部编译成功（HIP 模式） |
| 4.2 | CPU 测试基线 | — | ✅ | 32 个 CPU-only 测试全部通过 |
| 4.3 | GPU 测试精度校验 | `test_cuda_kernels`, `test_ops_scale`, `test_conv_scale`, `test_reduce`, `test_rope`, `test_fp16_ops` | ✅ | 全部通过 |
| 4.4 | 端到端推理验证 | `test_onnx`, `test_resnet`, `test_yolo`, `test_gpt2_generate` | ✅ | 全部通过 |

---

## 额外修复（移植过程中发现）

| 修复 | 文件 | 说明 |
| --- | --- | --- |
| cuda_runtime.h shim | `src/platform/cuda/cuda_runtime.h` | HIP 编译时提供 `<cuda_runtime.h>` 替代 |
| cuda_fp16.h shim | `src/platform/cuda/cuda_fp16.h` | HIP 编译时提供 `<cuda_fp16.h>` 替代 |
| cooperative_groups.h shim | `src/platform/cuda/cooperative_groups.h` | HIP 编译时提供替代 |
| mma.h shim | `src/platform/cuda/mma.h` | WMMA 头文件占位 |
| WMMA 条件编译 | `matmul_cuda.cu`, `mha_fused_cuda.cu` | `#ifdef __CUDACC__` 守卫 WMMA 代码 |
| nodiscard 修复 | `cuda_arena.cu`, `cuda_memory.cu`, `cuda_device.cu`, `cast_cuda.cu`, `rope_cuda.cu`, `mha_fused_cuda.cu` | `(void)` cast 处理 HIP API 返回值 |
| GCC 15 兼容 | `graph.c`, `test_conv.c`, `test_reduce.c` | unused function/variable 警告修复 |
| HIP 警告抑制 | `CMakeLists.txt` | `-Wno-sign-compare` 等 HIP 编译选项 |
| shfl_xor_sync 修复 | `cuda_reduce.cuh` | mask 从 `0xffffffff` 改为 `0xffffffffULL`（HIP 要求 64 位） |
| host-GCC HIP 兼容 | `cuda2hip.h` | 新增 Mode 2：host GCC 编译 .c 文件时提供 HIP 类型/函数映射 |
| cudaDeviceProp 兼容 | `cuda_ops.h` | `struct cudaDeviceProp*` → `cudaDeviceProp*`（避免 typedef 冲突） |

---

## Phase 5：性能优化（基于 BENCHMARK_HIP.md 建议）

| # | 任务 | 预期收益 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 5.1 | **CPU AVX-512 优化** — matmul、softmax、MHA、relu、add、layernorm 使用 SIMD intrinsics + OpenMP | 当前场景 2-4x | ✅ | `-mavx512f -mfma` 编译、`#pragma omp parallel for` 并行 |
| 5.2 | **批量推理** (B≥8) — `inference_session_run_batch` API | GPU 3-10x | ✅ | 顺序执行+算子内部 OpenMP 并行 |
| 5.3 | **FP16 CPU 推理路径** — matmul_f16、add_f16、relu_f16 CPU 算子 | 内存带宽 2x | ✅ | `_Float16` 转换 + 委托 FP32 计算 |
| 5.4 | **rocWMMA Tensor Core** — 安装 rocwmma 包后启用 RDNA3 WMMA | FP16 计算 2-4x | ⬜ | `librocwmma-dev` 可用，需 `sudo apt install` |
| 5.5 | **Kernel fusion** — conv+bias+relu 已内建于 conv_params_t.fuse_activation | 减少中间结果读写 | ✅ | conv 已支持 fuse_activation (ReLU/Sigmoid/GELU/SiLU) |

---

## Phase 6：多平台编译选项整合 ✅

| # | 任务 | 状态 | 说明 |
| --- | --- | --- | --- |
| 6.1 | `ENABLE_HIP` / `ENABLE_CUDA` 互斥 GPU 后端选择 | ✅ | HIP 默认 ON，CUDA 默认 OFF |
| 6.2 | `ENABLE_AVX512` / `ENABLE_AVX2` SIMD 选项 | ✅ | AVX512 默认 ON，仅对 C 文件生效 |
| 6.3 | `HIP_ARCH` / `CUDA_ARCH` 架构目标 | ✅ | `gfx1151` / `86` 默认值 |
| 6.4 | `GPU` 统一变量替代 `ENABLE_CUDA` 判断 | ✅ | 平台层和算子层用 `if(GPU)` 统一判断 |
| 6.5 | operator_registry.def 支持 `REGISTER_HIP` | ✅ | X-macro 扩展为 CPU/CUDA/HIP 三路 |
| 6.6 | `USE_CUDA` 守卫兼容 HIP 构建 | ✅ | HIP 构建时定义 `USE_CUDA` 复用现有守卫 |

---

## 已知问题

| 问题 | 影响 | 状态 | 说明 |
| --- | --- | --- | --- |
| `test_bert_mha` 长序列 (S=128) 内存异常 | 1/36 测试 | ⬜ | RDNA3 flash attention 内存映射问题，短序列正常 |

---

## 进度
| 状态 | 数量 | 内容 |
| --- | --- | --- |
| ✅ 已完成 | 22 | Phase 1-6 全部任务（5.4 除外） |
| ⬜ 待完成 | 2 | 5.4 rocWMMA、test_bert_mha 长序列修复 |

> **最后更新**: 2026-06-25。v0.9.0。AVX512 + ROCm/HIP 多平台集成完成。35/36 HIP 测试通过，32/32 CPU 测试通过。
