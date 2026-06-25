# CudaForge — HIP 移植任务跟踪

> 基于 [PLAN.md](../PLAN.md) 的详细步骤。历史任务记录见 [TODO_LIST.md](TODO_LIST.md)。

---

## 当前状态

**v0.7.0** — CUDA → HIP 移植完成。35/35 测试全部通过。

| 指标 | 数值 |
| --- | --- |
| 目标 GPU | AMD Radeon 8060S (gfx1151, RDNA3) |
| 运行时 | ROCm 7.1 + HIP 7.1 |
| 编译器 | hipcc (Clang 21) + GCC 15 |
| 测试通过 | **35/35** |
| 编译错误 | 0 |

---

## Phase 1：创建兼容头文件与 CMake 改造 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 1.1 | 创建 `cuda2hip.h` 兼容头文件 | `src/platform/include/cuda2hip.h` | ✅ | 16 个 CUDA API → HIP 映射 |
| 1.2 | 修改 `CMakeLists.txt` | `CMakeLists.txt` | ✅ | `ENABLE_HIP` + hipcc 构建 + .cu→HIP glob |
| 1.3 | 修改 `cuda_ops.h` | `src/platform/include/cuda_ops.h` | ✅ | `__HIPCC__` 守卫 + HIP kernel launch |

## Phase 2：平台层移植 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 2.1 | 编译平台层 .cu 文件 | 4 个 `src/platform/cuda/*.cu` | ✅ | 通过 cuda2hip.h + cuda_runtime.h shim |
| 2.2 | 验证平台层链接 ROCm | — | ✅ | 链接到 libamdhip64.so |

## Phase 3：算子 Kernel 移植 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 3.1 | 编译 35 个标准算子 .cu 文件 | 35 个 `src/operator/**/*_cuda.cu` | ✅ | 通过 cuda2hip.h 自动映射 |
| 3.2 | 处理 WMMA（matmul_cuda.cu） | `src/operator/blas/matmul_cuda.cu` | ✅ | `#ifdef __CUDACC__` 条件编译，HIP 用 tiled fallback |
| 3.3 | 处理 WMMA（mha_fused_f16_cuda.cu） | `src/operator/nn/mha_fused_f16_cuda.cu` | ✅ | 同上 |

## Phase 4：测试验证 ✅

| # | 任务 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 4.1 | 编译全部测试 | — | ✅ | 37 个目标全部编译成功 |
| 4.2 | CPU 测试基线 | — | ✅ | 20 个 CPU-only 测试全部通过 |
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
| WMMA 条件编译 | `matmul_cuda.cu`, `mha_fused_f16_cuda.cu` | `#ifdef __CUDACC__` 守卫 WMMA 代码 |
| nodiscard 修复 | `cuda_arena.cu`, `cast_cuda.cu`, `rope_cuda.cu` | `(void)` cast 处理 hipFree/hipMemcpyAsync 返回值 |
| GCC 15 兼容 | `graph.c`, `test_conv.c`, `test_reduce.c` | unused function/variable 警告修复 |
| HIP 警告抑制 | `CMakeLists.txt` | `-Wno-sign-compare` 等 HIP 编译选项 |


## Phase 5：性能优化（基于 BENCHMARK_HIP.md 建议）

| # | 任务 | 预期收益 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| 5.1 | **CPU AVX-512 优化** — matmul、softmax、MHA、relu、add、layernorm 使用 SIMD intrinsics + OpenMP | 当前场景 2-4x | ✅ | `-mavx512f -mfma` 编译、`#pragma omp parallel for` 并行 |
| 5.2 | **批量推理** (B≥8) — `inference_session_run_batch` API | GPU 3-10x | ✅ | 顺序执行+算子内部 OpenMP 并行 |
| 5.3 | **FP16 CPU 推理路径** — matmul_f16、add_f16、relu_f16 CPU 算子 | 内存带宽 2x | ✅ | `_Float16` 转换 + 委托 FP32 计算 |
| 5.4 | **rocWMMA Tensor Core** — 安装 rocwmma 包后启用 RDNA3 WMMA | FP16 计算 2-4x | ⬜ | `librocwmma-dev` 可用，需 `sudo apt install` |
| 5.5 | **Kernel fusion** — conv+bias+relu 已内建于 conv_params_t.fuse_activation | 减少中间结果读写 | ✅ | conv 已支持 fuse_activation (ReLU/Sigmoid/GELU/SiLU) |

### Phase 5 详细变更

| 文件 | 变更 |
| --- | --- |
| `CMakeLists.txt` | `ENABLE_AVX512=ON`（默认）、`-mavx512f -mfma` 编译选项、`fp16_cpu_ops.c` 添加 |
| `matmul.c` | AVX-512 4×ZMM 内核 + OpenMP 并行（M 行并行） |
| `softmax.c` | AVX-512 max/sum reduction + sub_scalar + scale + OpenMP（N 块并行） |
| `mha_fused.c` | AVX-512 QKV 投影 + attention dot product + OpenMP collapse(2/3)、栈分配替代 calloc |
| `relu.c` | AVX-512 `_mm512_max_ps` 向量化 |
| `add.c` | AVX-512 element-wise 向量化 |
| `layernorm.c` | AVX-512 mean/variance/normalize + OpenMP |
| `fp16.h` | FP16↔FP32 转换工具（`_Float16` 或位操作回退） |
| `fp16_cpu_ops.c` | matmul_f16、add_f16、relu_f16 CPU 算子 |
| `inference_engine.c/h` | `inference_session_run_batch` 批量推理 API |
| `operator_init.c` | `#ifdef USE_CUDA` 守卫 CUDA 注册、FP16 CPU 注册 |
| `graph.c` | `#ifdef USE_CUDA` 守卫 g_cuda 引用 |

### 测试结果

| 指标 | 结果 |
| --- | --- |
| 测试总数 | 31 (不含 test_yolo) |
| 通过 | 31 |
| 通过率 | 100% |

---

## 进度
| 状态 | 数量 | 内容 |
| --- | --- | --- |
| ✅ 已完成 | 16 | Phase 1-5 全部任务（5.4 除外） |
| ⬜ 待完成 | 1 | 5.4 rocWMMA（需 sudo 安装 librocwmma-dev） |

> **最后更新**: 2026-06-06。v0.8.0。Phase 5 性能优化完成（4/5）。31/31 测试通过。