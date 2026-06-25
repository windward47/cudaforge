# CudaForge — GPU 神经网络推理库

## 环境信息

| 项 | 值 |
| --- | --- |
| OS | Ubuntu (Linux x86_64) |
| CPU | AMD RYZEN AI MAX+ 395, 16C/32T, 3GHz, AVX-512 |
| RAM | 32GB (APU 统一内存，与 GPU 共享) |
| GPU | AMD Radeon 8060S, gfx1151, 40 CU, RDNA3, 96GB 共享内存 |
| GPU 运行时 | ROCm 7.1.0 + HIP 7.1 |
| C 编译器 | GCC 15.2 / hipcc (Clang 21) |
| CMake | 4.2 |

## 项目定位

基于 GPU 的神经网络推理算子库，当前正从 CUDA 移植到 HIP/ROCm。自底向上分三层：

```text
Application  →  模型加载、推理引擎、应用集成
Operator     →  算子注册/调度、张量运算（CPU + GPU）
Platform     →  硬件抽象、内存管理（CPU + GPU）、线程调度
```

**关键原则**：

- 每个算子必须同时提供 **GPU kernel** 和 **纯 C fallback**（验证用）
- 严格分层依赖：Application → Operator → Platform
- 禁止反向或跨层依赖
- `.cu` 文件编译为 **C++17**（hipcc），`.c` 文件编译为 **C11**（GCC）

## 构建状态

**v0.9.0** — AVX512 + ROCm/HIP 多平台集成完成。

| 模式 | 编译选项 | 测试结果 |
| --- | --- | --- |
| HIP + AVX512 + OpenMP | `ENABLE_HIP=ON ENABLE_AVX512=ON` | 35/36 通过 |
| CPU-only + AVX512 | `ENABLE_HIP=OFF ENABLE_CUDA=OFF` | 32/32 通过 |

### 编译选项

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_HIP` | ON | HIP/ROCm 后端（AMD GPU） |
| `ENABLE_CUDA` | OFF | CUDA 后端（NVIDIA GPU） |
| `ENABLE_AVX512` | ON | AVX-512 SIMD 优化（CPU 算子） |
| `ENABLE_AVX2` | OFF | AVX2 SIMD 优化（CPU 算子） |
| `ENABLE_OPENMP` | ON | OpenMP 并行 |
| `ENABLE_TESTS` | ON | 构建测试 |
| `HIP_ARCH` | gfx1151 | HIP 目标架构 |
| `CUDA_ARCH` | 86 | CUDA 目标架构 |

## 目录结构

```text
cudaforge/
├── CLAUDE.md              # ← 项目入口（你现在看的）
├── PLAN.md                # HIP 移植计划（详细步骤）
├── CMakeLists.txt         # 顶层构建
├── docs/
│   ├── ARCHITECTURE.md    # 架构规范（含 GPU 扩展）
│   ├── CODING_STYLE.md    # 编码规范
│   ├── CUDA_GUIDE.md      # CUDA kernel 模板、错误检查、优化策略
│   ├── TASK_TEMPLATES/    # 常见任务分步模板
│   └── TODO_HIP.md        # ← HIP 移植任务跟踪
├── src/
│   ├── platform/          # 平台抽象层（CPU + GPU）
│   │   ├── cpu/           # x86 CPU 实现
│   │   ├── cuda/          # GPU 设备/内存/平台初始化 (.cu)
│   │   └── include/       # platform.h, cuda_ops.h, cuda2hip.h
│   ├── operator/          # 算子实现（含 .c 和 .cu）
│   │   ├── blas/          # 矩阵运算（matmul）
│   │   └── nn/            # 神经网络算子（relu/conv/pool/batchnorm/activations）
│   └── application/       # 模型加载与推理引擎
└── tests/                 # 集成测试
```

## 行为约束

### 必须遵守

1. **算子成对实现**：新增一个算子时，`xxx.c`（CPU fallback）和 `xxx_cuda.cu`（GPU kernel）同时实现
2. **算子注册使用 X-macro**：在 `src/operator/operator_registry.def` 中添加 `REGISTER(...)` 条目，不要手动修改 `operator_init.c` 中的注册逻辑
3. **GPU API 调用必须通过 `cuda_ops_t` 接口层**，禁止直接调用 CUDA/HIP Runtime API
4. **kernel launch 使用 `CUDA_KERNEL_LAUNCH` 宏**（内部为 C++ variadic template，自动取 `&` 地址打包参数）
5. **HIP 编译时 `cuda2hip.h` 自动映射 CUDA API 到 HIP**——不要手动在 `.cu` 文件中写 `hip*` 调用
6. **提交前 GPU 测试必须过 `compute-sanitizer`**（无 memory leak / out-of-bounds / misaligned access）
7. **不要自动安装依赖**（CUDA Toolkit、ROCm、GPU 驱动等）— 必须先问用户
8. **不要修改 `build/` 目录下 CMake 生成的文件**
9. **一个 commit 只做一个事**：不要混合 feat / fix / refactor

### 推荐遵守

- 算子的 GPU kernel 先写 naive 版本 → 再优化（shared memory → wave-level → tensor core）
- 浮点对比测试时，CPU 和 GPU 结果做相对误差比较（`allclose`），不要 `assert_equal`
- 算子专用类型定义放在独立内部头文件（如 `conv_int.h`、`pooling_int.h`），`.c` 和 `.cu` 共用
- WMMA Tensor Core 代码用 `#ifdef __CUDACC__` 条件编译，HIP 用 tiled fallback

## 常用命令

# === HIP/ROCm 模式（AMD GPU，默认）===
cmake -B build -DENABLE_HIP=ON -DENABLE_CUDA=OFF -DENABLE_TESTS=ON \
      -DENABLE_AVX512=ON -DENABLE_OPENMP=ON
cmake --build build -j$(nproc)

# === CUDA 模式（NVIDIA GPU）===
cmake -B build -DENABLE_HIP=OFF -DENABLE_CUDA=ON -DCUDA_ARCH=86 -DENABLE_TESTS=ON
cmake --build build -j$(nproc)

# === 仅 CPU 模式（AVX-512）===
cmake -B build -DENABLE_HIP=OFF -DENABLE_CUDA=OFF -DENABLE_AVX512=ON -DENABLE_TESTS=ON
cmake --build build -j$(nproc)

# === 仅 CPU 模式（AVX2）===
cmake -B build -DENABLE_HIP=OFF -DENABLE_CUDA=OFF -DENABLE_AVX2=ON -DENABLE_TESTS=ON
cmake --build build -j$(nproc)

# === 运行测试 ===
ctest --test-dir build --output-on-failure -j$(nproc)

# === 批量 benchmark ===
./scripts/run_benchmarks.sh              # 运行所有 bench
./scripts/run_benchmarks.sh --profile-only  # 只运行算子级 profile

# === GPU 内存检查 ===
./scripts/run_sanitizer.sh               # 批量 compute-sanitizer
compute-sanitizer ./build/test_matmul

# === 代码质量检查 ===
./scripts/check_raw_cuda.sh              # 检测裸 CUDA API 调用
./scripts/check_registry.sh              # 验证 .def 注册一致性
./scripts/check_perf_regression.sh /tmp/current.csv docs/PROFILE_BASELINE.csv
```

## 文档索引

| 文档 | 内容 | 什么时候看 |
| --- | --- | --- |
| [PLAN.md](PLAN.md) | HIP 移植计划（详细步骤与检查点） | 执行移植任务时 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、GPU 平台、算子规范 | 开始写代码前 |
| [docs/CODING_STYLE.md](docs/CODING_STYLE.md) | C 命名/错误处理/头文件规范 | 写/审代码时 |
| [docs/CUDA_GUIDE.md](docs/CUDA_GUIDE.md) | CUDA kernel 模板、错误检查、优化策略 | 写/审 GPU kernel 时 |
| [docs/TODO_HIP.md](docs/TODO_HIP.md) | HIP 移植任务跟踪 | 了解当前进度 |
| [docs/TODO_LIST.md](docs/TODO_LIST.md) | 历史任务记录（CUDA 时代） | 查看已完成工作 |
