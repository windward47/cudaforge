# CudaForge — CUDA 神经网络推理库

## 环境信息

| 项 | 值 |
| --- | --- |
| OS | Windows 10 x64 |
| C 编译器 | MSVC 2022 (19.44) + MinGW GCC |
| CUDA Toolkit | 13.2 (nvcc) |
| GPU | NVIDIA GeForce RTX 2050 (sm_86, compute capability 8.6) |
| CMake 生成器 | Visual Studio 17 2022 (Windows) / Unix Makefiles (Linux) |

## 项目定位

基于 CUDA 的神经网络推理算子库。自底向上分三层：

```text
Application  →  模型加载、推理引擎、应用集成
Operator     →  算子注册/调度、张量运算（CPU + CUDA）
Platform     →  硬件抽象、内存管理（CPU + GPU）、线程调度
```

**关键原则**：

- 每个算子必须同时提供 **CUDA kernel** 和 **纯 C fallback**（验证用）
- 严格分层依赖：Application → Operator → Platform
- 禁止反向或跨层依赖
- `.cu` 文件编译为 **C++20**（nvcc），`.c` 文件编译为 **C11**（MSVC）

## 目录结构

```text
cuda_dev/
├── CLAUDE.md              # ← 项目入口（你现在看的）
├── CMakeLists.txt         # 顶层构建
├── docs/
│   ├── ARCHITECTURE.md    # 架构规范（含 CUDA 扩展）
│   ├── CODING_STYLE.md    # 编码规范
│   ├── CUDA_GUIDE.md      # CUDA 算子开发指南
│   └── TASK_TEMPLATES/    # 常见任务分步模板
├── src/
│   ├── platform/          # 平台抽象层（CPU + CUDA）
│   │   ├── cpu/           # x86 CPU 实现
│   │   ├── cuda/          # CUDA 设备/内存/平台初始化 (.cu)
│   │   └── include/       # platform.h, cuda_ops.h
│   ├── operator/          # 算子实现（含 .c 和 .cu）
│   │   ├── blas/          # 矩阵运算（matmul）
│   │   └── nn/            # 神经网络算子（relu/conv/pool/batchnorm/activations）
│   └── application/       # 模型加载与推理引擎
└── tests/                 # 集成测试（跟随各模块的 test/ 目录）
```

## 行为约束

### 必须遵守

1. **算子成对实现**：新增一个算子时，`xxx.c`（CPU fallback）和 `xxx_cuda.cu`（CUDA kernel）同时实现
2. **算子注册使用 X-macro**：在 `src/operator/operator_registry.def` 中添加 `REGISTER(...)` 条目，不要手动修改 `operator.c` 中的注册数组
3. **CUDA API 调用必须通过 `cuda_ops_t` 接口层**，禁止直接调用 CUDA Runtime API
4. **CUDA kernel launch 使用 `CUDA_KERNEL_LAUNCH` 宏**（内部为 C++ variadic template，自动取 `&` 地址打包参数）
5. **提交前 GPU 测试必须过 `compute-sanitizer`**（无 memory leak / out-of-bounds / misaligned access）
6. **不要自动安装依赖**（CUDA Toolkit、GPU 驱动等）— 必须先问用户
7. **不要修改 `build/` 目录下 CMake 生成的文件**
8. **一个 commit 只做一个事**：不要混合 feat / fix / refactor

### 推荐遵守

- 算子的 CUDA kernel 先写 naive 版本 → 再优化（shared memory → warp-level → tensor core）
- 浮点对比测试时，CPU 和 CUDA 结果做相对误差比较（`allclose`），不要 `assert_equal`
- 算子专用类型定义放在独立内部头文件（如 `conv_int.h`、`pooling_int.h`），`.c` 和 `.cu` 共用

## 常用命令

```bash
# === 完整构建（Windows，CUDA 模式）===
cmake -B build -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DCUDA_ARCH=86
cmake --build build --config Release -j$(nproc)

# === 仅 CPU 模式（Windows / Linux 通用）===
cmake -B build -DENABLE_CUDA=OFF -DENABLE_TESTS=ON
cmake --build build -j$(nproc)

# === 运行测试 ===
ctest --test-dir build -C Release -j$(nproc)

# === 批量 benchmark ===
./scripts/run_benchmarks.sh              # 运行所有 bench
./scripts/run_benchmarks.sh --profile-only  # 只运行算子级 profile

# === GPU 内存检查 ===
./scripts/run_sanitizer.sh               # 批量 compute-sanitizer
compute-sanitizer ./build/Release/test_matmul.exe

# === 代码质量检查 ===
./scripts/check_raw_cuda.sh              # 检测裸 CUDA API 调用
./scripts/check_registry.sh              # 验证 .def 注册一致性
./scripts/check_perf_regression.sh /tmp/current.csv docs/PROFILE_BASELINE.csv
```

## 文档索引

| 文档 | 内容 | 什么时候看 |
| --- | --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、CUDA 平台、算子规范 | 开始写代码前 |
| [docs/CODING_STYLE.md](docs/CODING_STYLE.md) | C 命名/错误处理/头文件规范 | 写/审代码时 |
| [docs/CUDA_GUIDE.md](docs/CUDA_GUIDE.md) | CUDA kernel 模板、错误检查、优化策略 | 写/审 CUDA kernel 时 |
| [docs/TASK_TEMPLATES/](docs/TASK_TEMPLATES/) | 新增算子、加 benchmark、加测试 | 接到具体任务时 |
| [docs/TODO_LIST.md](docs/TODO_LIST.md) | 当前待办任务 | 了解当前进度 |
