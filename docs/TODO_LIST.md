# Todo list

## 已完成

- [x] 整理编码规范 → 保存到 `docs/CODING_STYLE.md`
- [x] 整理架构约束（含 CUDA 扩展）→ 保存到 `docs/ARCHITECTURE.md`
- [x] 编写 Claude 指令 → 保存到 `CLAUDE.md`
- [x] 编写 CUDA 开发指南 → 保存到 `docs/CUDA_GUIDE.md`
- [x] 创建任务模板 → `docs/TASK_TEMPLATES/`
- [x] 搭建 `src/` 源码骨架（Platform/Operator/Application 三层 + CMakeLists）
- [x] 实现平台适配层（CPU x86 + CUDA 内存/设备管理）
  - [x] `g_platform` ops 表（aligned_alloc/core count/cache line）
  - [x] `g_cuda` ops 表（device/stream/memory 函数指针）
  - [x] tensor_create/destroy + tensor_copy_to_device/host 真实传输
  - [x] platform_init() / platform_finalize() 显式初始化入口
  - [x] cuda_platform_init() 将 memory ops 挂载到 g_cuda
- [x] 实现核心算子（ReLU、MatMul、Conv2D、Pooling、BatchNorm、Activations）
  - [x] ReLU（CPU + CUDA kernel）
  - [x] MatMul（CPU 朴素 + CUDA naive/tiled shared memory）
  - [x] Conv2D（CPU ref 7 层循环 + im2col，CUDA im2col + matmul）
  - [x] MaxPool2D / AvgPool2D（CPU + CUDA kernel）
  - [x] BatchNorm（CPU + CUDA kernel）
  - [x] Sigmoid / GELU（CPU + CUDA kernel）
- [x] 实现推理引擎（DAG 图 + 拓扑排序 + CPU/CUDA 调度）
  - [x] graph.h — 图数据结构（节点/张量槽/输入输出标记）
  - [x] graph.c — Kahn 拓扑排序
  - [x] graph_execute — operator 查找、输入输出拼装、activation/batchnorm 元数据传递、CPU/CUDA 分发
  - [x] 集成测试（Input→ReLU→Output 链路、多节点链、环检测、空指针安全）
- [x] 编写单元测试与基础集成测试
  - [x] test_relu（正常 + 空指针）
  - [x] test_matmul（正常 + 转置 + 空指针）
  - [x] test_conv（正常 + 空指针）
  - [x] test_activations（sigmoid + gelu + 空指针）
  - [x] test_pooling（maxpool + avgpool + 空指针）
  - [x] test_batchnorm（正常 + 空指针）
  - [x] test_graph（ReLU 链路 + 环检测 + 空指针）

## 待完成

### P0 — 必须优先解决

- [x] **CUDA 编译链路打通（Windows MSVC 工具链）**
  - [x] MinGW GCC 无法驱动 nvcc → 已切换到 MSVC（cl.exe）+ VS Build Tools 2022
  - [x] CUDA 13.2 MSBuild integration 已手动注册（.props/.targets/.xml/.dll）
  - [x] `__CUDACC__` 双编译模式：nvcc 编译 `.cu` 获得真实 CUDA 类型，MSVC 编译 `.c` 获得前向声明
  - [x] C++ variadic template kernel launch 宏（替代不兼容 C++20 的 C99 compound literal）
  - [x] 所有 `.c` / `.cu` 文件 + 11 个构建目标全部通过编译
  - [x] 7 个 CPU fallback 测试全部通过
  - [x] 验证：`cudaforge.exe` 成功初始化 GPU 并打印设备信息
  - [x] CUDA 算子注册系统：`.cu` 文件提供 `extern "C"` 注册函数，`operator_init.c` 统一调用
  - [x] 清理 `operator_registry_t` 中废弃的 `func_cuda` 字段

- [x] **每个 CUDA 算子通过 compute-sanitizer**（CUDA 12.0+ 替代 cuda-memcheck）
  - [x] test_relu — compute-sanitizer 无 memory leak / out-of-bounds
  - [x] test_matmul — compute-sanitizer 无错误
  - [x] test_conv — 修复 dilation 参数未初始化导致 OH/OW 计算错误 → compute-sanitizer 无错误
  - [x] test_pooling — compute-sanitizer 无错误
  - [x] test_batchnorm — compute-sanitizer 无错误
  - [x] test_activations — compute-sanitizer 无错误
  - [x] `tests/test_cuda_kernels.cu` — 统一 CUDA 集成测试（8 个算子全部通过）

### P1 — 推理引擎完善

- [x] **推理引擎边缘情况修复**
  - [x] `graph_execute` 中 OP_OUTPUT 在拓扑循环内的数据拷贝是冗余的（仅索引 `outputs[0]`），后续的循环已正确处理多输出 — 已添加 self-copy 跳过检查
  - [x] `graph_execute` CUDA 路径 bug 修复：device 指针未传入 op（op_inputs/op_outputs 使用 host 指针，现已替换为 device 指针）
  - [x] `graph_add_node` weights 数组 bug 修复：直接存储调用方栈指针导致 free 时崩溃，已改为 malloc 拷贝
  - [x] 空图、单节点图的 graph_build/graph_execute 行为测试
  - [x] 多输入/多输出图的集成测试
  - [x] 权重参数为 NULL 时的容错

- [x] **端到端小模型集成测试**
  - [x] 构建一个小型 CNN 图：Input → Conv2D → ReLU → Output
  - [x] 手工计算预期输出，CPU 路径验证
  - [x] CUDA 路径验证（Input → ReLU → Output 已通过 compute-sanitizer）

- [x] **算子 Benchmark**
  - [x] tests/bench_operators.cu — 统一 benchmark（MatMul/Conv2D/Activations/Pooling/BatchNorm，CPU vs CUDA）

### P2 — 性能优化

- [x] **内存池（Memory Arena）**
  - [x] 设计 Arena 分配器：一次 cudaMalloc 大块，内部偏移分配 → `src/platform/cuda/cuda_arena.cu`
  - [x] API：cuda_arena_init / cuda_arena_alloc / cuda_arena_reset / cuda_arena_destroy
  - [x] 集成到 platform 库，graph_execute 可调用

- [x] **算子 CUDA Kernel 优化（按顺序）**
  - [x] MatMul: naive → tiled shared memory ✅ → warp-level 32×32 tile（+ bank-conflict padding）
  - [x] Conv2D: im2col + gemm ✅ → 直接 CUDA kernel（+ 共享内存输入缓存，避免 im2col 显存开销）
  - [x] BatchNorm: 跨 block 规约 → 一次 pass（inference 合并访存 + training 协作组 grid sync）

### P3 — 可选功能

- [x] **ONNX 模型加载**
  - [x] 自包含最小 protobuf wire-format 解析器（零外部依赖，~200 行 C）
  - [x] ONNX 文件解析 → `inference_graph_t` 转换器（`onnx_parser.c` + `onnx_loader.c`）
  - [x] 常见 op 映射表（Conv → OP_CONV2D, Relu → OP_RELU, Gemm/MatMul → OP_MATMUL, MaxPool → OP_MAXPOOL2D, AveragePool → OP_AVGPOOL2D, BatchNormalization → OP_BATCHNORM, Sigmoid → OP_SIGMOID, Gelu → OP_GELU）
  - [x] 张量形状推导（中间 tensor shape 推断：elementwise / Conv / Pool / MatMul）
  - [x] 支持 proto2 和 proto3 两种 ONNX 格式
  - [x] 测试：4 个内嵌 ONNX 模型（ReLU / ReLU链 / Conv+ReLU / 标准 ONNX 库生成的 Conv+ReLU），CPU + CUDA 路径，28 个测试全部通过
  - [x] compute-sanitizer 零错误
  - [x] 推理引擎会话 API（`inference_session_load/run/destroy`）

- [ ] **ARM 平台支持**
  - [ ] `platform_cpu.c` 添加 ARM 检测分支（`/proc/cpuinfo` / `sysctl`）
  - [ ] ARM NEON 加速路径（算子 SIMD 版本）

- [ ] **混合精度推理（FP16）**
  - [ ] FP16 tensor 支持（data_type.h 扩展）
  - [ ] FP16 CUDA kernel（half2 / __nv_bfloat16）
  - [ ] FP16 fallback CPU 实现
