# CudaForge

<div align="center">

**轻量级 CUDA 神经网络推理引擎 — 零外部依赖，兼容 ONNX。**

[![CUDA](https://img.shields.io/badge/CUDA-13.2-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C Standard](https://img.shields.io/badge/C-11-A8B9CC?logo=c)](https://en.cppreference.com/w/c/11)
[![C++ Standard](https://img.shields.io/badge/C++-20-00599C?logo=cplusplus)](https://en.cppreference.com/w/cpp/20)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-808080)](https://github.com/windward47/cudaforge)

[English](README_en.md) | 简体中文

</div>

---

CudaForge 是一个**从零手写的 CUDA 推理引擎**，纯 C 语言实现。能加载标准 `.onnx` 模型并在 GPU 上执行推理——不依赖 cuDNN、不依赖 TensorRT、不依赖 protobuf。只需 CMake、C 编译器加 CUDA Toolkit 即可构建。

## 为什么选择 CudaForge？

| 对比维度 | CudaForge | 典型推理框架 |
| --- | --- | --- |
| 外部依赖 | **零**（自包含 protobuf 解析器） | cuDNN + TensorRT + protobuf + ... |
| 二进制体积 | ~500 KB | 100+ MB |
| 构建时间 | ~10 秒 | 数分钟 |
| 可读性 | 纯 C，一个下午就能通读 | 数百万行框架代码 |

CudaForge 的定位是**学习、原型验证、嵌入式部署**——不是为了在跑分上击败 TensorRT。每个 CUDA kernel 都是手写的，并配有纯 C fallback，可以清晰追踪数据从 ONNX 权重 → GPU 内核 → 推理结果的完整路径。

## 架构

```text
┌──────────────────────────────────────────────────┐
│  应用层 (Application)                              │
│  ONNX 加载器 · 计算图构建 · 推理会话 API            │
├──────────────────────────────────────────────────┤
│  算子层 (Operator)                                 │
│  ReLU · Conv2D · MatMul · Pool · BatchNorm · ...  │
├──────────────────────────────────────────────────┤
│  平台层 (Platform)                                 │
│  CPU 抽象 · CUDA 显存管理 · Stream 调度            │
└──────────────────────────────────────────────────┘
```

**严格分层依赖**：Application → Operator → Platform，禁止反向或跨层依赖。每个算子同时提供 CUDA kernel 和纯 C fallback，方便验证正确性。

## 快速开始

### 环境要求

- CMake ≥ 3.18
- C/C++ 编译器（MSVC 2022 / GCC / Clang）
- CUDA Toolkit ≥ 12.0
- Python ≥ 3.8（可选，用于生成测试模型）

### 构建

```bash
git clone https://github.com/windward47/cudaforge.git
cd cudaforge

# CUDA 模式（默认）— RTX 2050 对应 sm_86
cmake -B build -G "Visual Studio 17 2022" -A x64 \
      -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DCUDA_ARCH=86
cmake --build build --config Release -j$(nproc)

# 纯 CPU 模式
cmake -B build -DENABLE_CUDA=OFF -DENABLE_TESTS=ON
cmake --build build -j$(nproc)
```

### 运行

```bash
# 快速验证 — 打印平台信息
./build/Release/cudaforge.exe

# 完整测试套件（31 个测试程序）
ctest --test-dir build -C Release -j$(nproc)

# GPU 内存安全检查
compute-sanitizer ./build/Release/test_onnx.exe
```

预期输出：

```text
CudaForge v0.1.0
Platform: x86_64 (8 cores, 64 B cache line)
CUDA: enabled (1 device(s))
```

## 支持的算子

| 算子 | ONNX Op | CPU | CUDA | 备注 |
| --- | --- | --- | --- | --- |
| ReLU | `Relu` | ✓ | ✓ | 逐元素，支持 in-place |
| Sigmoid | `Sigmoid` | ✓ | ✓ | 逐元素 |
| GELU | `Gelu` | ✓ | ✓ | 高斯误差线性单元 |
| Conv2D | `Conv` | ✓ | ✓ | Direct + shared memory tiled 内核 |
| MatMul/Gemm | `MatMul` / `Gemm` | ✓ | ✓ | Warp-tiled 32×32 内核 |
| MaxPool2D | `MaxPool` | ✓ | ✓ | |
| AvgPool2D | `AveragePool` | ✓ | ✓ | |
| BatchNorm | `BatchNormalization` | ✓ | ✓ | Coalesced + cross-block reduction |
| Add | `Add` | ✓ | ✓ | 逐元素加法，支持 broadcast |
| Reshape | `Reshape` | ✓ | — | 零拷贝形状变换 |
| GlobalAveragePool | `GlobalAveragePool` | ✓ | ✓ | NCHW → NC×1×1 全局平均池化 |
| Softmax | `Softmax` | ✓ | ✓ | 沿轴 softmax，分类网络输出层 |
| SiLU | `SiLU` | ✓ | ✓ | Sigmoid 线性单元 |
| Mul | `Mul` | ✓ | ✓ | 逐元素乘法，支持 broadcast |
| Concat | `Concat` | ✓ | ✓ | 沿通道轴拼接 |
| Resize | `Resize` | ✓ | ✓ | 最近邻上采样 |
| Transpose | `Transpose` | ✓ | ✓ | N 维转置 |
| Sub | `Sub` | ✓ | ✓ | 逐元素减法，支持 broadcast |
| Div | `Div` | ✓ | ✓ | 逐元素除法，支持 broadcast |
| Slice | `Slice` | ✓ | ✓ | 沿多轴切片 |
| Split | `Split` | ✓ | ✓ | 沿指定轴拆分 |
| LayerNorm | `LayerNormalization` | ✓ | ✓ | 共享内存归约，最后一维归一化 |
| Gather | `Gather` | ✓ | ✓ | 沿轴按索引查表，支持 float 索引 |
| Squeeze/Unsqueeze | `Squeeze` / `Unsqueeze` | ✓ | ✓ | 设备端 memcpy，零拷贝 reshape |
| Exp | `Exp` | ✓ | ✓ | 逐元素指数，融入 activations 框架 |
| ReduceSum/Max | `ReduceSum` / `ReduceMax` | ✓ | ✓ | 共享内存 stride 归约，沿指定轴求和/取最大 |
| Cast | `Cast` | ✓ | ✓ | int64↔F32 类型转换，支持 ONNX `to` 属性 |
| ArgMax | `ArgMax` | ✓ | ✓ | 共享内存 (value,index) 对归约，返回最大值索引 |
| MHA_Fused | S-Attn 子图融合 | ✓ | ✓ | QKV 投影 + Multi-Head Attention + 输出投影融合为单 kernel |

## API 概览

### 高层 API：加载 ONNX 模型并推理

```c
#include "inference_engine.h"
#include "platform.h"

platform_init();
operator_init_all();
#ifdef USE_CUDA
cuda_platform_init(0);
#endif

// 加载模型
inference_session_t* session = inference_session_load("model.onnx");

// 准备输入输出
int64_t shape[] = {1, 4};
tensor_t* input  = tensor_create(DATA_TYPE_F32, 2, shape);
tensor_t* output = tensor_create(DATA_TYPE_F32, 2, shape);
// ... 填充 input->data ...

// 执行推理（1 = GPU, 0 = CPU）
tensor_t* inputs[]  = { input };
tensor_t* outputs[] = { output };
inference_session_run(session, inputs, outputs, 1);

// CUDA 模式下需将结果拷回主机
if (output->data_device) tensor_copy_to_host(output);
// ... 读取 output->data ...

inference_session_destroy(session);
platform_finalize();
```

### 底层 API：手动构建计算图

```c
graph_t* g = graph_create();
int tid_in  = graph_add_tensor(g, tensor_create(DATA_TYPE_F32, 2, shape));
int tid_out = graph_add_tensor(g, tensor_create(DATA_TYPE_F32, 2, shape));

graph_add_node(g, OP_RELU, 1, (int[]){tid_in}, 1, (int[]){tid_out}, 0, NULL, NULL, 0);
graph_build(g);
graph_execute(g, inputs, outputs, false);  // false = CPU 执行
graph_destroy(g);
```

详细 API 文档见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 开发指南

### 项目结构

```text
cudaforge/
├── src/
│   ├── platform/          # 硬件抽象层（CPU + CUDA）
│   │   ├── cpu/           #   x86 CPU 实现
│   │   ├── cuda/          #   CUDA 设备/显存/Arena
│   │   └── include/       #   platform.h, cuda_ops.h
│   ├── operator/          # 算子实现
│   │   ├── blas/          #   MatMul
│   │   ├── nn/            #   ReLU, Conv2D, Pool, BatchNorm, Activations
│   │   └── include/       #   operator.h
│   ├── application/       # 推理引擎 + 模型加载
│   │   ├── engine/        #   图构建器，推理会话
│   │   ├── model/         #   ONNX protobuf 解析器 + 加载器
│   │   └── include/       #   inference_engine.h
│   └── main.c             # 入口
├── tests/                 # 集成测试 + 性能基准
├── docs/                  # 架构文档、编码规范、CUDA 开发指南
└── third_party/unity/     # Unity 测试框架（vendored）
```

### 添加新算子

所有算子遵循统一的开发流程。以添加 `Softmax` 为例：

1. **创建内部头文件** — `src/operator/nn/softmax_int.h`，定义 `softmax_params_t`
2. **编写 CPU fallback** — `src/operator/nn/softmax.c`（纯 C，始终可用）
3. **编写 CUDA kernel** — `src/operator/nn/softmax_cuda.cu`（GPU 加速）
4. **注册算子** — 在 `operator_registry.c` 和 `operator.h` 中添加枚举值
5. **添加形状推导** — 在 `onnx_loader.c` 的 `infer_output_shape()` 中添加推导逻辑
6. **添加 ONNX 映射** — 在 `onnx_loader.c` 中建立 `"Softmax"` → 新算子的映射
7. **编写测试** — `src/operator/nn/test/test_softmax.c`（CPU + CUDA 都要测）
8. **更新 CMakeLists.txt** — 添加源文件和测试目标
9. **运行 compute-sanitizer** — 零错误才能合并

详细模板见 [docs/TASK_TEMPLATES/new_operator.md](docs/TASK_TEMPLATES/new_operator.md)。

### 参与贡献

1. Fork 本仓库
2. 创建特性分支（`git checkout -b feat/my-operator`）
3. 遵循[编码规范](docs/CODING_STYLE.md)和 [CUDA 开发指南](docs/CUDA_GUIDE.md)
4. 为 CPU fallback 和 CUDA kernel 分别编写测试
5. 对 CUDA 测试运行 `compute-sanitizer`（零错误）
6. 运行完整测试套件：`ctest --test-dir build -C Release`
7. 提交 PR，附上清晰的改动说明

**提交规范**：每次提交只做一件事。用 `feat:` / `fix:` / `refactor:` 前缀区分提交类型。

### 构建选项

| 选项 | 说明 | 默认值 |
| --- | --- | --- |
| `ENABLE_CUDA` | 启用 CUDA GPU 后端 | `ON` |
| `ENABLE_TESTS` | 构建测试套件 | `ON` |
| `ENABLE_AVX2` | 启用 AVX2 CPU 优化 | `OFF` |
| `ENABLE_AVX512` | 启用 AVX-512 CPU 优化 | `OFF` |
| `ENABLE_OPENMP` | 启用 OpenMP 并行 | `ON` |
| `ENABLE_COVERAGE` | 启用代码覆盖率 | `OFF` |
| `CUDA_ARCH` | CUDA 计算能力（如 `86`） | `86` |

## ONNX 兼容性

CudaForge 内置了一个**手写的 protobuf wire-format 解析器**（约 200 行 C），无需安装 protobuf 库。支持：

- 标准 `.onnx` 文件（proto2 和 proto3）
- IR version ≤ 13, opset ≤ 11
- `float_data` 和 `raw_data` 两种权重存储格式
- 所有已支持算子的自动形状推导

**局限性**：不支持动态 shape、不支持 INT8/FP16 量化、仅支持上表列出的算子。

## 常见问题

**Q: 为什么不直接用 ONNX Runtime / TensorRT？**
CudaForge 的初衷是学习推理引擎的底层原理。通读全部代码只需一个下午。同时适用于无法承载 100+ MB 依赖的嵌入式场景。

**Q: 能跑 ResNet / YOLO / BERT 吗？**
已通过五层验证：(1) MNIST CNN（Conv×2 + ReLU + MaxPool + Reshape + Gemm + Softmax）；(2) **ResNet-18**（1×3×224×224 → 1×1000，50 节点，CUDA 推理与 PyTorch 对比 max_diff = 5.25e-06，Top-1 一致）；(3) **YOLOv8n**（1×3×640×640 → 1×84×8400，234 节点，CPU/CUDA 推理 max_diff < 1e-3，compute-sanitizer 零错误）；(4) **BERT-like Phase A**（Embedding + LayerNorm + FFN + SiLU，CPU/CUDA 推理与 ONNX Runtime 对比 max_diff = 5.96e-07）；(5) **BERT-base Phase B**（Embedding + LayerNorm + FFN + Exp + ReduceSum/Max + ArgMax + Cast + Softmax + Concat，CPU max_diff=2.24e-08，CUDA max_diff=7.45e-09，compute-sanitizer 零错误）；(6) **BERT-base Phase C**（MHA 融合 kernel，22 节点 self-attention 子图 → 1 个 MHA_Fused kernel + 1 个 output MatMul，CPU vs CUDA 相对误差 < 1.3e-06，compute-sanitizer 零错误，BERT-base 单层 CUDA 推理 30.35 ms/iter）。覆盖 31 种算子（含 MHA_Fused 融合），**31 项测试全部通过**。

**Q: 支持 FP16 或 INT8 推理吗？**
当前仅支持 FP32。混合精度和量化推理在后续计划中。

**Q: 如何调试 CUDA kernel？**
使用 CUDA Toolkit 自带的 `compute-sanitizer`：

```bash
compute-sanitizer --tool memcheck ./build/Release/test_conv.exe
```

## 开源协议

MIT — 详见 [LICENSE](LICENSE)。

## 致谢

- [Unity Test Framework](https://github.com/ThrowTheSwitch/Unity) — C 语言单元测试框架（已 vendored）
- [ONNX](https://onnx.ai/) — 开放神经网络交换格式标准
