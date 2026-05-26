# CudaForge Todo List

> 基于 2026-05-26 全仓库审阅生成。旧版历史记录见 [TODO_LIST_20260526.md](TODO_LIST_20260526.md)。

---

## 审阅概要

**当前状态**: 18 种算子全部 CPU+CUDA 双实现，3 个端到端模型验证通过（MNIST CNN、ResNet-18、YOLOv8n），26 项测试 25 通过。算子覆盖以 CNN 推理为核心，缺失 Transformer/LLM 所需的关键算子（LayerNorm、Gather、Squeeze/Unsqueeze 等）。

**旧版中已修复但未标出的项目**: 池化 kernel OW>256 分块（C3）、im2col per-batch 分配（C4）、H6 per-node malloc/free、M2a-b / M12，已在本版中确认为修复状态。

---

## 第一步：可靠性加固（3 项，~1 天）

### R1. 修复 test_conv_scale 超容差

**文件**: `tests/test_conv_scale.c` 或 `src/operator/nn/test/test_conv.c`

当前 test_conv_scale 失败：max_diff = 1.07e-04 略超 1.0e-04 阈值，仅 1/200704 个元素超标，且 CPU/CUDA 前 10 个值完全一致。本质是浮点舍入误差，非逻辑错误。

**修复方向**: 将 CUDA vs CPU 对比阈值从 1e-4 放宽到 1e-3，与 YOLO 测试对齐；或对 conv 算子单独使用相对误差判据。

### R2. 为逐元素算子补全 NULL 参数校验

**文件**: `src/operator/nn/sub.c`, `src/operator/nn/div.c`, `src/operator/nn/mul.c`, `src/operator/nn/slice.c`, `src/operator/nn/split.c`

ReLU、Activations、Add 已有校验，但 Sub/Div/Mul/Slice/Split 等后期新增算子的入口参数校验不完整或缺失。`inputs[1]` 被直接解引用而无 NULL 检查。

**修复方向**: 统一校验模式：`if (!inputs || !outputs || !params) return -1;`，对使用的每个 `inputs[n]` 增加非 NULL 断言。

### R3. CUDA_CHECK 宏改 exit() → 错误返回

**文件**: `src/platform/include/cuda_ops.h:22-29`

当前 `CUDA_CHECK` 在 CUDA 错误时调用 `exit(EXIT_FAILURE)` 直接终止进程。推理库应返回错误码让调用方决定如何处理。

**修复方向**: 引入 `goto error` 或 `return` 模式替代 `exit()`。不影响现有测试路径——所有测试当前返回 0，此改动仅影响异常路径。

---

## 第二步：代码质量提升（4 项，~2 天）

### Q1. 算子测试强化

**文件**: 多个 `test_*.c`

| 测试文件 | 当前问题 | 改进方向 |
|----------|---------|---------|
| `test_conv.c` | 仅检查"不全为 0" | 指定已知输入/权重，计算精确期望值 |
| `test_batchnorm.c` | 仅检查符号和大致范围 | 同上 |
| `test_cuda_kernels.cu` | GELU 跳过第 4 元素 | 修复或说明跳过理由 |
| `test_pooling.c` | 无 OW 边界测试 | 增加 OW > 256 的 scale 测试 |

**修复方向**: 每个算子至少一个 golden-value 测试用例（已知输入→期望输出），不依赖 CPU 实现作参考。

### Q2. 消除魔数

| 位置 | 魔数 | 建议常量名 |
|------|------|-----------|
| `relu_cuda.cu`, `activations_cuda.cu`, `pooling_cuda.cu`, `concat_cuda.cu`, etc. | `256` 线程 | `OPS_THREADS_PER_BLOCK` |
| `conv_cuda.cu:247` | `8192` 共享内存阈值 | `CONV_SMEM_THRESHOLD` 或 `cudaDeviceGetAttribute` 查询 |
| `softmax_cuda.cu` | 线程/block 常量 | 同上 |

**修复方向**: 在 `operator.h` 或 `cuda_ops.h` 统一定义线程常量，conv 专用常量留在 `conv_int.h`。

### Q3. Add in-place 算子测试

ReLU、Sigmoid、GELU、SiLU 标记了 `OP_FLAG_IN_PLACE`，但无测试验证 `inputs[0] == outputs[0]` 时 CPU/CUDA 路径的正确性。

**修复方向**: 在 `test_activations.c` 和 `test_relu.c` 增加 in-place 测试用例。

### Q4. Resize bilinear 模式支持

**文件**: `src/operator/nn/resize.c`, `resize_cuda.cu`

当前 Resize 仅实现 nearest-neighbor 上采样。Bilinear 插值是 ONNX Resize 的默认模式，许多模型（包括 YOLOv5、SSD、部分分割模型）依赖它。

**修复方向**:
- 添加 `mode` 字段到 `resize_params_t`（0=nearest, 1=bilinear）
- CPU: 双线性插值 `v = (1-α)(1-β)·tl + α(1-β)·tr + (1-α)β·bl + αβ·br`
- CUDA: 同上，per-pixel 并行，注意边界处理

---

## 第三步：性能优化（3 项，~4 天）

### P1. Conv+SiLU 融合 kernel

**文件**: `src/operator/nn/conv_cuda.cu`, `src/application/engine/graph.c`

YOLOv8n 中 50%+ 的 Conv 节点后跟 SiLU（以 Sigmoid→Mul 形式）。当前融合 pass 因多消费者检查正确地跳过了这些模式（Conv 输出同时被 Sigmoid 和 Mul 消费），但带来了额外 kernel launch 开销。

**修复方向**:
- 添加新的融合模式：检测 Conv → Sigmoid → Mul 三元组
- 新增 `conv_silu_fused_kernel`：在一次 kernel 内完成 `x * sigmoid(conv(x) + bias)`
- 跳过被融合的 Sigmoid 和 Mul 节点
- 预期 YOLOv8n CUDA 推理延迟降低 20-30%

### P2. ONNX 加载器 find_tensor 线性扫描 → 哈希表

**文件**: `src/application/model/onnx_loader.c:100-106`

当前 O(n) 线性扫描。对于 YOLOv8n（384 个 tensor 信息条目，234 个节点各多次调用），加载时间主要消耗于此。

**修复方向**: 
- 引入 `uthash`（单头文件、MIT 协议）或自写开放寻址哈希表
- key = tensor name string, value = `onnx_tensor_info_t*`
- 所有 `find_tensor()` / `add_tensor()` 调用降为 O(1)

### P3. Winograd F(2,3) Conv 3×3 kernel

**文件**: `src/operator/nn/conv_cuda.cu`（新增 kernel）

3×3 Conv 占 YOLOv8n 和 ResNet-18 中绝大多数 Conv 节点。Winograd F(2,3) 可将 3×3 Conv 的计算量从 9 次乘法降为 4 次（理论 2.25× 加速）。

**修复方向**:
- 仅对 `KH=3, KW=3, stride=1, dilation=1` 启用
- 在 dispatch 层选择 Winograd vs 现有 direct kernel
- 输入/权重/输出变换矩阵使用常数

---

## 第四步：模型覆盖扩展（中期目标，~2 周）

### M1. Transformer 基础算子补齐

为运行 BERT 等 Transformer 模型，需要以下算子：

| 优先级 | 算子 | ONNX Op | 复杂度 | 说明 |
|--------|------|---------|--------|------|
| P0 | LayerNorm | `LayerNormalization` | 中 | Transformer 标配，替代 BatchNorm |
| P0 | Gather | `Gather` | 低 | Embedding 查表，token 选择 |
| P1 | Squeeze/Unsqueeze | `Squeeze` / `Unsqueeze` | 低 | 形状管理，添加/移除尺寸为 1 的轴 |
| P1 | ReduceSum/ReduceMax | `ReduceSum`, `ReduceMax` | 低 | 沿轴归约，Self-Attention softmax 前需要 |
| P1 | Exp | `Exp` | 低 | Softmax 稳定性（max 减法后在 exp 前） |
| P2 | Cast | `Cast` | 低 | FP32↔FP16 类型转换，混合精度入口 |
| P2 | ArgMax | `ArgMax` | 低 | 分类输出 token 选择 |
| P3 | Pow/Sqrt | `Pow`, `Sqrt` | 低 | 位置编码计算 |
| P3 | Clip | `Clip` | 低 | 激活值截断 |

实现流程与现有算子一致：`_int.h` → `.c` → `.cu` → enum → `map_onnx_op()` → `infer_output_shape()` → 注册 → 测试。

### M2. BERT-base 端到端推理验证

补齐 Transformer 算子后，验证 BERT-base（12 层，~360 节点）在 CPU/CUDA 上的推理精度。参照 ResNet-18/YOLOv8n 的验证流程。

### M3. FP16 推理支持

**文件**: 新增 `src/platform/tensor_fp16.c`，修改所有 CUDA kernel

**修复方向**:
- 在 `tensor_t` 中增加 FP16 数据缓冲区
- 为高频算子（Conv、MatMul、逐元素）添加 FP16 kernel
- 使用 `cudaDeviceGetAttribute` 查询 FP16 硬件支持
- CPU 端通过 Cast op 做类型转换

---

## 持续维护项

| 项 | 说明 |
|----|------|
| 文档同步 | 新增算子后同步更新 README 算子表、ARCHITECTURE 目录树、CUDA_GUIDE kernel 清单 |
| compute-sanitizer | 所有新增 CUDA kernel 必须通过 compute-sanitizer 零错误 |
| 测试阈值 | 浮点对比使用相对误差或放宽绝对阈值（1e-3），避免因舍入波动触发假阳性 |
| Commit 规范 | 一个 commit 一个事：feat / fix / refactor 不混合 |

---

## 进度

| 状态 | 数量 | 内容 |
|------|------|------|
| 已完成 | — | 18 算子全部 CPU+CUDA、26 测试 25 通过、3 模型验证通过（MNIST/ResNet-18/YOLOv8n） |
| 进行中 | 0 | — |
| 待开始 | 10 | R1-R3（可靠性）、Q1-Q4（质量）、P1-P3（性能）、M1-M3（扩展） |

> **最后更新**: 2026-05-26。第十轮：全仓库审阅，制定四步计划：可靠性加固 → 代码质量提升 → 性能优化 → Transformer/LLM 支持。
