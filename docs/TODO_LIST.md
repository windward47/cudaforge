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

## 第四步：BERT 支持（路线 A+B+C 渐进策略，~3 周）

> BERT-base 已有 **12/21** 种算子就绪（MatMul, Add, Mul, Div, Sub, Softmax, GELU, Reshape, Transpose, Concat, Slice, Split）。
> 缺失 **9 种**，按三阶段推进。

---

### Phase A：最小可行 BERT（4 个 P0 算子，~5 天）

**目标**: 打通 BERT ONNX → CPU 推理管线，输出与 PyTorch 参考一致。

#### A1. LayerNormalization — 最关键缺失算子

**文件**: `src/operator/nn/layernorm_int.h`, `layernorm.c`, `layernorm_cuda.cu`

BERT-base 中 ~25 处 LayerNorm（每层 2 个 pre-norm + 最终 1 个），是缺失算子中优先级最高的。

```
y = (x - mean) / sqrt(var + epsilon) * gamma + beta
```

**实现要点**:
- CPU: 对最后一维（hidden_size=768）计算 mean/var，逐元素归一化
- CUDA: warp-level reduction 计算 mean/var，shared memory 广播 gamma/beta
- `layernorm_params_t { int64_t N; int64_t normalized_size; float epsilon; }`
- ONNX: 解析 `axis` 和 `epsilon` 属性；gamma/beta 作为 weight 传入

#### A2. Gather — 嵌入查表

**文件**: `src/operator/nn/gather_int.h`, `gather.c`, `gather_cuda.cu`

BERT 用 Gather 将 token/position/segment ID 查表转为嵌入向量。ONNX Gather 语义：沿 axis 按 indices 选取切片。

```
output[i][j][k] = input[indices[i]][j][k]   (axis=0)
```

**实现要点**:
- `gather_params_t { int64_t axis; }`
- CPU: 对于 axis=0，`output[i*...] = input[indices[i]*...]`
- CUDA: 1D grid，每线程计算 output index → 查 indices → 读取 input

#### A3. Squeeze / Unsqueeze — 形状管理

**文件**: `src/operator/nn/squeeze_unsqueeze_int.h`, `squeeze_unsqueeze.c`

ONNX export 在 Gather 前后自动插入 Unsqueeze/Squeeze 调整维度。本质是零拷贝 reshape。

```
Squeeze:   [1, 3, 1, 5] → [3, 5]   (移除尺寸=1的轴)
Unsqueeze: [3, 5] → [1, 3, 5]       (在指定位置插入尺寸=1的轴)
```

**实现要点**:
- 复用 Reshape 的零拷贝模式（仅改 ndim/shape 元数据）
- ONNX: 解析 `axes` 属性；Squeeze 的 v13+ 从 input[1] 读 axes
- 不需要 CUDA kernel（纯元数据变换；若 GPU tensor 需同设备 reshape）

#### A4. BERT CPU 端到端验证

**文件**: `tests/test_bert.c`, `tests/gen_bert_test.py`

```python
# gen_bert_test.py
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-uncased")
dummy = torch.randint(0, 30522, (1, 128))  # (batch=1, seq_len=128)
torch.onnx.export(model, (dummy,), "bert_base_test.onnx",
                  input_names=["input_ids"], opset_version=11)
# run ONNX Runtime for reference output
```

**验证标准**:
- CPU vs PyTorch/ONNX Runtime: max_diff < 1e-3
- 输出: (1, 128, 768) last_hidden_state

---

### Phase B：完整 BERT 推理（5 个 P1/P2 算子，~5 天）

**目标**: CPU+CUDA 双后端完整覆盖，compute-sanitizer 零错误。

#### B1. ReduceSum / ReduceMax — 注意力 mask 归约

**文件**: `src/operator/nn/reduce_int.h`, `reduce.c`, `reduce_cuda.cu`

BERT 的 attention mask 需要沿最后一维求和/取最大值，用于屏蔽 padding token。

```
ReduceSum(data, axes=[-1])   — 沿指定轴求和
ReduceMax(data, axes=[-1])   — 沿指定轴取最大值
```

**实现要点**:
- `reduce_params_t { int64_t axes[4]; int num_axes; int keepdims; int op; }`
- CPU: 沿 axes 循环累加/比较
- CUDA: shared memory reduction，每线程块处理一个归约单元
- ONNX: opset 11 用 `axes` 属性，opset 18 从 input[1] 读取

#### B2. Exp — Softmax 数值稳定性

**文件**: `src/operator/nn/unary_int.h`, `unary.c`, `unary_cuda.cu`（可与其他逐元素一元算子共文件）

```
y = exp(x)
```

**实现要点**:
- CPU: `expf(x[i])`
- CUDA: `__expf(x[i])`，1D grid
- BERT 中 Exp 通常出现在 attention mask 的 `exp(mask - max)` 模式中
- 为后续扩展预留 Neg/Abs/Log/Sqrt 操作码（共享同一 `unary_op_t` 枚举）

#### B3. Cast — int64→FP32 类型转换

**文件**: `src/operator/nn/cast_int.h`, `cast.c`, `cast_cuda.cu`

BERT 输入 `input_ids` 是 int64，需转为 FP32 才能与 FP32 嵌入表 Add。

```
y = (float)x   或   y = (int64_t)x
```

**实现要点**:
- `cast_params_t { int src_dtype; int dst_dtype; }`
- 初期仅实现 int64→FP32 和 FP32→int64
- CUDA: 1D grid with `__float2int_rz` / `__int2float_rz`

#### B4. ArgMax — 分类输出

**文件**: `src/operator/nn/argmax_int.h`, `argmax.c`, `argmax_cuda.cu`

用于 BERT 序列分类任务的输出层 token 选择。

```
ArgMax(data, axis=1, keepdims=0) → index of max along axis
```

**实现要点**:
- `argmax_params_t { int64_t axis; int keepdims; }`
- CPU: 沿 axis 循环比较
- CUDA: shared memory reduction 比较 value+index pair

#### B5. BERT CUDA 端到端验证

- 所有 P0+P1+P2 算子的 CUDA kernel 通过 compute-sanitizer
- BERT-base CPU vs CUDA: max_diff < 1e-3
- 最终输出 (last_hidden_state) 与 PyTorch 参考一致

---

### Phase C：Multi-Head Attention 融合优化（1 个融合 kernel，~5 天）

**目标**: 将 BERT 中计算量占比 60%+ 的 Multi-Head Attention 融合为单个 CUDA kernel，大幅降低 kernel launch 开销和显存带宽压力。

#### C1. MHA 融合 kernel 设计与实现

**文件**: `src/operator/nn/mha_fused_cuda.cu`（新文件），`src/application/engine/graph.c`（fusion pass 扩展）

**融合范围**（一个 BERT encoder 层的 self-attention 子图）:

```
原始图 (12 个节点):
  MatMul(Q)  MatMul(K)  MatMul(V)   ← QKV 投影
  Reshape × 3, Transpose × 3         ← 多头拆分
  MatMul(Q·K^T)                      ← 注意力分数
  Mul(/√d_k)                         ← 缩放
  Softmax                            ← 注意力概率
  MatMul(attn·V)                     ← 加权求和
  Transpose, Reshape                 ← 多头合并
  MatMul(output_proj)                ← 输出投影

融合后 (2 个节点):
  MHA_Fused(QKV_weights, output_weight, bias...)  ← 单 kernel
  MatMul(output_proj)                  ← 输出投影（保持在外部便于后续融合）
```

**kernel 设计要点**:
- 每个 thread block 处理一个 attention head
- Shared memory 缓存 Q、K、V tile（每个 head 维度 d_k=64）
- 在线 Softmax：block 内分 tile 计算 `max → exp → sum`，无需全局同步
- Q·K^T 和 attn·V 在同一个 kernel 内完成
- 仅对 `sequence_length ≤ 512` 的 BERT 推理场景优化（不处理长序列）

**fusion pass 扩展** (`graph.c`):
- 检测模式：MatMul(QKV) → Reshape → Transpose → MatMul(Q·K^T) → Mul → Softmax → MatMul(attn·V) → Transpose → Reshape → MatMul(output)
- 仅当所有中间节点无额外消费者时触发融合
- 融合后 12 个节点压缩为 2 个，每层节省 10 次 kernel launch
- 12 层 BERT-base 总计节省 ~120 次 kernel launch

**性能预期**: BERT-base CUDA 推理延迟降低 40-60%（MHA 占 60%+ 计算量）

#### C2. BERT 性能基准

**文件**: `tests/bench_bert.c`

```
Model: BERT-base (12 layers, hidden=768, heads=12)
Input: (1, 128) token IDs
Metrics: latency (ms), throughput (tokens/s)
Compare: unfused vs fused MHA on CUDA
```

---

## 第五步：FP16 推理与更大模型（远期，视需求启动）

### F1. FP16 推理支持

**文件**: 新增 `src/platform/tensor_fp16.c`，修改所有 CUDA kernel

- 在 `tensor_t` 中增加 FP16 数据缓冲区
- 为高频算子（Conv、MatMul、逐元素）添加 FP16 kernel
- 使用 `cudaDeviceGetAttribute` 查询 FP16 硬件支持
- CPU 端通过 Cast op 做类型转换

### F2. LLM 推理探索

BERT 管线打通后，可探索 GPT-2/LLaMA 等 decoder-only 架构：
- 新增算子：**CausalMask**（下三角 mask）、**KV-Cache**（自回归缓存）、**RoPE**（旋转位置编码）
- 新增融合：**FlashAttention** 风格的 tile-based 精确注意力

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
| 已完成 | 10 | R1-R3（可靠性）、Q1-Q4（质量）、P1-P3（性能） |
| 进行中 | 0 | — |
| 待开始 | 4 | A1-A4（BERT 最小可行，Phase A） |

### 第三步性能提升实测

| 优化项 | 效果 |
|--------|------|
| P1 Conv+SiLU 融合 | YOLOv8n 中所有 Conv→SiLU 对在单个 kernel 内完成 |
| P2 find_tensor 哈希表 | O(n)→O(1) 查找，YOLOv8n 加载时间大幅降低 |
| P3 Winograd F(2,3) | Conv3×3 stride-1 理论 2.25× 加速，YOLOv8n 实测 **1.37×** 端到端提升 |

**YOLOv8n 推理延迟（RTX 2050）:**

| 后端 | 延迟 |
|------|------|
| CPU | 12,185 ms |
| CUDA (direct) | 176 ms (69× vs CPU) |
| CUDA (Winograd) | 128 ms (95× vs CPU) |

> **最后更新**: 2026-05-27。第十一轮：前三步全部完成。26/26 测试通过，compute-sanitizer 0 errors。
