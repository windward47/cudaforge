# CudaForge Todo List

> 基于 2026-05-28 全状态审阅生成。Phase A/B/C（BERT 全套支持 + MHA 融合优化）已全部完成。旧版历史记录见 [TODO_LIST_20260528.md](TODO_LIST_20260528.md)。

---

## 审阅概要

**当前状态**: 31 种算子全部 CPU+CUDA 双实现，3 个端到端模型验证通过（ResNet-18、YOLOv8n、BERT-base），**31 项测试全通过**。算子覆盖涵盖 CNN 和 Transformer/BERT 推理全链路。Phase C MHA 融合 kernel 已集成，compute-sanitizer 零错误。

---

## 第一步：MHA Kernel 深度优化（3 项，~5 天）

### M1. FlashAttention 风格 tiled 精确注意力

**文件**: `src/operator/nn/mha_fused_cuda.cu`

当前 MHA CUDA kernel 对每个 query 位置、每个 head 重算 K/V（O(B·S·H·S·D·d) 计算量）。FlashAttention 通过分 tile 计算 + online softmax 将显存访问从 O(S²·d) 降到 O(S·d)，适合中等序列长度。

**修复方向**:
- 实现 Q·K^T 的分 tile 计算，每次只加载一个 Q tile 和一个 K tile 到 shared memory
- 用 online softmax 累积 attention 权重和 V 加权和
- 预期 BERT-base (S=8) 延迟再降低 30-50%

### M2. QKV 投影融合

**文件**: `src/operator/nn/mha_fused_cuda.cu`

当前 kernel 对每个 head 分别计算 X·WQ、X·WK、X·WV（三次独立矩阵乘法）。可以将 QKV 权重拼接为 W_QKV (D, 3D)，一次矩阵乘法完成三组投影。

**修复方向**:
- 在 MHA fusion pass 中将 QKV 权重拼接为单一权重张量
- Kernel 内一次性计算 X·W_QKV，结果写入 shared memory 后拆分为 Q/K/V
- 减少 3× 的权重读取带宽

### M3. Tensor Core FP16 MHA

**文件**: `src/operator/nn/mha_fused_cuda.cu`（FP16 版本）

sm_86 (RTX 2050) 支持 Tensor Core FP16。将 MHA 中的矩阵乘法（QKV 投影和 attention 计算）映射到 Tensor Core 指令，理论吞吐量提升 4-8×。

**修复方向**:
- QKV 投影: FP16 wmma
- Attention S·K^T + P·V: 使用 wmma 做批量小矩阵乘法
- 输出投影: FP16 wmma
- 注意: 需要混合精度（accumulator 保持 FP32 精度）

---

## 第二步：ONNX 兼容性修复（2 项，~1 天） — 已完成

### ~~O1. 修复外部数据加载~~ ✅

**文件**: `src/application/model/onnx_loader.c`

添加了完整的错误日志覆盖：fopen 失败、fseek 失败、fread 短读、OOM、路径过长、缺少 location/length。所有失败路径现在都会 `fprintf(stderr, ...)` 输出 tensor 名称和具体错误原因。

### ~~O2. opset ≥ 13 算子属性兼容~~ ✅

**文件**: `src/application/model/onnx_loader.c`

- **opset 解析**: 新增 `g_model_opset` 全局变量，从 `ModelProto.opset_import`（field 8）解析 default domain 版本号
- **Softmax**: 默认 axis 从硬编码 `1` 改为 opset 感知（`< 13` → 1, `≥ 13` → -1）
- **Reshape**: 移除 `is_initializer` 强制要求，支持 opset 14+ 的非 initializer shape 输入
- **ReduceSum/ReduceMax**: 已有 input[1] fallback，无需修改

---

## 第三步：FP16 推理与更大模型（远期，视需求启动）

### F1. FP16 推理支持

**文件**: 新增 `src/platform/tensor_fp16.c`，修改所有 CUDA kernel

- 在 `tensor_t` 中增加 FP16 数据缓冲区
- 为高频算子（Conv、MatMul、逐元素、LayerNorm、MHA）添加 FP16 kernel
- 使用 `cudaDeviceGetAttribute` 查询 FP16 硬件支持
- CPU 端通过 Cast op 做类型转换

### F2. LLM 推理探索

BERT 管线打通后，可探索 GPT-2/LLaMA 等 decoder-only 架构：
- 新增算子: **CausalMask**（下三角 mask）、**KV-Cache**（自回归缓存）、**RoPE**（旋转位置编码）
- 新增融合: **FlashAttention** 风格的 tile-based 精确注意力
- 支持 GQA (Grouped-Query Attention) 和 MQA (Multi-Query Attention) 变体

---

## 持续维护项

| 项 | 说明 |
| --- | --- |
| 文档同步 | 新增算子后同步更新 README 算子表、ARCHITECTURE 目录树、CUDA_GUIDE kernel 清单 |
| compute-sanitizer | 所有新增 CUDA kernel 必须通过 compute-sanitizer 零错误 |
| 测试阈值 | 浮点对比使用相对误差或放宽绝对阈值（1e-3），避免因舍入波动触发假阳性 |
| Commit 规范 | 一个 commit 一个事: feat / fix / refactor 不混合 |
| cuda_device_free 警告 | 排查 tensor_destroy 中的 device memory 释放路径，确认无 double-free |

---

## 进度

| 状态 | 数量 | 内容 |
| --- | --- | --- |
| 已完成 | 2 | O1（外部数据加载错误日志）、O2（opset≥13 属性兼容） |
| 待开始 | 3 | M1-M3（MHA 深度优化） |
| 远期 | 2 | F1（FP16 推理）、F2（LLM 推理探索） |

> **最后更新**: 2026-05-28。O1/O2（ONNX 兼容性修复）已完成，31/31 测试全绿，compute-sanitizer 零错误。进入 MHA 深度优化阶段。
