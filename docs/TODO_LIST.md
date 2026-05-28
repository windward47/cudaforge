# CudaForge Todo List

> 基于 2026-05-28 全状态审阅生成。Phase A/B/C（BERT 全套支持 + MHA 融合优化）已全部完成。旧版历史记录见 [TODO_LIST_20260528.md](TODO_LIST_20260528.md)。

---

## 审阅概要

**当前状态**: 31 种算子全部 CPU+CUDA 双实现，3 个端到端模型验证通过（ResNet-18、YOLOv8n、BERT-base），**31 项测试全通过**。算子覆盖涵盖 CNN 和 Transformer/BERT 推理全链路。Phase C MHA 融合 kernel 已集成，compute-sanitizer 零错误。

---

## 第一步：MHA Kernel 深度优化（3 项，~5 天）

### ~~M1. FlashAttention 风格 tiled 精确注意力~~ ✅

**文件**: `src/operator/nn/mha_fused_cuda.cu`

- **动态共享内存**: `2*S*d` floats (16KB for BERT-base) 替代静态 `64*64*2` (64KB)，SM 占用率从 1 block/SM 提升到 2
- **Tiled K/V 处理**: K/V 按 S 维度分 tile (MHA_MAX_S_SMEM=64)，每 tile 独立加载到 shared memory
- **Online softmax**: tile 间通过 rescale 累积器保持数值一致性，支持任意序列长度
- **性能**: BERT-base CUDA 30.35 → 22.17 ms/iter (**-27%**), speedup vs CPU 1.47x → 2.27x

### ~~M2. QKV 投影融合~~ ✅

**文件**: `src/operator/nn/mha_fused_cuda.cu`

- **K/V 内循环合并**: K 和 V 的投影计算共享 `xv` 加载 — 一次 global memory 读取同时计算 K 和 V，减少 33% 的 X 读取
- **Q 投影**: 保持 per-head 计算（cooperative 全 D 计算需要 D=768 共享内存，超出 48KB 限制）
- **实现方式**: Option B（kernel 侧优化），无需修改 fusion pass 或权重布局
- **共享内存**: 48KB（K_smem + V_smem），在 sm_86 默认限制内

### M3. Tensor Core FP16 MHA（部分完成）

**文件**: `src/operator/nn/mha_fused_f16_cuda.cu`（独立文件，避免 shared memory 类型冲突）

- **混合 FP16 kernel**: 标量 FP16 QKV 投影 + WMMA 16×16×16 输出投影
- **Shared memory**: K(S×d) + V(S×d) + merged(S_pad×d) + W_tile(WMMA_K×WMMA_N) + tbuf(16×16) = ~8KB
- **精度**: max_rel=2.87e-04（远低于 1e-2 阈值）
- **性能**: 29.95 ms/iter（FP32 25.88 ms，慢 16% — QKV 标量转换开销主导）
- **瓶颈分析**: QKV 投影占总计算 95%，标量 `__float2half`/`__hmul`/`__half2float` 转换链开销抵消了 WMMA 输出投影的收益
- **待优化**: WMMA 用于 QKV 投影（需将 X 和权重预加载到 shared memory，避免大局部数组）
- **算子注册**: `mha_fused_f16_cuda`，CPU stub 返回 -1
- **compute-sanitizer**: 0 错误

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
| 已完成 | 5 | O1（外部数据加载错误日志）、O2（opset≥13 属性兼容）、M1（FlashAttention 风格 tiled 注意力）、M2（QKV 投影融合）、M3（FP16 标量 kernel 框架） |
| 待优化 | 1 | M3 WMMA Tensor Core 版本（需解决 store_matrix_sync 类型匹配） |
| 远期 | 2 | F1（FP16 推理）、F2（LLM 推理探索） |

> **最后更新**: 2026-05-29。M1-M3 已完成（M3 为 FP16 标量 kernel，WMMA 版本待优化），31/31 测试全绿，compute-sanitizer 零错误。
