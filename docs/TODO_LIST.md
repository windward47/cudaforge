# CudaForge Todo List

> 基于 2026-05-28 全状态审阅生成。Phase A/B/C（BERT 全套支持 + MHA 融合优化）已全部完成。旧版历史记录见 [TODO_LIST_20260528.md](TODO_LIST_20260528.md)。

---

## 审阅概要

**当前状态**: 34 种算子全部 CPU+CUDA 双实现（含 FP16 变体），4 个端到端模型验证通过（ResNet-18、YOLOv8n、BERT-base、GPT-2 测试模型），**30 项测试全通过**。算子覆盖涵盖 CNN、Transformer/BERT、LLM 推理全链路。MHA 融合 kernel (FP32+FP16 WMMA)、decode kernel (KV-cache+GQA)、RoPE、CausalMask 均已实现。

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

### ~~M3. Tensor Core FP16 MHA~~ ✅

**文件**: `src/operator/nn/mha_fused_f16_cuda.cu`（独立文件，避免 shared memory 类型冲突）

- **全 WMMA kernel**: X 一次性加载为 FP16 到 shared memory，QKV 投影和输出投影全部使用 WMMA 16×16×16 Tensor Cores
- **WO 分 tile 加载**: 每次加载 WMMA_K×WMMA_N = 256 个 FP16 元素到 W_tile，避免 WO 全量加载超出 shared memory 限制
- **Shared memory**: X_h(24KB) + K(1KB) + V(1KB) + M(2KB) + W_tile(1KB) + tbuf(1KB) = ~30KB
- **精度**: max_rel=1.27e-06（与 FP32 一致）
- **性能**: **4.66 ms/iter**（FP32 25.97 ms，**5.57× 加速**）
- **算子注册**: `mha_fused_f16_cuda`，CPU stub 返回 -1
- **compute-sanitizer**: 0 错误，31/31 测试全通过

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

### ~~F1. FP16 推理支持~~ ✅

**文件**: 多文件修改

- **Cast FP16<->FP32**: `cast_int.h` 新增 `ONNX_DTYPE_FLOAT16=10`，CPU 和 CUDA 实现 FP16<->FP32 转换（CPU 使用软件 IEEE 754 编解码，CUDA 使用 `__float2half`/`__half2float`）
- **Dtype-aware 调度**: `graph.c` 的 `graph_execute()` 检查输出 tensor dtype，FP16 时先查找 `_f16` 变体，找不到回退 `_f32`
- **FP16 逐元素算子**: `elementwise_f16_cuda.cu` — ReLU、Sigmoid、GELU、SiLU、Exp、Add、Mul、Sub、Div（支持 broadcast）
- **FP16 MHA**: 已有 `mha_fused_f16_cuda`（WMMA Tensor Core，5.57× 加速）
- **tensor 系统**: 已支持 FP16（`DATA_TYPE_F16` 已定义，`tensor_create`/`copy_to_device`/`copy_to_host` 按 dtype size 计算字节）
- **注册算子**: 11 个 FP16 CUDA 算子（relu/sigmoid/gelu/silu/exp/add/mul/sub/div/mha_fused/cast）
- **31/31 测试通过**

### F2. LLM 推理探索（部分完成）

已实现 decoder-only 模型核心组件：

- **CausalMask 算子**: `OP_CAUSAL_MASK`，下三角 attention mask（CPU + CUDA）
- **mha_decode kernel**: `OP_MHA_DECODE`，单 token 解码 + KV-cache（CPU + CUDA）
  - KV-cache: 预分配 `(B, max_seq, H, d)` 缓冲区，`cache_len` 追踪填充位置
  - 注意力: 遍历 cache 中 0..cache_len 位置，online softmax
  - 输出投影: merged 通过 shared memory 广播到所有线程
- **Graph KV-cache 支持**: `graph_set_kv_cache()` 标记持久 tensor，`graph_execute()` 跳过 D2H 拷贝
- **算子注册**: 扩展到 128 slots（原 64 不够用）
- **测试**: `test_mha_decode`（CPU + CUDA 通过），`bench_mha_decode`（延迟测量）
- **GPT-2 测试模型**: `gen_gpt2_test.py`（hidden=64, heads=4, 2 层）
- **性能**: 单 token CUDA 解码慢于 CPU（launch 开销主导），需要 batch 解码或 graph 级优化

已追加实现:

- **RoPE 算子**: `OP_ROPE`，旋转位置编码（CPU + CUDA），支持 LLaMA/Mistral
- **永久 MHA 融合**: `graph_set_permanent_fusion()` — 跳过 graph_execute 后的 restore，适合 decode 循环

已追加实现:

- **GQA 支持**: `num_kv_heads` 参数，支持标准 MHA / GQA / MQA
- **mha_decode kernel 优化**: 协作式 K/V 计算 + 移除 atomicAdd → **4x 加速** (90ms → 22.5ms)

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
| 已完成 | 7 | O1-O2（ONNX 兼容性）、M1-M3（MHA 优化 + FP16）、F1（FP16 推理）、F2（LLM 推理） |
| 远期 | 0 | — |

> **最后更新**: 2026-05-30。F2 全部完成：CausalMask + mha_decode(GQA) + KV-cache + RoPE + 永久融合 + kernel 优化。30/30 测试通过，compute-sanitizer 零错误。BERT-base 单层 MHA 4.66ms(FP16 WMMA)，decode 22.5ms。
