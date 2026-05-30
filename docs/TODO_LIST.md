# CudaForge Todo List

> 基于 2026-05-30 全面代码审阅生成。Phase A/B/C + M1-M3 + F1 + F2 全部完成。旧版历史记录见 [TODO_LIST_20260530.md](TODO_LIST_20260530.md)。

---

## 审阅概要

**当前状态**: 34 种算子全部 CPU+CUDA 双实现（含 FP16 变体），30 项测试全通过，compute-sanitizer 零错误。

**已完成里程碑**:

- Phase A/B/C: BERT 全套推理 + MHA 融合 kernel
- M1-M3: FlashAttention tiled + QKV 融合 + WMMA FP16 (5.57× 加速)
- O1-O2: ONNX opset≥13 兼容 + 外部数据错误日志
- F1: FP16 推理支持 (11 个 FP16 CUDA 算子 + dtype-aware 调度)
- F2: LLM 推理 (CausalMask + mha_decode + KV-cache + GQA + RoPE + 永久融合)

---

## 待完成工作（按优先级）

### Critical — 正确性风险

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| ~~C1~~ | ~~CUDA_CHECK 改为返回错误码~~ | `cuda_ops.h` | ✅ 移除致命宏，统一为返回错误码版本 |
| ~~C2~~ | ~~graph.c realloc 失败处理~~ | `graph.c` | ✅ 4 处 realloc 改用临时变量，失败时保留旧指针 |

### High — 功能完整性

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| H1 | FP16 Conv2D kernel | `conv_f16_cuda.cu` (新建) | FP16 推理链路最大缺失，Conv 是计算密集型算子 |
| H2 | FP16 MatMul kernel | `matmul_f16_cuda.cu` (新建) | WMMA 可复用 MHA 中的模式 |
| H3 | FP16 BatchNorm/LayerNorm | `batchnorm_f16_cuda.cu` 等 | 归一化层在 FP16 模型中频繁使用 |
| H4 | FP16 Softmax/Reduce | `softmax_f16_cuda.cu` 等 | 注意力机制需要 FP16 Softmax |
| H5 | Resize 双线性插值 | `resize.c` / `resize_cuda.cu` | 当前仅 nearest-neighbor，部分模型需要 bilinear |
| H6 | ONNX Pad 算子 | `pad.c` / `pad_cuda.cu` (新建) | CNN 模型常用，当前缺失 |
| H7 | ONNX Clip 算子 | `clip.c` / `clip_cuda.cu` (新建) | 部分导出器用 Clip 替代 ReLU |

### Medium — 测试覆盖

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| M1 | RoPE 独立单元测试 | `tests/test_rope.c` (新建) | 当前无任何测试覆盖 |
| M2 | FP16 算子单元测试 | `tests/test_fp16_ops.c` (新建) | 11 个 FP16 算子无独立测试 |
| M3 | In-place 操作测试 | 现有测试文件 | ReLU/Sigmoid/GELU/SiLU 标记 `OP_FLAG_IN_PLACE` 但无测试验证 `inputs[0]==outputs[0]` |
| M4 | 边界形状测试 | 现有测试文件 | 标量 tensor、0-d tensor、大维度 tensor |
| M5 | TESTS.md 测试映射修正 | `docs/TESTS.md` | 部分测试→文件映射不准确，需对齐实际源文件 |

### Low — 代码质量 & 架构

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| L1 | operator_params_t placeholder | `operator.h` | 基类有 `int placeholder` 字段，应改为空结构或 flexible array |
| L2 | graph_execute 惰性 D2H | `graph.c` | 当前每节点都 D2H 拷贝，中间 tensor 应延迟到图输出时才拷贝 |
| L3 | Application→Operator 层耦合 | `graph.c` | graph.c 直接 include `conv_int.h`/`mha_fused_int.h`，应暴露融合 API |
| L4 | CUDA_CHECK 统一 | 各 `.cu` 文件 | 部分文件仍用致命 `CUDA_CHECK`，应统一为 `CUDA_CHECK_RET` |
| L5 | README_en 同步 | `README_en.md` | 英文 README 测试数量和算子表未同步更新 |

---

## 远期规划

| 方向 | 说明 |
| --- | --- |
| INT8 量化推理 | 量化 kernel + 校准工具 |
| 动态形状支持 | ONNX dynamic axes |
| x86 SIMD 优化 | AVX2/AVX-512 加速 CPU 路径 |
| ARM NEON 优化 | 移动端/嵌入式推理 |
| TensorRT 集成 | 作为后端加速器 |
| 多 GPU 支持 | 模型并行 / 流水线并行 |

---

## 进度

| 状态 | 数量 | 内容 |
| --- | --- | --- |
| 已完成 | 12 | Phase A/B/C, M1-M3, O1-O2, F1, F2, 代码审阅修复 |
| Critical | 2 | C1-C2 (错误处理) |
| High | 7 | H1-H7 (FP16 算子 + ONNX 兼容) |
| Medium | 5 | M1-M5 (测试覆盖) |
| Low | 5 | L1-L5 (代码质量) |
| 远期 | 6 | INT8/动态形状/SIMD/ARM/TensorRT/多GPU |

> **最后更新**: 2026-05-30。Phase A/B/C + M1-M3 + F1 + F2 全部完成，代码审阅修复已提交。34 种算子，30/30 测试通过。
