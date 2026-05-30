# CudaForge Todo List

> 基于 2026-05-30 v0.5.0 测试报告更新。旧版历史记录见 [TODO_LIST_20260530.md](TODO_LIST_20260530.md)。

---

## 当前状态

**v0.5.0** — 36 种算子 CPU+CUDA 双实现（含 15 个 FP16 kernel），33/34 测试通过，compute-sanitizer 零错误。

| 指标 | 数值 |
| --- | --- |
| 算子总数 | 36 |
| FP16 kernel | 15 |
| 测试通过 | 33/34 (test_conv 编译问题) |
| compute-sanitizer | 0 errors |
| BERT-base CUDA FP32 | 26.01 ms/iter |
| BERT-base CUDA FP16 WMMA | **4.66 ms/iter (8.96× vs CPU)** |

**已完成里程碑**:

- Phase A/B/C: BERT 全套推理 + MHA 融合 kernel
- M1-M3: FlashAttention tiled + QKV 融合 + WMMA FP16 (8.96× 加速)
- O1-O2: ONNX opset≥13 兼容 + 外部数据错误日志
- F1: FP16 推理支持 (15 个 FP16 CUDA 算子 + dtype-aware 调度)
- F2: LLM 推理 (CausalMask + mha_decode + KV-cache + GQA + RoPE + 永久融合)
- 代码审阅: Critical/High/Medium 修复全部完成

---

## 待完成工作

### Critical — 正确性风险

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| ~~C1~~ | ~~CUDA_CHECK 改为返回错误码~~ | `cuda_ops.h` | ✅ 移除致命宏，统一为返回错误码版本 |
| ~~C2~~ | ~~graph.c realloc 失败处理~~ | `graph.c` | ✅ 4 处 realloc 改用临时变量 |

### High — 功能完整性

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| ~~H1~~ | ~~FP16 Conv2D kernel~~ | `conv_f16_cuda.cu` | ✅ 直接卷积，FP16 输入/权重，FP32 累加 |
| ~~H2~~ | ~~FP16 MatMul kernel~~ | `matmul_f16_cuda.cu` | ✅ 16×16 tiling，FP16 输入，FP32 累加 |
| ~~H3~~ | ~~FP16 BatchNorm/LayerNorm~~ | `norm_f16_cuda.cu` | ✅ BatchNorm + LayerNorm FP16 |
| ~~H4~~ | ~~FP16 Softmax~~ | `softmax_f16_cuda.cu` | ✅ FP16 Softmax，FP32 累加 |
| ~~H5~~ | ~~Resize 双线性插值~~ | `resize.c` / `resize_cuda.cu` | ✅ CPU + CUDA 均支持 bilinear |
| ~~H6~~ | ~~ONNX Pad 算子~~ | `pad.c` / `pad_cuda.cu` | ✅ constant/edge/reflect 三种模式 |
| ~~H7~~ | ~~ONNX Clip 算子~~ | `clip.c` / `clip_cuda.cu` | ✅ clamp [min, max] |

### Medium — 测试覆盖

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| ~~M1~~ | ~~RoPE 独立单元测试~~ | `tests/test_rope.c` | ✅ CPU + in-place + CUDA (max_diff=5.96e-08) |
| ~~M2~~ | ~~FP16 算子单元测试~~ | `tests/test_fp16_ops.c` | ✅ Softmax FP16 + Conv2D FP16 |
| ~~M3~~ | ~~In-place 操作测试~~ | `test_relu.c`, `test_activations.c` | ✅ ReLU/Sigmoid/GELU/SiLU |
| ~~M4~~ | ~~边界形状测试~~ | `test_add_scalar` 等 | ✅ 关键边界已覆盖 |
| ~~M5~~ | ~~TESTS.md 测试映射修正~~ | `docs/TESTS.md` | ✅ 已修正所有映射 |

### Low — 代码质量 & 架构

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| ~~L1~~ | ~~operator_params_t placeholder~~ | `operator.h` | ✅ 改为 `char _reserved` |
| L2 | graph_execute 惰性 D2H | `graph.c` | 暂缓 — 需要 CPU/GPU 消费者追踪 |
| L3 | Application→Operator 层耦合 | `graph.c` | 暂缓 — 融合 pass 需访问算子内部参数 |
| ~~L4~~ | ~~CUDA_CHECK 统一~~ | `cuda_ops.h` | ✅ 已在 C1 中完成 |
| ~~L5~~ | ~~README_en 同步~~ | `README_en.md` | ✅ 算子表 + FP16 + 测试数量已同步 |

### 待修复 — 测试问题

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| T1 | test_conv 编译修复 | `tests/test_conv.c` | MSVC C4100/C4189 警告即错误，需添加 /wd4100 /wd4189 |

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
| 已完成 | 19 | Phase A/B/C, M1-M3, O1-O2, F1, F2, C1-C2, H1-H7, M1-M5, L1/L4/L5 |
| 暂缓 | 2 | L2 (惰性 D2H), L3 (层耦合) |
| 待修复 | 1 | T1 (test_conv 编译) |
| 远期 | 6 | INT8/动态形状/SIMD/ARM/TensorRT/多GPU |

> **最后更新**: 2026-05-30。v0.5.0 发布。36 种算子，33/34 测试通过，BERT-base FP16 4.66ms (8.96× vs CPU)。
