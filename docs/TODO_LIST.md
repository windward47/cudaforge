# CudaForge Todo List

> 旧版历史记录见 [TODO_LIST_20260623.md](TODO_LIST_20260623.md) 及更早版本。

---

## 当前状态

**v1.0.0** — 36 种算子 CPU+CUDA 双实现，INT8 量化，Flash Attention v2，CUDA Graph。

| 指标 | 数值 |
| --- | --- |
| 算子总数 | 36 |
| FP16 kernel | 15 |
| 测试通过 | 36/36 |
| compute-sanitizer | 0 errors |
| BERT-base CUDA FP16 WMMA | **4.66 ms/iter** |
| Flash Attention S=128 | **64.1 ms** |
| Flash Attention S=512 | **309.6 ms** |
| INT8 MatMul 精度 | max_rel=6.97e-04 |

---

## R4: Flash Attention 多行 Q Tiling + Tensor Core ⭐⭐⭐

**来源**：Flash Attention 2 论文 Section 3.3 + `cudaTensorCoreGemm.cu` 参考实现。

**问题**：当前 flash attention kernel 每个 block 只处理 1 行 Q（`grid(B*S, 1, H_q)`），attention 是向量×矩阵（1×d × d×d），WMMA 16×16 无法发挥 Tensor Core 优势。FA2 论文指出 non-matmul FLOP 比 matmul 贵 16×。

**目标**：重构为多行 Q tiling（BM=64），4 warps 各处理 16 行 Q，启用 WMMA Tensor Core。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R4-a | 多行 Q tiling kernel | `mha_fused_cuda.cu` | grid 改为 `(ceil(S/BM), B, H_q)`，每个 block 处理 BM=64 行 Q |
| R4-b | Warp 分配 | `mha_fused_cuda.cu` | 4 warps 各处理 16 行 Q，K/V 共享，消除 warp 间通信 |
| R4-c | WMMA Q·Kᵀ | `mha_fused_cuda.cu` | scores = Q_half · K_half^T，FP16 in → FP32 out |
| R4-d | WMMA P·V | `mha_fused_cuda.cu` | out = P_half · V_half，FP16 in → FP32 out |
| R4-e | 输出投影 WMMA | `mha_fused_cuda.cu` | Y = out · WO 用 WMMA 加速 |
| R4-f | 测试 + benchmark | `test_bert_mha.c`, `bench_bert_mha.c` | 正确性 + 性能对比 |

### 参考文件

- `flash-attention-main/csrc/flash_attn/src/flash_fwd_kernel.h` — FA2 warp 分 Q 实现
- `cuda-samples-12.8/.../cudaTensorCoreGemm.cu` — WMMA GEMM 模式
- `src/operator/nn/mha_fused_f16_cuda.cu` — 现有 WMMA 投影代码

### 设计要点

```text
Grid:  (ceil(S/64), B, H_q)
Block: (128 threads = 4 warps)

每个 warp: 处理 Q 的 16 行 (64/4=16)
  Warp 0: Q rows 0-15
  Warp 1: Q rows 16-31
  Warp 2: Q rows 32-47
  Warp 3: Q rows 48-63

K/V: 所有 warp 共享，从 smem 加载
scores: 每个 warp 独立计算 16×BN，无需跨 warp 通信
out:    每个 warp 独立计算 16×d，最后 atomicAdd 到 Y
```

---

## R5: 代码清理与文档 ⭐

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R5-a | 清理 FP16 kernel 中的禁用代码 | `mha_fused_cuda.cu` | 移除 `mha_flash_attn_v2_f16_kernel` 或标记为 experimental |
| R5-b | 更新 ARCHITECTURE.md | `docs/ARCHITECTURE.md` | 新增 Flash Attention 架构说明 |
| R5-c | 更新 CUDA_GUIDE.md | `docs/CUDA_GUIDE.md` | 新增 WMMA/FA2 优化指南 |

---

## 进度

| 状态 | 内容 |
| --- | --- |
| 已完成 | R1 全部, R2 全部, R3-a, Flash Attention v2 (warp 归约 + exp2f + K/V 预计算 + Split-KV + grid H_q) |
| 进行中 | R4 多行 Q Tiling + Tensor Core |
| 计划中 | R5 代码清理 |

> **最后更新**: 2026-06-23。
