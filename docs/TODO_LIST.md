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
| R5-a | FP16 kernel 标记为 experimental | `mha_fused_cuda.cu` | ✅ 已标记，host dispatch 中注释说明 |
| R5-b | 更新 ARCHITECTURE.md | `docs/ARCHITECTURE.md` | ✅ 新增 Flash Attention 架构说明 |
| R5-c | 更新 CUDA_GUIDE.md | `docs/CUDA_GUIDE.md` | ✅ 新增 WMMA 实战注意 + FA2 优化路径 |
| R5-d | Profiling 工作流文档 | `docs/PROFILING_GUIDE.md` | ✅ nsys/ncu/sanitizer 三层流程 + 实战案例 |
| R5-e | Profiling 脚本 | `scripts/run_profiling.sh` | ✅ 一键 sanitizer + nsys + occupancy 估算 |

---

## R6: Profiling 驱动的下一轮优化 ⭐⭐

**来源**：nsys 分析报告（RTX 2050, sm_86, 100KB smem/SM, 16 SMs）

**当前状态**（FP16 WMMA kernel）：
- smem 33.5KB → 2 blocks/SM（已从 1 提升到 2）
- 72 regs/thread（寄存器非瓶颈：65536/(72×64)=14 blocks 可达）
- kernel 时间 S=512: 22ms（nsys 纯 kernel）

**分析报告瓶颈**：
1. FP16 kernel smem 仍是 occupancy 瓶颈（33.5KB 限 2 blocks/SM，理论可 3+）
2. acc_o_fmem 持久化在 smem 占 16KB（FA_F16_BM×FA_MAX_D×4），可用 WMMA fragment 替代
3. preloaded kernel (S≤8 短序列) 占 benchmark 总时间 3367ms，无 FP16 WMMA 加速
4. benchmark 内存传输开销大（D2H 68%），实际推理应避免每 iter copy

| # | 任务 | 文件 | 优先级 | 预期收益 | 说明 |
| --- | --- | --- | --- | --- | --- |
| R6-a | acc_o 用 fragment 替代 smem | `mha_fused_cuda.cu` | — | **失败** | ❌ 已验证：fragment 化导致寄存器 72→121/thread，kernel 变慢 3.4×（无 spill 但指令级并行度下降）。省 16KB smem 换寄存器压力不划算，已回退 |
| R6-b | preloaded kernel FP16 WMMA | `mha_fused_cuda.cu` | — | ✅ **13.8×** | S≤64 路径改为走 FP16 flash WMMA kernel（precompute + flash），S=8 从 31.9ms→2.3ms。preloaded FP32 kernel 保留为 fallback |
| R6-c | 输出投影 WMMA | `mha_fused_cuda.cu` | — | **不可行** | ❌ 已评估：WO 是 FP32，WMMA 需 FP16 转换（48 tile × 256 转换 = 12K 转换），转换开销 + smem 压力抵消 Tensor Core 收益，回退 |
| R6-d | cp.async 双 buffer | `mha_fused_cuda.cu` | — | 待评估 | 需先改 precompute 输出 FP16（依赖链长），double buffer 增加 9KB smem（仍 2 blocks/SM）。当前 K/V 加载非瓶颈（precompute 1ms vs flash 7ms），优先级低 |

> 进度：R6-a 已验证失败回退。R6-b/c/d 需架构级改动（precompute FP16 输出 / WO 分块 / double buffer），当前 smem/寄存器预算紧张，留作后续。
>
> **当前最优配置**（sm_86 RTX 2050）：FP16 WMMA kernel, BM=32, 2 warps, K/V smem 复用, 33.5KB smem, 72 regs, 2 blocks/SM。

---

## 进度

| 状态 | 内容 |
| --- | --- |
| 已完成 | R1 全部, R2 全部, R3-a, R4 全部, R5 全部, Flash Attention v2 (FP32 + FP16 WMMA, smem 精简 2 blocks/SM) |
| 进行中 | — |
| 已验证失败 | R6-a (寄存器压力), R6-c (WO FP16 转换开销) |
| 已完成 | R6-b (S≤64 FP16 WMMA, 13.8×) |
| 待评估 | R6-d (cp.async, 优先级低) |

> **最后更新**: 2026-06-23。Profiling 工作流文档化完成，R6 优化方案基于 nsys 报告制定。
