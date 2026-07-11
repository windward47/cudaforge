# CudaForge Todo List

> 旧版历史记录见 [TODO_LIST_20260623.md](TODO_LIST_20260623.md) 及更早版本。

---

## 当前状态

**v1.2.0** — 36 种算子 CPU+CUDA 双实现，INT8 量化，Flash Attention v2，CUDA Graph, RoPE 扩展(NeoX+inv_freq+FP16+decode融合), native 生成循环(mha_decode+KV-cache)

| 指标 | 数值 |
| --- | --- |
| 算子总数 | 36 |
| FP16 kernel | 16 |
| 测试通过 | 38/38 |
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

## R7: 端到端推理性能分析 ⭐⭐⭐

**动机**：R1-R6 的优化集中在 MHA 单算子（11-14× 加速），但端到端推理中 MHA 只占一部分。需用 nsys 分析完整推理图，定位**新的瓶颈算子**（LayerNorm/MatMul/Softmax/Conv 等），让后续优化投入产出比最高。

| # | 任务 | 说明 |
| --- | --- | --- |
| R7-a | nsys 分析 GPT-2 端到端 | ✅ 发现 98% 时间是 per-node eager D2H（4346 次） |
| R7-b | nsys 验证优化效果 | ✅ D2H 降到 159 次，kernel 时间成主瓶颈 |
| R7-c | 惰性 D2H | ✅ GPT-2 prefill 4.6ms→0.66ms（7×），36/36 通过 |
| R7-d | 端到端基线 + 新瓶颈 | ✅ 见下表 |

### GPT-2 端到端性能基线（R7 后）

| 指标 | 优化前 | 优化后 |
| --- | --- | --- |
| GPT-2 prefill | 4.635 ms/iter | **0.660 ms/iter** (7×) |
| D2H 次数 | 4346 | 159 |
| D2H 时间 | 5.7 ms (98%) | 0.2 ms (25%) |

### 新瓶颈（kernel 时间分布，R7 后）

| Kernel | calls | total_ms | 占比 |
| --- | --- | --- | --- |
| matmul_f32_tiled | 689 | 3.29 | 32% |
| transpose_f32 | 424 | 1.52 | 15% |
| add (broadcast+no-broadcast) | 1113 | 1.73 | 17% |
| mul (broadcast+no-broadcast) | 742 | 1.14 | 11% |
| layernorm | 265 | 0.68 | 7% |
| reshape_copy | 424 | 0.61 | 6% |

> **下一步优化方向**：matmul（32%）已是 tiled kernel，可考虑 WMMA FP16；transpose（15%）异常高，可能是 GPT-2 的 attention reshape；elementwise add/mul（28%）可融合。
>
> 进度：R7 全部完成。惰性 D2H 是端到端最大收益项（7×）。

---

## R8: 分体式 Flash Attention (分离输出投影) ⭐

**来源**：GLM5.1 代码审阅指出 atomicAdd 是性能杀手，建议分离输出投影。

**评估结果**：已实施并 benchmark，**回退**。

| 路径 | 一体式 (R7) | 分体式 (R8) | 结论 |
| --- | --- | --- | --- |
| S≤64 短序列 | preloaded FP32 2.4ms | flash+proj 3 kernel | ❌ 分体开销大 |
| S=512 长序列 | 33ms | 45ms | ❌ output_proj 标量循环慢 |
| GPT-2 端到端 | 0.66ms | 2.34ms | ❌ 额外 malloc+launch |

**根因**：
- atomicAdd 在输出投影段（不在 KV 循环内），实际影响有限
- 分体式引入 O_attn buffer 的 cudaMalloc/free + 额外 kernel launch
- output_proj kernel 标量循环 (H_q×d×D) 比 WMMA 慢
- 短序列的 3 阶段 (precompute+flash+proj) 开销 > 一体式

**结论**：GLM5.1 的 atomicAdd 指出在理论上有价值，但实际 benchmark 显示一体式更快。分体式需配合 output_proj 的 WMMA 实现 + buffer 复用才有收益，当前不划算。

> 进度：R8 已验证回退。一体式 (R7) 是当前最优。

---

## R9: RoPE 位置编码扩展 ⭐⭐⭐

**来源**：位置编码方案调研（对比 ggml / flash-attention 参考实现）。

**背景**：RoPE 当前是「已实现但未接入推理链路的孤立算子」--GPT-2 测试模型不用位置编码，`onnx_loader.c` 不识别 RotaryEmbedding，`generate.c` 不调用 RoPE。当前实现仅 interleaved 布局、B=1、无 pos_offset、无缩放、CUDA 每线程重算 sin/cos。要支持 LLaMA/Mistral，需补齐 NeoX 布局、batch、pos_offset、FP16、频率缓存、上下文扩展。

**调研结论**（关键架构决策）：
- **inv_freq 表驱动**：所有上下文扩展变体（Linear/NTK/Dynamic NTK/YaRN/LongRoPE）最终都是改变 `angle = pos · θ_i` 中的频率 `θ_i` 或位置 `pos`。因此 RoPE kernel 应直接接收 `inv_freq[d/2]` device 表（而非 `base` 标量），一次改动覆盖全部变体，且查表比重算 `powf/cosf/sinf` 快得多。参考 flash-attention 解耦哲学。
- **NeoX half-split 布局**：LLaMA/Mistral 训练即此布局（前半段 `i` 与后半段 `i+d/2` 配对），推理必须匹配。当前 interleaved（相邻对 `2i`/`2i+1`）保留给 GPT-J 系。用布局枚举字段统一两种，kernel 内一个分支。参考 ggml `rope.cu:177-181`。
- **learned PE 不动**：GPT-2/BERT 的 learned absolute PE 已烘焙在 ONNX 图（Gather+Add），走通用路径无需新增算子。

**两阶段**：阶段一（R9-a~e，算子增强）→ 阶段二（R9-f~h，端到端接入）。

| # | 任务 | 文件 | 优先级 | 状态 | 说明 |
| --- | --- | --- | --- | --- | --- |
| R9-a | 参数结构重构 + inv_freq 表化 + NeoX 布局 | `rope_int.h`/`rope.c`/`rope_cuda.cu`/`test_rope.c` | ⭐⭐⭐ | ✅ | `rope_params_t` 加 `layout`(interleaved/half-split)+`inv_freq`表指针；kernel 改查表 + 布局分支；向后兼容 base 标量路径 |
| R9-b | batch(B>1) + pos_offset | `rope.c`/`rope_cuda.cu`/`test_rope.c` | ⭐⭐⭐ | ✅ | kernel 加 B 维；pos_offset 适配 KV-cache 续写(pos=offset+idx)；新增 batch/offset 测试 |
| R9-c | FP16 RoPE | 新建 `rope_f16_cuda.cu`/`fp16_cpu_ops.c`/`test_rope.c` | ⭐⭐ | ✅ | 模板 `softmax_f16_cuda.cu`；CPU 委托模式；注册 `rope_f16_cuda`/`rope_f16`；FP16 kernel 数 15->16 |
| R9-d | CUDA sin/cos 预计算表优化 | `rope_cuda.cu`/`rope_f16_cuda.cu` | ⭐⭐ | ✅ | shared memory 预计算 inv_freq，消除每线程 powf；精度不变 |
| R9-e | theta 缩放（Linear + NTK-aware） | `test_rope.c` | ⭐ | ✅ | inv_freq 表驱动验证：Linear(inv_freq/scale) + NTK(base'=base·scale^(d/(d-2)))；高频保留低频拉伸 |
| R9-f | RoPE 接入推理图引擎 | 新建 `test_rope_graph.c` | ⭐⭐ | ✅ | native 图 [INPUT->OP_ROPE->OUTPUT] 经 graph_execute 验证 dispatch 链路；CPU+CUDA 双跑。ONNX 映射降级(标准 ONNX 无此算子) |
| R9-g | GPT-2 测试模型加 RoPE | - | ⭐⭐⭐ | ❌ 已评估 | ONNX 路径不可行：PyTorch 无内置 RoPE，自定义 op 导出后 ORT 无法生成参考输出。R9-f native 图已覆盖"推理链路接入"验证 |
| R9-h | mha_decode 内部融合 RoPE | `mha_decode_int.h`/`mha_decode.c`/`mha_decode_cuda.cu`/`test_mha_decode.c` | ⭐ | ✅ | 方案 A：参数加 rope_base/layout/inv_freq(base=0 禁用)；CPU/CUDA kernel 在 Q/K 投影后、K 写 cache 前旋转(pos=cache_len)；附带修复 2 个预存 CUDA bug(merged 归约+cache copy) |

### R9-h 子任务（方案 A：mha_decode 内部融合 RoPE）

**调研结论**：推荐方案 A（内部融合）而非方案 B（改接口接收已旋转 Q/K）或方案 C（图层面拆分）。理由：(1) `pos_offset` 直接用 `mha_decode_params_t.cache_len`（已被 `graph_update_cache_len` 每步更新），无需改图引擎；(2) decode 单 token 对 launch 延迟敏感，保持单 kernel 最优；(3) 向后兼容（base=0 走原路径）；(4) cache 存旋转后 K，prefill/decode 语义统一。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R9-h1 | 参数结构扩展 | `mha_decode_int.h` | 加 `float rope_base`(0=禁用)、`int32_t rope_layout`、`const float* rope_inv_freq`(NULL 用 base 现算) |
| R9-h2 | CPU 融合 RoPE | `mha_decode.c` | K_new 算完后写 cache 前旋转(pos=cache_len)；Q 算完后 attention 前旋转；复用 rope.c 成对旋转逻辑 |
| R9-h3 | CUDA 融合 RoPE | `mha_decode_cuda.cu` | Q 旋转(Q_reg 单线程,简单)；K 旋转需重构线程循环(每线程处理整 kv_h 的 d 维,d≤64 放寄存器)或两阶段(shared mem 暂存再旋转) |
| R9-h4 | 测试 | `test_mha_decode.c` | 新增 RoPE 测试 case(rope_base=10000, layout=HALFSPLIT)；decode_ref 加旋转；验证 cache 存旋转后 K |

**执行约定**：逐个任务实现+编译+test_mha_decode+全量 ctest 回归+commit（遵循"一个 commit 一件事"）。commit message 用 `(R9-hN)` 后缀。

**执行约定**：逐个任务实现+编译+test_rope+全量 ctest 回归+commit（遵循"一个 commit 一件事"）。commit message 用 `(R9-x)` 后缀。

**完成情况**：R9-a~f 全部完成（37/37 测试通过），R9-g 已评估不可行（native 图已覆盖），R9-h 完成（mha_decode 内部融合 RoPE + 修复 2 个预存 CUDA bug）。RoPE 全链路就绪：算子层（NeoX/interleaved 双布局、inv_freq 表驱动、B>1、pos_offset、FP16、Linear/NTK 缩放、shared memory 优化）+ 推理图接入 + decode 融合（KV-cache 存旋转后 K，pos=cache_len）。

---

## R10: 生成循环接入 mha_decode + KV-cache ⭐⭐⭐

**来源**：R9-h 完成后，mha_decode 已支持 RoPE 融合，但 `generate.c` 仍对每个 token 重跑整图（行 9-11 注释承认是 future work）。需把 mha_decode+KV-cache 接入生成循环，实现真正的逐 token 自回归生成。

**调研结论**（方案选择）：
- **方案 A（改 generate.c + ONNX KV-cache 路径）不推荐**：现有 gpt2_full.onnx 无 KV-cache 端口、用绝对位置 embedding（非 RoPE），需改导出脚本+loader+融合检测器，工作量大且脆弱。
- **方案 C（generate.c 加 KV-cache 分支但图不变）不可行**：现有图是 OP_MHA_FUSED（无 cache 端口）。
- **方案 B（native 生成测试）推荐**：C 代码直接构建 `[embed -> mha_decode(RoPE) -> FFN -> lm_head]` native 图，跑 prefill->decode 生成循环。所有底层能力已就绪（mha_decode RoPE 融合、graph KV-cache API、native 图构建范例 test_rope_graph.c/test_mha_decode.c）。

**关键设计**：
- prefill 和 decode 统一用 mha_decode 逐 token 填 cache（cache_len 从 0 递增），避免 prefill/decode 切换复杂性。
- RoPE 已融合进 mha_decode（R9-h），图里**不另加** OP_ROPE 节点（避免双重旋转），`rope_base=10000` 启用。
- 每步调用 `graph_update_cache_len(g, cache_len)` 递增所有 MHA_DECODE 节点位置。
- `graph_set_kv_cache` 标记 K/V cache tensor 持久化，`graph_set_permanent_fusion` 避免重复融合。

| # | 任务 | 文件 | 优先级 | 说明 |
| --- | --- | --- | --- | --- |
| R10-a | native 单层 transformer 图 | 新建 `tests/test_native_generate.c` | ⭐⭐⭐ | ✅ 构建 [embed->mha_decode(RoPE)+residual->FFN+residual->lm_head] 图；随机权重；参考 test_rope_graph.c + test_mha_decode.c 范例 |
| R10-b | prefill->decode 生成循环 | `tests/test_native_generate.c` | ⭐⭐⭐ | ✅ cache_len 递增 + graph_update_cache_len + graph_set_kv_cache + graph_set_permanent_fusion；prefill 2 token，decode 贪心生成 2 token |
| R10-c | CPU 验证 | `tests/test_native_generate.c` | ⭐⭐⭐ | ✅ 4 步(logits max_diff<3e-7)；KV-cache 跨步持久；cache_len 正确递增；CMake 注册；38/38 测试通过 |

**执行约定**：逐个任务实现+编译+全量 ctest 回归+commit。commit message 用 `(R10-x)` 后缀。

**完成情况**：R10 全部完成（38/38 测试通过）。native 单层 transformer 图（embed+mha_decode[RoPE]+FFN+lm_head）经 graph_execute 跑通 prefill->decode 生成循环，验证 mha_decode RoPE 融合 + KV-cache 持久化 + cache_len 递增在端到端生成中正确工作。CUDA 验证完成（generate_tokens CPU/CUDA 生成 token 序列完全一致）。

---

## R11: 端到端性能优化 ⭐⭐⭐

**来源**：R7 瓶颈分析（matmul 32%, transpose 15%, add/mul 28%, layernorm 7%）。调研发现根因与 R7 推测不同：transpose 主因是 Gemm transB 权重转置（非 attention reshape，已被 MHA fusion 消除）；add/mul 主因是 GELU tanh 展开 + causal mask Where + residual，且有 add_cuda.cu 标量 D2H sync bug 放大延迟。

**调研结论**（关键发现）：
- **add_cuda.cu:38-40 标量 D2H sync bug**：`B_numel==1` 时 `memcpy_d2h` + `stream_synchronize` 强制 host 同步，1113 次 add 调用里引入显著延迟。R7 的惰性 D2H 未覆盖此路径。P0 修复。
- **transpose 主因是 Gemm transB**：matmul_cuda.cu:385 对 transB 做物理转置再 GEMM，~250 次/50iter。非 attention reshape（已被 MHA fusion 消除）。
- **独立 matmul WMMA 对 GPT-2 无收益**：小矩阵（8×64）远低于 TC 门槛，FP16 转换不划算（R6-c 已验证回退）。
- **layernorm_residual kernel 是死代码**：layernorm_cuda.cu:60 已实现但 dispatch 未接线。
- **MatMul->Activation fusion 检测了但没实现**：graph.c:802 的 OP_MATMUL 分支缺失。

| # | 任务 | 文件 | 优先级 | 状态 | 说明 |
| --- | --- | --- | --- | --- | --- |
| R11-a | 修复 add 标量 D2H sync | `add_cuda.cu` | ⭐⭐⭐ P0 | ✅ | 消除 B_numel==1 时的 memcpy_d2h+stream_synchronize 隐式同步；新增 scalar_broadcast kernel |
| R11-b | 重跑 bench_e2e nsys | `scripts/run_profiling.sh` | ⭐⭐⭐ P0 | ✅ | 确认瓶颈分布：matmul 34%, transpose 16%, add/mul 25%, layernorm 7%, reshape 6% |
| R11-c | 消除 Gemm transB 物理 transpose | `matmul_cuda.cu` | ⭐⭐⭐ P0 | ✅ | tiled kernel 加 transpose_b 参数直接读转置布局；消除物理 transpose kernel launch+alloc。naive 也统一 API(仍用物理转置,cache 友好) |
| R11-d | layernorm_residual / matmul+act fusion | - | ⭐⭐ P1 | ⬜ 待后续 | 死代码接线 + GELU 融合；需更多架构改动 |

**完成情况**：R11-a~c 完成（38/38 测试通过）。add D2H sync bug 修复 + tiled transB in-place 读取消除物理 transpose。nsys 确认瓶颈分布与 R7 一致。bench_e2e 基线 0.66ms(R7) -> ~0.7ms(R11, 含 R9/R10 叠加)，benchmark 噪声较大(GPU 频率波动 0.58-1.1ms)。R11-d(layernorm+residual/matmul+act fusion)留作后续。

**执行约定**：逐个任务实现+编译+全量 ctest 回归+commit。commit message 用 `(R11-x)` 后缀。

---

## 进度

| 状态 | 内容 |
| --- | --- |
| 已完成 | R1 全部, R2 全部, R3-a, R4 全部, R5 全部, Flash Attention v2 (FP32 + FP16 WMMA, smem 精简 2 blocks/SM), R9 全部 (RoPE 扩展), R10 全部 (native 生成循环), R11-a~c (add D2H bug修复 + transB in-place消除 + nsys确认瓶颈) |
| 进行中 | - |
| 已验证失败 | R6-a (寄存器压力), R6-c (WO FP16 转换开销), R8 (分体式开销大, 回退), R9-g (ONNX RoPE 导出不可行, native 图已覆盖) |
| 已完成 | R6-b (S≤64 FP16 WMMA, 13.8×) |
| 待评估 | R6-d (cp.async, 优先级低), R11-d (layernorm+residual/matmul+act fusion) |

> **最后更新**: 2026-07-12。R9 RoPE + R10 生成循环 + R11 端到端性能优化（R11-a~c）完成（38/38 测试通过）。R11 修复 add D2H sync bug + 消除 tiled transB 物理 transpose，nsys 确认瓶颈分布（matmul 34%/transpose 16%/add+mul 25%）。R11-d（layernorm+residual fusion / matmul+act fusion）留作后续。
