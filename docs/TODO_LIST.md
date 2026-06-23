# CudaForge Todo List

> 基于 [CUDA-Agent](https://github.com/BytedTsinghua-SIA/CUDA-Agent)（RL 训练 CUDA kernel）、[CUDAForge](https://github.com/)（多 Agent LLM 优化）、[ggml](https://github.com/ggml-org/ggml)（量化推理引擎）三个项目分析整合。旧版历史记录见 [TODO_LIST_20260607b.md](TODO_LIST_20260607b.md)。参考项目见 `ggml-ref/`、`CUDA-Agent-main/`。

---

## 当前状态

**v0.9.0** — 36 种算子 CPU+CUDA 双实现（含 15 个 FP16 kernel），35/35 测试通过，AVX2 + Flash Attention 完成。

| 指标 | 数值 |
| --- | --- |
| 算子总数 | 36 |
| FP16 kernel | 15 |
| 测试通过 | 35/35 |
| compute-sanitizer | 0 errors |
| BERT-base CUDA FP32 | 26.01 ms/iter |
| BERT-base CUDA FP16 WMMA | **4.66 ms/iter (8.96× vs CPU)** |
| GPT-2 logits max_diff | **4.77e-07** (vs ONNX Runtime) |

---

## R1: 架构优化

> 综合 CUDA-Agent 的自注册/验证模式 + CUDAForge 的 NCU 硬件指标驱动优化。

### R1-1: 算子自动注册 ⭐⭐⭐

**来源**：CUDA-Agent `binding_registry.h` 静态自动注册模式。

**目标**：消除 `operator_init.c` 中 76 个前向声明 + 76 个调用的重复代码，新增算子从改 3 个文件降为改 2 个文件。

**当前问题**（`src/operator/operator_init.c`）：

```c
// 76 个前向声明 + 76 个调用，每新增一个算子都要手动添加
int register_relu_f32(void);
int register_matmul_f32(void);
// ... 76 行
int operator_init_all(void) {
    ret += register_relu_f32();
    // ... 76 行
}
```

**方案**：引入 X-macro 模式，所有注册条目集中在 `operator_registry.def`：

```c
// src/operator/operator_registry.def
// REGISTER_CPU(fn) / REGISTER_CUDA(fn)
REGISTER_CPU(register_relu_f32)
REGISTER_CPU(register_matmul_f32)
// ...
REGISTER_CUDA(register_relu_f32_cuda)
// ...

// src/operator/operator_init.c
int operator_init_all(void) {
    int ret = 0;
    #define REGISTER_CPU(fn)  ret += fn();
    #define REGISTER_CUDA(fn) ret += fn();
    #include "operator_registry.def"
    #undef REGISTER_CPU
    #undef REGISTER_CUDA
    return ret;
}
```

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-1a | 创建 `operator_registry.def` | `src/operator/operator_registry.def` | 从 `operator_init.c` 迁移所有 76 条注册 |
| R1-1b | 改造 `operator_init.c` 使用 X-macro | `src/operator/operator_init.c` | `#include` + 展开宏，消除 152 行重复代码 |
| R1-1c | 编写验证脚本 | `scripts/check_registry.sh` | 检查 `.def` 与实际 `.cu`/`.c` 文件是否一致 |
| R1-1d | 更新文档 | `ARCHITECTURE.md`, `CODING_STYLE.md` | 说明新的注册方式 |

> ✅ **R1-1 已完成** — `operator_init.c` 从 163 行减至 45 行（-72%），新增算子只需改 2 个文件。

### R1-2: NCU 性能 Profile 基线 ⭐⭐⭐

**来源**：CUDAForge `run_ncu.py` 收集 24 项 NCU 硬件指标，反馈给 LLM 诊断瓶颈。

**目标**：为每个算子建立可量化的性能指标基线，回归测试时自动检测性能退化。

**当前问题**：只有端到端 latency（BERT 4.66ms），无算子级 SM 占用率/带宽利用率/寄存器压力等指标。

**方案**：建立算子级 profile 指标卡：

```c
// 算子性能 profile 结构
typedef struct {
    const char* op_name;
    float latency_ms;            // CUDA event 计时
    float sm_occupancy_pct;      // SM 占用率
    float dram_throughput_pct;   // 显存带宽利用率
    int   registers_per_thread;  // 寄存器压力
    float l1_hit_rate_pct;       // L1 缓存命中率
    float achieved_warps_pct;    // 实际 warp 占用率
} op_profile_t;
```

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-2a | 扩展 benchmark 框架 | `tests/bench_profile.cu` | 集成 NCU metrics 采集 |
| R1-2b | 建立算子 profile 基线 | `docs/PROFILE_BASELINE.md` | 记录各算子在 RTX 2050 上的性能指标 |
| R1-2c | CI 回归检测脚本 | `scripts/check_perf_regression.sh` | 对比当前 profile 与基线，偏差 >10% 告警 |

> ✅ **R1-2 已完成** — 25 个测试点覆盖 7 类算子，CUDA event 精确计时，回归检测脚本可用。

### R1-3: 运行时 Kernel 配置切换 ⭐⭐

**来源**：CUDA-Agent `config` 参数模式，运行时切换 block size。

**目标**：根据输入尺寸动态选择最优 block size，无需重新编译。

**当前问题**：block size 硬编码（`OPS_THREADS_PER_BLOCK=256`），matmul 有基于维度的启发式但不可外部配置。

**方案**：在 `operator_params_t` 中新增 `tuning_config` 字段：

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-3a | 扩展 `operator_params_t` | `src/operator/include/operator.h` | 新增 `tuning_config` 字段 |
| R1-3b | 改造高频算子 launcher | `relu_cuda.cu`, `softmax_cuda.cu`, `matmul_cuda.cu` | 支持 config 切换 |
| R1-3c | graph 调度器根据 tensor 尺寸选 config | `src/application/graph.c` | 输入 <1K 用 config=1, >1M 用 config=2 |

> ✅ **R1-3 已完成** — relu/softmax 支持 128/256/512 三档 template，matmul 支持 4 种 kernel 显式选择。

### R1-4: 测试工具库 + 多随机种子验证 ⭐⭐

**来源**：CUDA-Agent 5 轮随机验证 + CUDAForge 确定性 benchmark（固定种子 + `deterministic_algorithms`）。

**目标**：消除测试代码重复，提高覆盖率。

**当前问题**：

- `random_fill()`、`max_abs_diff()`、`compare()` 在 5+ 个文件中重复实现
- 无集中 `test_utils.h`
- 单元测试用固定输入，只覆盖 1 种情况

**方案**：

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-4a | 创建 `test_utils.h` | `tests/test_utils.h` | `test_random_fill()`, `test_allclose()`, `test_max_abs_diff()`, `RUN_N_TRIALS()` 宏 |
| R1-4b | 重构现有测试使用 test_utils.h | `test_bert_mha.c`, `test_mha_decode.c`, `test_conv_scale.c`, `test_ops_scale.c` | 消除重复代码 |
| R1-4c | 关键算子 5 轮随机验证 | `test_conv.c`, `test_matmul.c`, `test_softmax.c`, `test_layernorm.c` | 用 `RUN_N_TRIALS(5, ...)` 包装 |
| R1-4d | 端到端 3 轮随机验证 | `test_bert.c`, `test_gpt2.c` | 用 `RUN_N_TRIALS(3, ...)` 包装 |

> ✅ **R1-4 已完成** — `test_utils.h` 提供 8 个工具函数/宏，conv/matmul/softmax 各 5 轮随机验证通过。

### R1-5: 裸 CUDA API 调用检测 ⭐

**来源**：CUDA-Agent `block_torch_functional()` 约束验证模式。

**目标**：自动化检测绕过 `g_cuda` 接口层的直接 CUDA 调用。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-5a | CI 静态扫描脚本 | `scripts/check_raw_cuda.sh` | grep `.cu` 文件中裸 `cudaMalloc`/`cudaMemcpy` |
| R1-5b | Debug 模式 wrapper | `src/platform/include/cuda_debug_wrap.h` | 宏替换裸 CUDA API 为断言失败 |

> ✅ **R1-5 已完成** — 检测脚本发现 4 个已知违规，Debug wrapper 拦截 16 个 CUDA API。

### R1-6: GPU 硬件能力查询扩展 ⭐

**来源**：CUDAForge `gpu_specs.py` 为 8 种 GPU 提供详细规格，注入 LLM prompt。

**目标**：运行时查询 GPU 能力，为 kernel 调优提供硬件上下文。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-6a | 扩展 `cudaDeviceProp` 查询 | `src/platform/cuda/cuda_platform.cu` | 暴露 SM 数、共享内存/SM、寄存器/SM、TFLOPS |
| R1-6b | 新增 `gpu_caps_t` 结构 | `src/platform/include/cuda_ops.h` | 运行时 GPU 能力快照 |

> ✅ **R1-6 已完成** — `gpu_caps_t` 含 22 个字段，`cuda_print_gpu_caps()` 格式化输出可用。

### R1-7: 优化 Checklist 文档 ⭐

**来源**：CUDAForge 两阶段 Judge 模式（先诊断瓶颈，再选择优化策略）。

**目标**：按算子类型给出明确的优化路径。

| 算子类型 | 代表算子 | 优先优化方向 | 现状 |
| --- | --- | --- | --- |
| Element-wise | ReLU, GELU, SiLU, Add, Mul | Vectorized loads (float4) | 待优化 |
| Reduction | Softmax, LayerNorm, BatchNorm | Shared memory reduction | 已实现 |
| GEMM | MatMul, Conv | Tiling → Warp → Tensor Core | 4 级全覆盖 |
| Attention | MHA Fused | 融合 + tiled online softmax | 已实现 |
| Pooling | MaxPool, AvgPool | Shared memory 窗口复用 | 待优化 |
| Permute | Transpose, Reshape, Slice | 合并访问 + 零拷贝 | 基础实现 |

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-7a | 更新 CUDA_GUIDE.md | `docs/CUDA_GUIDE.md` | 新增"按算子类型的优化路径"+"NCU 诊断 Checklist"章节 |

> ✅ **R1-7 已完成** — CUDA_GUIDE.md 新增第 5/6 节（优化路径 + NCU 诊断 Checklist）。

### R1-8: 批量 Benchmark Runner ⭐

**来源**：CUDAForge 270 个 KernelBench 任务的自动化 benchmark 框架。

**目标**：一键跑所有 benchmark 并输出对比表。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R1-8a | 创建 benchmark runner 脚本 | `scripts/run_benchmarks.sh` | 自动运行所有 `bench_*.exe`，输出 CSV |
| R1-8b | 集成 compute-sanitizer | `scripts/run_sanitizer.sh` | 批量运行 compute-sanitizer，汇总结果 |

> ✅ **R1-8 已完成** — 4 个 benchmark 全部 PASS，支持 CSV/JSON/profile-only 模式。

---

## 已完成工作（v0.6.0）

### 里程碑

- Phase A/B/C: BERT 全套推理 + MHA 融合 kernel
- M1-M3: FlashAttention tiled + QKV 融合 + WMMA FP16 (8.96x 加速)
- O1-O2: ONNX opset>=13 兼容 + 外部数据错误日志
- F1: FP16 推理支持 (15 个 FP16 CUDA 算子 + dtype-aware 调度)
- F2: LLM 推理 (CausalMask + mha_decode + KV-cache + GQA + RoPE + 永久融合)
- 代码审阅: Critical/High/Medium 修复全部完成

### Critical — 正确性风险

| # | 任务 | 状态 |
| --- | --- | --- |
| C1 | CUDA_CHECK 改为返回错误码 | ✅ |
| C2 | graph.c realloc 失败处理 | ✅ |

### High — 功能完整性

| # | 任务 | 状态 |
| --- | --- | --- |
| H1 | FP16 Conv2D kernel | ✅ |
| H2 | FP16 MatMul kernel | ✅ |
| H3 | FP16 BatchNorm/LayerNorm | ✅ |
| H4 | FP16 Softmax | ✅ |
| H5 | Resize 双线性插值 | ✅ |
| H6 | ONNX Pad 算子 | ✅ |
| H7 | ONNX Clip 算子 | ✅ |

### Medium — 测试覆盖

| # | 任务 | 状态 |
| --- | --- | --- |
| M1 | RoPE 独立单元测试 | ✅ |
| M2 | FP16 算子单元测试 | ✅ |
| M3 | In-place 操作测试 | ✅ |
| M4 | 边界形状测试 | ✅ |
| M5 | TESTS.md 测试映射修正 | ✅ |

### Low — 代码质量

| # | 任务 | 状态 |
| --- | --- | --- |
| L1 | operator_params_t placeholder | ✅ |
| L4 | CUDA_CHECK 统一 | ✅ |
| L5 | README_en 同步 | ✅ |

### 暂缓

| # | 任务 | 说明 |
| --- | --- | --- |
| L2 | graph_execute 惰性 D2H | 需要 CPU/GPU 消费者追踪 |
| L3 | Application->Operator 层耦合 | 融合 pass 需访问算子内部参数 |

### F2: LLM 推理基础设施

| 项 | 状态 |
| --- | --- |
| OP_WHERE / OP_TANH / INT64 / BOOL tensor | ✅ |
| 批量 MatMul / Transpose/Concat 推断 | ✅ |
| 自回归生成 API / GPT-2 测试模型 | ✅ |
| GPT-2 端到端推理 (max_diff=4.77e-07) | ✅ |
| KV-cache decode | ✅ |

---

## R2: ggml 借鉴优化

> 参考 [ggml](https://github.com/ggml-org/ggml) 的量化系统、算子融合、CUDA Graph、VMM 内存池等设计模式。
> 参考项目已下载到 `ggml-ref/` 目录。

### R2-1: INT8 量化推理 ⭐⭐⭐

**来源**：ggml 的 42 种量化类型 + block quantization 设计。

**目标**：权重 INT8 量化，显存减半，推理速度提升。

**方案**：实现 `DATA_TYPE_I8` block quantization（每 128 个 float 一组，1 字节 scale + 128 字节量化值）。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R2-1a | 定义 INT8 block 结构 | `src/operator/nn/quantize_int.h` | ✅ `block_q8_t` (64元素/block, 3.75x 压缩) |
| R2-1b | CPU 量化/反量化 | `src/operator/nn/quantize.c` | ✅ `quantize_f32_to_q8()`, `dequantize_q8_to_f32()` |
| R2-1c | CUDA 量化 kernel | `src/operator/nn/quantize_cuda.cu` | ✅ GPU 量化/反量化 + INT8 MatMul |
| R2-1d | INT8 MatMul kernel | `src/operator/nn/quantize_cuda.cu` | ✅ `matmul_q8_f32_cuda` INT8×FP32 |
| R2-1e | ONNX loader 量化支持 | `src/application/model/onnx_loader.c` | 待实施 |
| R2-1f | 精度测试 | `tests/test_quantize.cu` | ✅ max_rel=6.97e-04 |

### R2-2: 算子融合扩展 ⭐⭐⭐

**来源**：ggml 的 10+ 种融合模式（MatMul+bias, RMSNorm+Mul+Add, SwiGLU 等）。

**目标**：减少 kernel launch 次数和显存往返。

**当前已实现**：Conv+Activation, MHA Fused。

**新增融合**：

| # | 融合模式 | 节省 | 说明 |
| --- | --- | --- | --- |
| R2-2a | MatMul + Add(bias) | 1 次 kernel launch + 1 次显存读写 | ✅ tiled + warp-tiled 融合 kernel |
| R2-2b | LayerNorm + Mul(scale) + Add(residual) | 2 次 kernel launch | 待实施 |
| R2-2c | SiLU + Mul (SwiGLU gate) | 1 次 kernel launch | 待实施 |
| R2-2d | GELU + Mul (GeGLU gate) | 1 次 kernel launch | 待实施 |

### R2-3: Warp-level 归约原语库 ⭐⭐

**来源**：ggml `common.cuh` 的 `warp_reduce_sum()`, `warp_reduce_max()`, `block_reduce()` 模板。

**目标**：统一 warp/block 归约接口，消除各算子中的重复实现。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R2-3a | 创建 CUDA 归约原语 | `src/platform/cuda/cuda_reduce.cuh` | ✅ `warp_reduce_sum/max`, `block_reduce_sum/max`, `quad_reduce_sum/max` |
| R2-3b | 重构现有算子使用 | `softmax_cuda.cu` | ✅ softmax 已重构，layernorm/reduce 待后续 |

### R2-4: CUDA VMM 内存池 ⭐⭐

**来源**：ggml 的 `ggml_cuda_pool_vmm`（虚拟内存管理池）。

**目标**：消除 `cudaMalloc` 开销，推理时零分配。

**方案**：预留 32GB 虚拟地址空间，按需映射物理页，LIFO 释放。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R2-4a | 实现显存池 | `src/platform/cuda/cuda_memory.cu` | ✅ free-list allocator，power-of-2 bucket，64MB cap |
| R2-4b | 集成池化接口 | `src/platform/cuda/cuda_memory.cu` | ✅ `cuda_device_alloc_pooled` + `cuda_device_free_sized` |
| R2-4c | 显存池 benchmark | `tests/bench_mem_pool.cu` | 待实施 |

### R2-5: CUDA Graph 集成 ⭐

**来源**：ggml 的 `ggml_cuda_graph`（图捕获 + 重放）。

**目标**：重复推理时减少 kernel launch 开销。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R2-5a | graph_execute 支持 CUDA Graph 捕获 | `src/application/engine/graph.c` | 初步实现已回退（SEGFAULT），需更仔细的 stream 管理 |
| R2-5b | 图缓存管理 | `src/application/engine/graph_cache.c` | 按 input shape 缓存 CUDA Graph |

### R2-6: Flash Attention FP16 优化 ⭐

**来源**：ggml 的 3 种 Flash Attention 内核族（MMA/vec/tile）。

**目标**：FP16 prefill 使用 Tensor Core MMA 加速。

| # | 任务 | 文件 | 说明 |
| --- | --- | --- | --- |
| R2-6a | MMA Flash Attention FP16 | `mha_fused_f16_cuda.cu` | ✅ S≤64 已有 WMMA 路径，S>64 待后续 |
| R2-6b | 与 FP32 混合调度集成 | `mha_fused_cuda.cu` | 待后续 |

### R2-7: Flash Attention v2 参考优化 ⭐⭐⭐

**来源**：`flash-attention-main/` 官方 FA2 实现（Tri Dao）。

**目标**：参考官方 FA2 的 tiling/softmax/归约策略，优化 `mha_fused_cuda.cu` 的 FP32 Flash Attention kernel。当前实现每个 block 只处理 1 行 Q，且用 smem 做全局归约，与官方差距 2-5×。

**参考文件**：

- `flash-attention-main/csrc/flash_attn/src/kernel_traits.h` — tiling 配置
- `flash-attention-main/csrc/flash_attn/src/softmax.h` — warp-level online softmax
- `flash-attention-main/csrc/flash_attn/src/flash_fwd_kernel.h` — forward kernel
- `flash-attention-main/csrc/flash_attn/src/flash_fwd_launch_template.h` — dispatch

| # | 任务 | 文件 | 优先级 | 预期收益 | 说明 |
| --- | --- | --- | --- | --- | --- |
| R2-7a | 二维 Tiling | `mha_fused_cuda.cu`, `mha_fused_int.h` | P0 | 2-5× | grid 改为 `(S/kBlockM, B, H_q)`，每 block 处理 kBlockM=64 行 Q |
| R2-7b | Warp-level 归约 | 新建 `cuda_reduce.cuh`, 修改 `mha_fused_cuda.cu` | P1 | 1.3-1.5× | `warp_reduce_sum/max` 替代 smem 归约 |
| R2-7c | exp2f 优化 | `mha_fused_cuda.cu` | P1 | 1.1-1.2× | `exp2f(x*log2e)` 替代 `expf(x)` |
| R2-7d | K/V 预计算 + 全局内存加载 | `mha_fused_cuda.cu` | P2 | 避免重复投影 | K/V 只计算一次存入 global buffer，attention kernel 直接读取 |
| R2-7e | Split-KV 长序列 | `mha_fused_cuda.cu` | P3 | S>512 时并行 | K/V 分割成多份并行计算 + LSE 加权合并 |

> ✅ **R2-7 全部完成** — warp 归约 + exp2f + K/V 预计算 + Split-KV 长序列。35/35 测试通过。

### R2-8: Flash Attention 硬件级优化 ⭐⭐⭐

**来源**：`cuda-samples-12.8/Samples/3_CUDA_Features/cudaTensorCoreGemm/cudaTensorCoreGemm.cu` NVIDIA 官方 Tensor Core GEMM 示例。

**目标**：参考官方 GEMM 的 smem 布局、向量化拷贝、流水线技术，进一步优化 flash attention kernel 的 memory efficiency。

**参考文件**：

- `cuda-samples-12.8/.../cudaTensorCoreGemm.cu` — skew padding (line 151-163), int4 拷贝 (line 133,317), 双 buffer (line 295-363)
- `flash-attention-main/csrc/flash_attn/src/kernel_traits.h` — smem swizzle (line 79-95)

| # | 任务 | 文件 | 优先级 | 预期收益 | 说明 |
| --- | --- | --- | --- | --- | --- |
| R2-8a | smem skew padding | `mha_fused_cuda.cu` | 高 | 1.2-1.5× | K/V smem 每行加 SKEW padding 消除 bank conflict |
| R2-8b | int4 向量化加载 | `mha_fused_cuda.cu` | 高 | 1.3-1.5× | K/V 从 global memory 用 int4 (16B) 加载替代逐 float |
| R2-8c | 双 buffer 流水线 | `mha_fused_cuda.cu` | 中 | 1.1-1.3× | 加载下一个 K/V tile 的同时计算当前 tile |

> ✅ **R2-8a ~ R2-8b 已完成** — smem skew padding (FA_SKEW=8) + int4 向量化加载。35/35 测试通过。R2-8c (双 buffer) 待后续优化。

---

## 远期规划

| 方向 | 来源 | 说明 |
| --- | --- | --- |
| MatMul AVX2 微内核 | CudaForge | 8x4 register tiling + FMA |
| ARM NEON 优化 | — | 移动端/嵌入式推理 |
| 多 GPU 支持 | ggml | 模型并行 / 流水线并行 |
| CUDA Graph 调度 | ggml | 图捕获 + 重放 |
| 模板实例化生成 | ggml | Python 脚本自动生成 .cu 实例化文件 |

---

## 进度

| 状态 | 数量 | 内容 |
| --- | --- | --- |
| 已完成 | 44 | Phase A/B/C, M1-M3, O1-O2, F1, F2, C1-C2, H1-H7, M1-M5, L1/L4/L5, T1, R1-1 ~ R1-8, SIMD-1 ~ SIMD-5, Flash Attention, R2-1, R2-2a~d, R2-3a, R2-4, R2-6a(partial), R2-7, R2-8a ~ R2-8b |
| 暂缓 | 2 | L2 (惰性 D2H), L3 (层耦合) |
| 计划中 | 3 | R2-1e, R2-3b, R2-5 |
| 远期 | 5 | AVX2 微内核/ARM NEON/多GPU/CUDA Graph/模板生成 |

> **最后更新**: 2026-06-22。R2-1 INT8 量化, R2-2b~d 融合 kernel 完成。
