# GPU Profiling 工作流指南

本项目使用 NVIDIA Nsight Systems / Nsight Compute / compute-sanitizer 三层工具分析性能与正确性。本文档记录工具路径、命令、分析流程，以及实战经验（Flash Attention v2 优化案例）。

## 工具环境

| 工具 | 路径 | 用途 |
| --- | --- | --- |
| Nsight Systems (nsys) | `C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe` | 整体时间线、kernel/API 时间分布、内存传输 |
| Nsight Compute (ncu) | `C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.1.1/ncu.bat` | kernel 级硬件指标（occupancy、roofline、warp state） |
| compute-sanitizer | `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/compute-sanitizer.bat` | 内存错误（leak/OOB/misaligned/race） |
| sqlite3 (nsys 自带) | `.../Nsight Systems 2025.6.3/target-windows-x64/sqlite3.exe` | 查询 nsys 生成的 .sqlite 报告 |

> ncu 需要管理员权限访问 GPU 性能计数器（`ERR_NVGPUCTRPERM`）。无管理员时退化为仅 launch stats；occupancy 可从 nsys 的 regs/smem 理论推算。

## GPU 规格（RTX 2050, sm_86, GA107）

- 16 SMs
- 65536 寄存器/SM
- 100KB 共享内存/SM（48KB 默认 + 52KB opt-in）
- 1536 线程/SM 上限

occupancy 公式：`blocks_per_SM = min(regs_limit, smem_limit, thread_limit, 16)`
- regs: `65536 / (regs_per_thread × threads_per_block)`
- smem: `100KB / smem_per_block`

## 三层分析流程

### 1. compute-sanitizer（正确性先行）

每次改 kernel 后**必须先过**，否则性能数据可能建立在 UB 之上。

```bash
# 内存检查
compute-sanitizer --tool memcheck ./build/Release/test_bert_mha.exe

# 期望: ERROR SUMMARY: 0 errors
# 若有 error，先修正确性再谈性能
```

### 2. Nsight Systems（整体瓶颈定位）

```bash
# profile benchmark，生成 .nsys-rep + .sqlite
nsys profile --stats=true --trace=cuda --force-overwrite=true -o prof ./build/Release/bench_bert_mha.exe

# 查 kernel 时间分布（按 kernel 名聚合）
sqlite3 prof.sqlite -header -column "
SELECT s.value as kernel, COUNT(*) as calls,
       printf('%.3f', SUM(k.end-k.start)/1e6) as total_ms,
       printf('%.3f', AVG(k.end-k.start)/1e6) as avg_ms,
       k.registersPerThread as regs,
       printf('%.1f', (k.dynamicSharedMemory+k.staticSharedMemory)/1024.0) as smem_kb
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
GROUP BY s.value ORDER BY total_ms DESC LIMIT 15;"
```

按 grid 维度细分（区分不同序列长度）：

```bash
sqlite3 prof.sqlite -header -column "
SELECT printf('S=%d', k.gridX*64) as seq, COUNT(*) as calls,
       printf('%.3f', AVG(k.end-k.start)/1e6) as avg_ms,
       k.registersPerThread as regs,
       printf('%.1f', k.dynamicSharedMemory/1024.0) as dyn_smem,
       printf('%.1f', k.staticSharedMemory/1024.0) as stat_smem
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE s.value LIKE 'mha_flash_attn_v2_f16%' GROUP BY k.gridX;"
```

### 3. Nsight Compute（kernel 级深挖，需管理员）

```bash
# occupancy + launch stats（单 kernel）
ncu --target-processes all --launch-count 1 --section Occupancy --section LaunchStats ./build/Release/test_bert_mha.exe

# 完整指标（roofline、warp state、memory workload）
ncu --target-processes all --launch-skip 20 --launch-count 3 \
    --section SpeedOfLight --section MemoryWorkloadAnalysis \
    --section ComputeWorkloadAnalysis --section WarpStateStats ./build/Release/bench_bert_mha.exe
```

> 无管理员权限时 ncu 无法读 perf counter，但 nsys 已提供 regs/smem，可手动算 occupancy。

## 实战案例：Flash Attention v2 优化

### 案例 1：grid 加 H_q 维度（R3-a）

**nsys 发现**：`mha_flash_attn_v2_kernel` 占 99% 时间，grid 仅 `(B*S)`=512 blocks，每个 block 串行处理 12 个 head。

**根因**：head 串行循环，并行度不足。

**修复**：grid 改为 `(B*S, 1, H_q)`，每个 head 独立一个 block。

**结果**：S=512 从 391ms → 310ms（1.27×）。

### 案例 2：多行 Q tiling + warp 级归约（R4-a/b）

**nsys 发现**：仍每 block 1 行 Q，向量×矩阵无法发挥算力。

**修复**：BM=64，4 warps 各 16 行 Q，K/V 共享，warp 级 `warp_reduce_sum/max`（无 smem、无跨 warp 通信）。

**结果**：S=512 从 310ms → 92.5ms（3.34×）。

### 案例 3：FP16 WMMA Tensor Core（R4-c/d）

**调试**：WMMA fragment 有正确值但 `store_matrix_sync` 到动态 `__half*→float*` reinterpret_cast 的共享内存静默失败（sm_86 已知问题）。

**修复**：S_fmem/acc_o_fmem 改用静态 `__shared__ float`。

**结果**：S=512 从 92.5ms → 31.6ms（2.9×）。

### 案例 4：smem 精简提升 occupancy（本轮）

**nsys 发现**：FP16 kernel 68KB smem（32KB static + 36KB dynamic），RTX 2050 每 SM 100KB → **仅 1 block/SM**（occupancy 8.3%）。

**根因**：smem 是 occupancy 瓶颈。

**修复**：
1. K/V 共用 smem（union）— K 在 Q·Kᵀ 后消费完，V 才加载，两者不共存
2. BM 64→32 — Q/S/acc_o/P 减半，block 用 64 线程（2 warps × 16 行，仍适配 WMMA）

**结果**：smem 68KB → 33.5KB → **2 blocks/SM**。kernel 时间再降 ~15%（S=512: 26.3ms → 22.0ms）。

## 集成到测试流程

### 必查项（每次 kernel 改动）

1. **compute-sanitizer 0 errors**（正确性底线）
2. **36/36 测试通过**（ctest）
3. **benchmark 无性能回退**（与上次对比）

### 性能回归检测（可选，定期）

```bash
# 1. 跑 benchmark 存基线
./build/Release/bench_bert_mha.exe 2>&1 | grep "S=" > prof_baseline.txt

# 2. nsys 抓 kernel 时间分布
nsys profile --stats=true --trace=cuda --force-overwrite=true -o prof_current ./build/Release/bench_bert_mha.exe

# 3. 对比 smem/regs（occupancy 是否回退）
sqlite3 prof_current.sqlite -header -column "SELECT s.value, k.registersPerThread, (k.dynamicSharedMemory+k.staticSharedMemory)/1024.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id WHERE s.value LIKE 'mha_flash%' GROUP BY s.value;"
```

### 关键指标速查

| 指标 | 健康值 | 来源 |
| --- | --- | --- |
| compute-sanitizer errors | 0 | memcheck |
| smem/block (FP16 kernel) | ≤ 50KB（目标 2+ blocks/SM） | nsys dynamicSharedMemory + staticSharedMemory |
| regs/thread | ≤ 64 | nsys registersPerThread |
| kernel 时间占比 | 主 kernel < 95% | nsys cuda_gpu_kern_sum |
| 精度 max_rel | < 1% (FP16) / < 0.1% (FP32) | test_bert_mha precision 段 |

## 常见陷阱

1. **动态 smem + WMMA store**：`extern __shared__ __half` 经 `reinterpret_cast<float*>` 后 `store_matrix_sync` 静默失败 → 用静态 `__shared__ float`。
2. **smem opt-in 阈值**：动态 smem > 48KB 需 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`，否则 launch 失败 "invalid argument"。静态 + 动态合计超 48KB 也要 opt-in。
3. **CUDA Graph + stream 0**：所有 kernel 用 stream 0（legacy stream）无法被 capture → 需统一非默认 stream，否则禁用 capture。
4. **测试数据条件性**：`random_fill` 产生全负数据，使投影 Q~190、softmax 退化 one-hot，FP16 舍入扰动 argmax 造成巨大误差 → 测试需缩放数据模拟 post-LayerNorm 良态。
