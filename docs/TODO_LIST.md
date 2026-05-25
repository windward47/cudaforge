# CudaForge Todo List

> 基于 2026-05-24 全仓库代码审查生成。旧版历史记录见 [TODO_LIST_20260524.md](TODO_LIST_20260524.md)。

---

## 关键 (Critical) — 影响正确性，需立即修复

### C1. ✅ 修复 `static` 参数结构体（线程安全 Bug）

**文件**: `src/application/model/onnx_loader.c:558-617`

`conv_params_t cp`、`pool_params_t pp`、`matmul_params_t mp`、`batchnorm_params_t bp` 声明为 `static`：

- 连续调用 `inference_session_load()` 会互相覆盖参数
- 同一图中多个同类型节点（如两个不同 stride 的 Conv）会共享最后一个参数集
- 使得整个 ONNX 加载路径在多线程环境下不安全

**修复方向**: 去掉 `static`，改为栈上普通局部变量或按节点动态分配。

### C2. ✅ 修复 Float 属性 wire type 5 解码错误

**文件**: `src/application/model/onnx_parser.c:86-90` + `pb_field_get_float()`

`pb_field_get_float()` 对 wire type 5（32-bit fixed）编码的属性始终返回 **0.0**：

- wire type 5 的原始字节被误标为 `PB_WIRE_VARINT`
- `pb_field_get_float` 走 `varint_value == 0` 分支，丢弃了实际存储的 float 值
- 任何包含 float 类型 AttributeProto 字段（如 BatchNorm 的 `epsilon`）的 ONNX 模型都会加载错误的参数

**修复方向**: 在 `pb_field_get_float` 中增加 wire type 5 分支，从 `length_delimited.data` 中 `memcpy` 出 `float`。

### C3. 修复 Pooling kernel OW > 256 数据丢失 Bug

**文件**: `src/operator/nn/pooling_cuda.cu:67`

`launch_pool()` 中 `dim3 block(OW > 256 ? 256 : (unsigned)OW, 1, 1)` 当 OW > 256 时只启动 256 个线程，但 grid 维度只有 `(OH, C, N)` 没有沿 OW 分块，导致 OW 超过 256 的列永远不被处理。

- 输出中 OW > 256 的列为未初始化垃圾值
- 任何宽度 > 256 的池化操作都会产生静默错误结果

**修复方向**: 将 grid.x 改为 `(OW + BLOCK_SIZE - 1) / BLOCK_SIZE` 沿 OW 分块，kernel 内用 `blockIdx.x * blockDim.x + threadIdx.x` 计算 ow。

### C5. ✅ 补齐图像分类所需算子 — Softmax

**目标**: 使框架能够运行 ONNX 图像分类模型。Softmax 是分类网络输出层的标配，将 logits 转为概率分布。

**实现清单**:
- `src/operator/nn/softmax.c` — CPU 实现（沿 axis 的 softmax）
- `src/operator/nn/softmax_cuda.cu` — CUDA kernel（shared memory reduction per row）
- `src/operator/nn/softmax_int.h` — `softmax_params_t { int64_t axis; }`
- `src/application/engine/graph.h` — 添加 `OP_SOFTMAX`
- `src/application/engine/graph.c` — `op_name()` 添加映射
- `src/application/model/onnx_loader.c` — `map_onnx_op()` 添加 "Softmax"
- `src/operator/operator_init.c` — 注册 CPU + CUDA
- `CMakeLists.txt` — 添加源文件 + 测试 target
- `src/operator/nn/test/test_softmax.c` — 单元测试

### C6. ✅ 补齐图像分类所需算子 — Add（逐元素加法）

**目标**: 元素级加法。ONNX Conv 常用 Conv(weight) + Add(bias) 模式，残差网络依赖 Add 做跳跃连接。

**实现清单**:
- `src/operator/nn/add.c` — CPU 逐元素加法（支持 broadcast: A[NCHW] + B[C]）
- `src/operator/nn/add_cuda.cu` — CUDA kernel（逐元素并行 + broadcast）
- `src/operator/nn/add_int.h` — `add_params_t { bool broadcast; }`
- `src/application/engine/graph.h` — 添加 `OP_ADD`
- `src/application/engine/graph.c` — `op_name()` 添加映射
- `src/application/model/onnx_loader.c` — `map_onnx_op()` 添加 "Add"
- `src/operator/operator_init.c` — 注册 CPU + CUDA
- `CMakeLists.txt` — 添加源文件 + 测试 target
- `src/operator/nn/test/test_add.c` — 单元测试（含广播）

### C7. ✅ 补齐图像分类所需算子 — GlobalAveragePool

**目标**: 全局平均池化。现代 CNN（ResNet、MobileNet）用它替代 Flatten→FC，将 NCHW 变为 NC×1×1。

**实现清单**:
- `src/operator/nn/globalavgpool.c` — CPU 实现（对 H,W 维度求均值）
- `src/operator/nn/globalavgpool_cuda.cu` — CUDA kernel（per-channel reduction）
- `src/operator/nn/globalavgpool_int.h` — `globalavgpool_params_t { int64_t N, C, H, W; }`
- `src/application/engine/graph.h` — 添加 `OP_GLOBALAVGPOOL`
- `src/application/engine/graph.c` — `op_name()` 添加映射
- `src/application/model/onnx_loader.c` — `map_onnx_op()` 添加 "GlobalAveragePool"
- `src/operator/operator_init.c` — 注册 CPU + CUDA
- `CMakeLists.txt` — 添加源文件 + 测试 target
- `src/operator/nn/test/test_globalavgpool.c` — 单元测试

### C8. ✅ 补齐图像分类所需算子 — Reshape

**目标**: 张量形状变换（零拷贝，仅修改 ndim/shape 元数据）。Conv→Gemm 之间需要 Reshape 展平特征图。

**实现清单**:
- `src/operator/nn/reshape.c` — CPU 实现（零拷贝: 仅改 shape/ndim）
- `src/operator/nn/reshape_int.h` — `reshape_params_t { int ndim; int64_t shape[8]; }`
- `src/application/engine/graph.h` — 添加 `OP_RESHAPE`
- `src/application/engine/graph.c` — `op_name()` 添加映射
- `src/application/model/onnx_loader.c` — `map_onnx_op()` 添加 "Reshape"，处理 shape 属性
- `src/operator/operator_init.c` — 注册 CPU
- `CMakeLists.txt` — 添加源文件 + 测试 target
- `src/operator/nn/test/test_reshape.c` — 单元测试

### C4. 修复 conv2d_im2col per-batch 逐元素动态分配

**文件**: `src/operator/nn/conv_cuda.cu:61,79`

`conv2d_im2col()` 在 `for (int64_t n = 0; n < N; n++)` 循环内执行 `g_cuda.device_alloc` + `g_cuda.device_free`：

- 对于 N=64 的 batch，产生 64 次 cudaMalloc/cudaFree 调用
- cudaMalloc/cudaFree 是同步操作，极度拖慢 im2col 路径
- 同时产生显存碎片

**修复方向**: 将 col_buf 的分配移到循环外，循环内复用同一 buffer。

---

## 高优先级 (High)

### H1. ✅ 修复 CUDA 推理后 GPU→CPU 数据竞争

**文件**: `src/application/engine/graph.c:354-368`

`graph_execute` CUDA 路径使用 `cudaMemcpyAsync` 将结果拷回主机后，**未调用同步**就直接 `memcpy` 读取。`cudaMemcpyAsync` 不保证返回时数据已到达 host，即使使用 default stream 也不保证 host 可见性。

**修复方向**: 在 host 端 `memcpy` 之前调用 `g_cuda.stream_synchronize(0)`。

### H2. ✅ 替换 `_strdup` 为跨平台实现

**文件**: `src/application/model/onnx_loader.c:949,964`

`_strdup` 是 MSVC 特有函数，Linux GCC/Clang 下链接失败，ONNX 加载器目前仅限 Windows。

**修复方向**: 改为标准 C 的 `malloc` + `memcpy`，或添加平台宏。

### H3. ✅ 补全逐元素算子的 NULL 参数校验

**文件**: `src/operator/nn/relu.c:15`, `src/operator/nn/activations.c:12`

通过 `inputs[1]` 传递 `int64_t n` 元数据，但入口只校验 `inputs[0]` 和 `outputs[0]`，不检查 `inputs[1]`。传入 NULL 时静默崩溃而非返回错误码。

**修复方向**: 校验分支增加 `!inputs[1]` 检查，返回 `-EINVAL`。

### H4. ✅ 测试文件改用 `g_cuda` 接口代替直接 CUDA API 调用

**文件**: `tests/test_cuda_kernels.cu:26-27`, `tests/bench_operators.cu:38-42`

CLAUDE.md 规则 #2 明确规定所有 CUDA API 调用必须通过 `g_cuda` 接口层，但测试和 benchmark 直接调用了 `cudaMalloc`/`cudaMemcpy`/`cudaFree`/`cudaDeviceSynchronize`。

**修复方向**: 改为 `g_cuda.device_alloc` / `g_cuda.memcpy_h2d` / `g_cuda.device_free` / `g_cuda.stream_synchronize`。

### H5. ✅ 删除废弃的初始值设定项重复解析循环 + 83 行注释

**文件**: `src/application/model/onnx_loader.c:695-724`

第一个初始值设定项解析循环（L695-724）为每个 initializer 创建空名字条目，随后第二个循环（L777-803）又重新解析。导致：

- `parse_tensor_proto` 被调用两次 → 内存分配翻倍
- `MAX_TENSOR_INFOS` 槽位消耗翻倍
- 空名字条目无实际用途

**修复方向**: 删除 L695-724 的循环，只保留 L777-803 的解析逻辑。

### H6. 修复 graph_execute 逐节点 malloc/free 开销

**文件**: `src/application/engine/graph.c:278-279`

每个节点执行都 `malloc` + `free` 临时 `op_inputs`/`op_outputs` 数组。对于 ResNet-50（~100+ 节点），这是 100+ 次堆分配。

**修复方向**: 使用固定大小的栈上数组 `void* buf[32]` 分配 op_inputs 和 op_outputs，对大节点 fallback 到 malloc。

### H7. Batchnorm cuda 入口增加 NULL 参数校验

**文件**: `src/operator/nn/batchnorm_cuda.cu:152`

`batchnorm_f32_cuda` 只校验 `inputs[0]` 和 `outputs[0]`，但实际使用 `inputs[1]`~`inputs[5]`（gamma, beta, mean, var, hw）。传入 NULL 时静默崩溃。

**修复方向**: 校验分支增加 `!inputs[1] || !inputs[2] || !inputs[3] || !inputs[4] || !inputs[5]` 检查。

---

## 中优先级 (Medium)

### M1. ✅ 精简 83 行废弃注释

**文件**: `src/application/model/onnx_loader.c:688-770`

一大段内部设计推演注释（包含错误推断）遗留在代码中，应精简为 2-3 行说明或完全删除。

### M2a. ✅ C3: 修复 Pooling kernel OW > 256 数据丢失 Bug → 见 C3

### M2b. ✅ C4: 修复 conv2d_im2col per-batch 动态分配 → 见 C4

### M3. ✅ MatMul 实现 Tensor Core (sm_86)

**文件**: `src/operator/blas/matmul_cuda.cu`

已实现 `matmul_f32_tc_kernel`（~80行），使用 `nvcuda::wmma` API 的 m16n16k16 FP16→FP32 MMA 操作。dispatch 增加 Tensor Core 路径：`M >= 512 && N >= 512 && K >= 512 && M % 16 == 0 && N % 16 == 0`。FP32 输入通过 `__float2half()` 转换为 FP16 加载，累加和输出保持 FP32。

### M4. ✅ conv_cuda 硬编码魔数改为设备属性查询

**文件**: `src/operator/nn/conv_cuda.cu`

已移除硬编码 `CONV_MAX_SMEM_BYTES`，改为 function-static 缓存模式通过 `cudaDeviceGetAttribute(..., cudaDevAttrMaxSharedMemoryPerBlock, 0)` 动态查询。首个调用查询设备，后续调用复用缓存值。查询失败时 fallback 到 48KB (sm_86 default)。

### M5. Conv 共享内存 kernel 性能优化

**文件**: `src/operator/nn/conv_cuda.cu:140-214`

`conv2d_f32_direct_smem_kernel` 每 channel 重载整个 input window，未做 C 维度分块。且使用 `extern __shared__` 未做 bank conflict padding。

**修复方向**:
- 对 C 维度做 Kc tile（如一次处理 4 个 channel）
- 共享内存 +1 padding 避免 bank conflict（同 matmul warp kernel 的做法）
- 使用 `__ldg()` read-only cache 加载 input

### M6. ✅ 元素级算子融合（Kernel Fusion）— Application 层

**文件**: `src/application/engine/graph.c`, `src/operator/nn/conv_cuda.cu`, `src/operator/nn/conv_int.h`

已在 `graph_execute` 增加 CUDA 路径 kernel fusion pre-pass：扫描拓扑序检测 Conv/MatMul→ReLU/Sigmoid/GELU 模式，在 Conv params 设置 `fuse_activation` 标记，跳过被融合的激活节点。Conv CUDA kernel（direct + smem）均已内联 ReLU/Sigmoid/GELU 激活函数。输出张量路由通过 `effective_output_tids` 非破坏性重定向到激活的输出，保证重复执行安全和图拓扑不变。

### M2. 池化算子增加输出尺寸合法性校验

**文件**: `src/operator/nn/pooling.c:18-19`, `src/operator/nn/pooling_cuda.cu:82-83`

`(H + 2*pad - kernel) / stride + 1` 可能返回 0 或负数，CPU 端作为循环边界导致未定义行为。

**修复方向**: 在 shape 计算后增加 `TEST_ASSERT(OH > 0 && OW > 0)` 或返回错误码。

### M3. 强化测试断言

| 文件 | 问题 |
| --- | --- |
| `src/operator/nn/test/test_conv.c:54-59` | 只检查输出 "不全为 0"（NaN 也能通过） |
| `src/operator/nn/test/test_batchnorm.c:34-39` | 只检查符号和大致范围，不验精确值 |
| `tests/test_cuda_kernels.cu:210-213` | GELU 测试跳过第 4 个元素 |

**修复方向**: 以已知输入/权重计算出精确期望值进行 assert。

### M4. 添加 in-place 算子测试

ReLU、Sigmoid、GELU 标记了 `OP_FLAG_IN_PLACE`，但无测试验证 `inputs[0] == outputs[0]` 时的行为（CPU + CUDA 路径）。

### M5. 删除 `op_name()` 中的死代码

**文件**: `src/application/engine/graph.c:202-216`

`suffix` 变量被赋值后通过 `(void)suffix` 静默丢弃。实际后缀拼接在 `graph_execute` L255 独立完成。

### M6. 消除 Magic Numbers

| 位置 | 魔数 | 建议 |
| --- | --- | --- |
| `src/operator/nn/relu_cuda.cu:24` | `256` 线程 | 定义 `THREADS_PER_BLOCK` 常量 |
| `src/operator/nn/activations_cuda.cu:22` | `256` 线程 | 同上 |
| `src/operator/nn/pooling_cuda.cu:67` | `256` 线程 | 同上 |
| `src/operator/nn/conv_cuda.cu:247` | `8192` 共享内存阈值 | 定义命名常量或 `cudaDeviceGetAttribute` 查询 |

### M7. ONNX 加载器 `find_tensor` 线性扫描改哈希表

**文件**: `src/application/model/onnx_loader.c:100-106`

当前 O(n) 线性扫描，每个节点属性解析、weight 匹配都调用。对于 >1000 tensor 的模型，加载时间主要消耗于此。

**修复方向**: 使用开源 `uthash` 或简单的开放寻址哈希表替代线性扫描。

### M8. protobuf `pb_find_field` 线扫 → 索引跳表

**文件**: `src/application/model/onnx_parser.c:140-146`

`pb_find_field` 每次扫描整个 fields 数组。`onnx_loader.c` 中的链式 `pb_find_field` → `pb_field_as_message` 调用导致每个属性访问都是 O(n²)。

**修复方向**: 解析时建立 `int index_by_fn[256]` 跳表，将 O(n) 查找降为 O(1)。

### M9. `CUDA_CHECK` 宏 exit() → 错误返回

**文件**: `src/platform/include/cuda_ops.h:22-29`

`CUDA_CHECK` 在错误时调用 `exit(EXIT_FAILURE)`。作为推理库，应该返回错误码让调用方决定如何处理，而不是直接终止进程。

**修复方向**: 改为 `goto error` 模式或使用 `setjmp/longjmp` 通知调用方。

**文件**: `src/platform/tensor.c:77-79`

`cudaMemcpyAsync` 后立即 `device_free` 的语义有隐式假设（设备内存不会在异步复制完成前被重用），调用方 `graph_execute` 在没有同步的情况下立即读取 host 数据。需要注释说明为什么当前做法是安全的（与 H1 相关）。

### M10. ✅ M5: 删除 `op_name()` 中的死代码 → 已完成

### M11. `tensor_copy_to_host` 增加行为注释

**文件**: `src/platform/tensor.c:77-79`

`cudaMemcpyAsync` 后立即 `device_free` 的语义有隐式假设（设备内存不会在异步复制完成前被重用），调用方 `graph_execute` 在没有同步的情况下立即读取 host 数据。需要注释说明为什么当前做法是安全的（与 H1 相关）。

### M12. Conv→MatMul 跨算子直接调用改为通过注册表

**文件**: `src/operator/nn/conv.c:10-11`, `src/operator/nn/conv_cuda.cu:6-7`

Conv 通过 `extern` 声明直接调用 `matmul_f32` / `matmul_f32_cuda`，绕过了算子注册表调度层。建议改为通过注册表查找，降低模块耦合。

---

## 低优先级 (Low)

### L1. CMake `find_package(OpenMP)` 增加结果检查

**文件**: `CMakeLists.txt:41`

`find_package(OpenMP)` 无 `REQUIRED`，也未检查 `OpenMP_FOUND`。OpenMP 未找到时会在链接阶段产生难以理解的错误。

### L2. 内部头文件 `conv_int.h` / `matmul_int.h` 改为 PRIVATE

**文件**: `CMakeLists.txt:100-104, 138-142`

内部实现头文件通过 `PUBLIC` 暴露给所有链接者，应改为 `PRIVATE`。

### L3. MSVC 警告 `/wd4996` 用平台宏替代

**文件**: `CMakeLists.txt:145, 262`

用 `#ifdef _MSC_VER` → `_strdup` / `#else` → `strdup` 的跨平台宏替代 `/wd4996` 抑制。

### L4. `main.c` 补充 `cuda_platform_finalize()` 调用

**文件**: `src/main.c:40`

`platform_finalize()` 之后应调用 `cuda_platform_finalize()` 释放 CUDA 资源。

### L5. `data_type_get_info()` 返回值增加 NULL 检查

**文件**: `src/platform/tensor.c:26, 56, 74`

在 `tensor_create`、`tensor_copy_to_device`、`tensor_copy_to_host` 中 `info->size` 解引用前检查 `info == NULL`。

### L6. 文档与代码同步

| 文档 | 问题 |
| --- | --- |
| `docs/ARCHITECTURE.md:57-60` | 描述的 `g_platform_x86` 与实际的 `s_platform_x86` 命名不一致 |
| `docs/ARCHITECTURE.md:63-73` | 引用了不存在的 `src/operator/math/`、`src/operator/utils/` 目录 |
| `docs/CUDA_GUIDE.md:198-199` | 引用了不存在的 `tensor_rand()`、`tensor_like()` 函数 |

### L7. 测试文件中补充 `platform_finalize()` 调用

`tests/test_cuda_kernels.cu`、`tests/bench_operators.cu` 调用了 `platform_init()` 但未调用 `platform_finalize()`，与其他测试行为不一致。

### L8. CUDA memory 函数返回 cudaError_t 而非忽略

**文件**: `src/platform/cuda/cuda_memory.cu`

`cuda_device_alloc` 等函数在 `CUDA_CHECK` 失败时直接 `exit()`，错误不可恢复。调用方也无法区分 "分配失败" 和 "进程退出"。

**修复方向**: 将 `CUDA_CHECK` 改为 `cudaError_t` 返回值检查，失败时返回错误码而非退出。

### L9. `tensor_copy_to_host` 即时释放 device 内存策略优化

**文件**: `src/platform/tensor.c:78-79`

每次 D2H 后立即释放 device 内存。对于被多个下游节点消费的中间张量，导致重复 alloc→copy→free。建议增加 LRU 缓存或引用计数。

---

## 进度

| 状态 | 数量 | 清单 |
| --- | --- | --- |
| 已完成 | 31 | C1-C8, H1-H7, M1-M6, M10-M12, L1, L5, 以及 ONNX external_data 加载修复 + 真实 MNIST CNN 端到端推理验证 + MatMul Tensor Core + Conv 设备属性查询 + Kernel Fusion |
| 进行中 | 0 | — |
| 待修复 | 7 | L2-L4, L6-L9 |

> **最后更新**: 2026-05-25。第五轮: M3 (MatMul Tensor Core wmma), M4 (Conv 动态共享内存查询), M6 (Conv/MatMul→Activation kernel fusion) 完成，14/14 测试通过。
