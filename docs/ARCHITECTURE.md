# 架构规范

## 1. 整体架构

本项目采用**分层 + 模块化**架构，自底向上分为三层：

```
┌─────────────────────────────────────────┐
│           Application 层                 │  ← 模型加载、推理引擎、应用集成
├─────────────────────────────────────────┤
│           Operator 层                    │  ← 算子注册/调度、张量运算
├─────────────────────────────────────────┤
│           Platform 层                    │  ← 硬件抽象、内存管理、线程调度
└─────────────────────────────────────────┘
```

### 依赖方向

```
Application → Operator → Platform
```

每一层只依赖其下层，严禁反向依赖或跨层依赖。

## 2. 各层职责

### 2.1 Platform 层

**目录位置**：`src/platform/`

提供硬件和操作系统抽象，屏蔽平台差异。分为 **CPU** 和 **CUDA** 两大子系统。

#### CPU 子系统（`src/platform/cpu/`）

- **硬件抽象**（HAL）：缓存行大小、向量指令集检测（AVX2/NEON/SVE）、内存屏障、原子操作
- **内存管理**：对齐分配、页锁定内存、内存池
- **线程与并行**：线程池、任务调度、NUMA 亲和性

#### CUDA 子系统（`src/platform/cuda/`）

- **设备管理**：设备枚举与属性查询（计算能力、SM 数量、显存大小）、CUDA 上下文创建/销毁
- **显存管理**：`cudaMalloc`/`cudaFree` 封装、统一内存（Unified Memory）、页锁定内存（Pinned Memory）、显存池
- **Stream 管理**：CUDA Stream 创建/销毁/同步、Event 记录与测量
- **工具封装**：错误检查宏 `CUDA_CHECK`、kernel launch 配置辅助（grid/block 尺寸计算）

```c
/* 平台抽象层接口 — 各平台各自实现 */
typedef struct {
    const char* name;
    int   (*init)(void);
    int   (*finalize)(void);
    void* (*alloc)(size_t size, size_t align);
    void  (*free)(void* ptr);
    int   (*get_core_count)(void);
    int   (*get_cache_line_size)(void);
} platform_t;

extern const platform_t g_platform_x86;
extern const platform_t g_platform_arm;
extern const platform_t g_platform_riscv;
```

### 2.2 Operator 层

**目录位置**：`src/operator/`

核心算子库，每个算子独立模块，通过统一接口注册。

| 模块目录 | 说明 |
| --- | --- |
| `src/operator/math/` | 数学运算（add/mul/norm/softmax 等） |
| `src/operator/nn/` | 神经网络算子（conv/pool/batch_norm/relu 等） |
| `src/operator/blas/` | 矩阵运算（matmul/gemm/gemv） |
| `src/operator/utils/` | 辅助算子（reshape/transpose/concat/slice） |

**算子接口规范**：

```c
/* 每个算子遵循的统一签名 */
typedef int (*operator_func_t)(const void* inputs[],
                                void* outputs[],
                                const operator_params_t* params,
                                stream_t* stream);

typedef struct {
    const char*       name;          /* 算子名，如 "matmul_f32" */
    const char*       data_type;     /* "f32", "f16", "i8" 等 */
    operator_func_t   func;          /* 算子函数指针 */
    int               version;       /* 算子版本 */
    uint32_t          flags;         /* 特性标记（IN_PLACE, ALLOW_ALIAS 等） */
} operator_registry_t;
```

### 2.3 Application 层

**目录位置**：`src/application/`

基于算子层构建的上层能力。

- 模型加载与解析（ONNX、TFLite 等格式）
- 推理引擎（模型 → 计算图 → 算子调度 → 执行）
- 应用集成 API（C API / CLI / FFI）

## 3. 算子开发规范

### 算子实现文件结构

```
src/operator/nn/
├── conv.c              # 通用实现（纯 C fallback）
├── conv_int.h          # 内部共享声明
├── conv_x86.c          # x86 优化实现（AVX2/AVX512）
├── conv_arm.c          # ARM 优化实现（NEON/SVE）
└── test/
    └── test_conv.c     # 单元测试
```

### 算子实现要求

1. **必须有通用 fallback**：纯 C 实现作为兜底，平台优化作为可选加速
2. **输入校验**：所有算子入口做维度匹配和数据类型检查
3. **in-place 支持标注**：如果算子支持原地操作（输入输出同一内存），在 `operator_registry_t.flags` 中标注
4. **数值稳定性**：浮点算子需给出误差上界（ULP），大数相加使用 Kahan 求和等稳定算法

```c
/* 算子模板 */
int conv2d_f32(const void* inputs[],
               void* outputs[],
               const operator_params_t* params,
               stream_t* stream) {
    /* 1. 参数校验 */
    if (!inputs[0] || !inputs[1] || !outputs[0]) return -EINVAL;
    const tensor_t* input  = (const tensor_t*)inputs[0];
    const tensor_t* weight = (const tensor_t*)inputs[1];
    tensor_t* output       = (tensor_t*)outputs[0];

    /* 2. 形状校验 */
    int ret = conv_check_shapes(input, weight, output, params);
    if (ret != 0) return ret;

    /* 3. 选择优化路径 */
    #if defined(USE_AVX512)
        ret = conv2d_f32_avx512(input, weight, output, params, stream);
    #elif defined(USE_NEON)
        ret = conv2d_f32_neon(input, weight, output, params, stream);
    #else
        ret = conv2d_f32_ref(input, weight, output, params, stream);
    #endif

    return ret;
}
```

### 算子注册

```c
/* 注册表入口 */
static const operator_registry_t s_registries[] = {
    {"conv2d_f32",  "f32", conv2d_f32,  1, OP_FLAG_NONE},
    {"matmul_f32",  "f32", matmul_f32,  1, OP_FLAG_NONE},
    {"relu_f32",    "f32", relu_f32,    1, OP_FLAG_IN_PLACE},
    {"reshape",     "any", reshape,     1, OP_FLAG_IN_PLACE},
    /* ... */
};
```

## 4. 平台适配规范

### CPU 平台抽象层（PAL）

平台适配遵循 **策略模式**，运行时通过函数指针表切换实现。

```c
/* src/platform/include/platform_ops.h */
typedef struct {
    int   (*init)(void);
    void  (*finalize)(void);
    void* (*alloc_aligned)(size_t size, size_t align);
    void  (*free_aligned)(void* ptr);
    int   (*get_cache_line_size)(void);
    int   (*get_simd_flags)(void);   /* 返回位掩码：SIMD_AVX2 | SIMD_NEON ... */
    int   (*get_core_count)(void);
    int   (*pin_thread)(int cpu_id);
} platform_ops_t;

extern const platform_ops_t* g_platform;
```

```c
/* x86 实现示例 */
static const platform_ops_t s_platform_x86 = {
    .init              = x86_init,
    .finalize          = x86_finalize,
    .alloc_aligned     = x86_alloc_aligned,
    .free_aligned      = x86_free_aligned,
    .get_cache_line_size = x86_get_cache_line_size,
    .get_simd_flags    = x86_get_simd_flags,
    .get_core_count    = x86_get_core_count,
    .pin_thread        = x86_pin_thread,
};
```

### CUDA 平台抽象

```c
/* src/platform/include/cuda_ops.h */
typedef struct {
    int   (*init)(int device_id);
    void  (*finalize)(void);
    void* (*device_alloc)(size_t size);
    void  (*device_free)(void* ptr);
    void* (*host_alloc_pinned)(size_t size);
    void  (*host_free_pinned)(void* ptr);
    int   (*memcpy_h2d)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*memcpy_d2h)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*memcpy_d2d)(void* dst, const void* src, size_t bytes, cudaStream_t stream);
    int   (*stream_create)(cudaStream_t* stream);
    int   (*stream_synchronize)(cudaStream_t stream);
    int   (*stream_destroy)(cudaStream_t stream);
    int   (*get_device_count)(int* count);
    int   (*get_device_props)(cudaDeviceProp* props, int device_id);
} cuda_ops_t;

extern const cuda_ops_t g_cuda;
```

CUDA ops 全局单例，所有算子通过 `g_cuda.memcpy_h2d(...)` 调用，不直接调用 CUDA API。

### 新增平台步骤

1. **CPU 平台**：在 `src/platform/cpu/` 下新建 `platform_<name>/` 目录，实现 `platform_ops_t`，在平台检测宏中注册
2. **CUDA 平台**：CUDA 由 `src/platform/cuda/` 统一管理，扩展时在现有模块中新增 `.cu` 文件
3. 编写 CMake toolchain file 支持交叉编译
4. 运行完整算子测试套件

## 5. 数据类型系统

统一的数据类型定义，贯穿所有层：

```c
typedef enum {
    DATA_TYPE_F32 = 0,
    DATA_TYPE_F16,      /* IEEE 754 half-precision */
    DATA_TYPE_BF16,     /* Brain floating-point */
    DATA_TYPE_I32,
    DATA_TYPE_I8,
    DATA_TYPE_U8,
    DATA_TYPE_COUNT
} data_type_t;

/* 类型特征查询 */
typedef struct {
    const char* name;
    size_t      size;        /* 字节数 */
    int         is_float;    /* 是否为浮点类型 */
    int         is_signed;   /* 是否有符号 */
} data_type_info_t;

const data_type_info_t* data_type_get_info(data_type_t dt);
```

## 6. 内存模型

### 内存层级

```
Register        ← 寄存器（编译器管理）
  └─ L1 Cache   ← 一级缓存（~32KB/core）
       └─ L2 Cache   ← 二级缓存（~256KB/core）
            └─ L3 Cache   ← 三级缓存（共享）
                 └─ DRAM        ← 主存
                      └─ HOST/DEVICE  ← 跨设备内存（如 NPU/GPU）
```

### GPU 内存层级

```
GPU DRAM (Global Memory)   ← 显存（~16-80 GB）
  └─ L2 Cache               ← 二级缓存（共享，~几 MB）
       └─ L1/Shared Memory   ← 一级缓存 / 共享内存（~48-128 KB per SM）
            └─ Register      ← 寄存器（~256 per thread）
```

### 分配策略

- **CPU 小对象（< 1KB）**：栈分配
- **CPU 中等对象（1KB ~ 1MB）**：平台对齐分配（池化）
- **CPU 大对象（> 1MB）**：页对齐分配，支持 NUMA 绑定
- **GPU 显存**：通过 `g_cuda.device_alloc()` 分配，支持显存池复用
- **Pinned Memory**：CPU-GPU 传输用页锁定内存，通过 `g_cuda.host_alloc_pinned()` 分配
- **Unified Memory**：适合不规则访问模式，通过 `cudaMallocManaged` 分配
- **共享内存**：kernel 内部声明 `__shared__`，动态分配通过 kernel launch 参数传递

## 7. 并行模型

```c
/* 并行原语 */
typedef struct {
    int64_t  start;
    int64_t  end;
    int64_t  step;
    int      thread_id;
    int      thread_count;
    void*    args;         /* 用户自定义参数 */
} parallel_for_context_t;

/* 并行执行 for 循环 */
int parallel_for(int64_t start, int64_t end,
                 parallel_func_t func, void* args);

/* 使用示例：并行 ReLU */
int relu_f32_parallel(const tensor_t* input, tensor_t* output) {
    return parallel_for(0, input->numel, relu_f32_kernel, output);
}
```

### CUDA 并行模型

GPU 端的并行通过 CUDA kernel launch 和 Stream 实现。

`.cu` 文件经由 nvcc 编译为 **C++20**。因此 kernel launch 使用 C++ variadic template（自动对实参取 `&` 打包为 `cudaLaunchKernel` 所需的 `void**`），而非 C99 compound literal。

```c
/* src/platform/include/cuda_ops.h — CUDA kernel launch 模板辅助函数 */

/* C++ variadic template：自动对每个 kernel 参数取 & 地址，正确打包为 void** */
template<typename... Args>
static inline void _cuda_kernel_call(const void* kernel, dim3 grid, dim3 block,
                                      size_t shared_mem, cudaStream_t stream,
                                      Args&&... args) {
    void* params[] = { (void*)&args... };
    CUDA_CHECK(cudaLaunchKernel(kernel, grid, block, params, shared_mem, stream));
}

#define CUDA_KERNEL_LAUNCH(kernel, grid, block, shared_mem, stream, ...) \
    _cuda_kernel_call((const void*)(kernel), grid, block, shared_mem, stream, __VA_ARGS__)

/* grid/block 尺寸计算 */
typedef struct {
    dim3 grid_dim;
    dim3 block_dim;
} kernel_config_t;

static inline kernel_config_t cuda_configure_1d(int64_t n, int threads_per_block) {
    kernel_config_t cfg;
    cfg.block_dim = dim3(threads_per_block, 1, 1);
    cfg.grid_dim  = dim3((unsigned int)((n + threads_per_block - 1) / threads_per_block), 1, 1);
    return cfg;
}
```

> **注意**：上方的 template 和 dim3 仅在 `__CUDACC__`（nvcc 编译）时可见。宿主 C 编译器（MSVC）看到的 `cuda_ops.h` 是经过 `#ifndef __CUDACC__` 守卫裁剪的版本，其中 `CUDA_KERNEL_LAUNCH` 为空操作。

### Stream / Event 模型

```c
/* Stream 创建和使用 */
cudaStream_t stream;
CUDA_CHECK(g_cuda.stream_create(&stream));

/* 异步 kernel launch（在同一 Stream 中按序执行）*/
kernel<<<grid, block, 0, stream>>>(...);

/* Host-Device 异步传输 */
CUDA_CHECK(g_cuda.memcpy_h2d(d_data, h_data, bytes, stream));

/* Event 测量时间 */
cudaEvent_t start, stop;
CUDA_CHECK(cudaEventCreate(&start));
CUDA_CHECK(cudaEventCreate(&stop));

CUDA_CHECK(cudaEventRecord(start, stream));
kernel<<<grid, block, 0, stream>>>(...);
CUDA_CHECK(cudaEventRecord(stop, stream));

CUDA_CHECK(cudaEventSynchronize(stop));
float ms;
CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
```

使用原则：

- 默认使用 default stream（`cudaStreamPerThread`），性能关键路径使用独立 Stream 实现 overlap
- 同一个 Stream 内的操作保序，不同 Stream 间不保序
- Host 端同步优先使用 `cudaEventSynchronize` 而非 `cudaStreamSynchronize`（更精细）

## 8. 构建系统

### CMake 结构

```
CMakeLists.txt              # 顶层
cmake/
├── toolchains/
│   ├── x86_64-linux-gnu.cmake
│   ├── aarch64-linux-gnu.cmake
│   └── riscv64-linux-gnu.cmake
├── simd_detect.cmake       # SIMD 指令集自动检测
└── options.cmake           # 编译选项
```

### 构建选项

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `ENABLE_AVX2` | 启用 AVX2 优化 | OFF |
| `ENABLE_AVX512` | 启用 AVX-512 优化 | OFF |
| `ENABLE_NEON` | 启用 ARM NEON 优化 | OFF |
| `ENABLE_SVE` | 启用 ARM SVE 优化 | OFF |
| `ENABLE_OPENMP` | 启用 OpenMP 并行 | ON |
| `ENABLE_CUDA` | 启用 CUDA backend | ON |
| `CUDA_ARCH` | CUDA 架构目标（纯数字，如 `75` `86` `89`，不要加 `sm_` 前缀）| `86` |
| `ENABLE_TESTS` | 构建测试 | ON |
| `ENABLE_COVERAGE` | 启用覆盖率 | OFF |

## 9. 架构违规检查

以下情况视为**架构违规**：

- Operator 层直接调用系统调用或 POSIX API（应通过 Platform 层）
- Application 层绕过 Operator 层直接调用 Platform 层
- 平台相关宏（`__AVX2__`、`__ARM_NEON`）出现在 Platform 层以外的代码中
- 算子的通用 fallback 使用了平台特定 intrinsics
- `.c` 文件直接包含平台相关的私有头文件（应通过 `platform.h`）
