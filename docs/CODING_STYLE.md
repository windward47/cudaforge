# 编码规范

## 1. 通用原则

- **可移植性优先**：算子代码需在多硬件平台（x86、ARM、RISC-V 等）和编译器（GCC、Clang、MSVC）下正确编译运行
- **性能意识**：算子核心路径需考虑缓存友好、避免不必要的内存分配、减少分支预测失败
- **最小依赖**：核心算子层不依赖操作系统或标准库以外的第三方库
- **防御性编程**：对输入指针做非空检查、对维度/形状做合法性校验

## 2. 语言标准

- `.c` 文件使用 **C11** 标准（`CMAKE_C_STANDARD=11`），由 MSVC/GCC/Clang 编译
- `.cu` 文件由 nvcc 编译为 **C++20**（`CMAKE_CUDA_STANDARD=20`）
- 不能在 `.cu` 中使用 C99 语法，如 compound literal `(const void*[]){...}` — 用 C++ variadic template 替代
- 不能在 `.cu` 中使用 C 风格的 designated initializer（如 `.init = fn`），用 C++20 聚合初始化或构造函数替代
- 平台适配的特殊代码可使用编译器扩展，但必须用宏隔离

```c
/* 平台特定内联函数 */
#ifdef __GNUC__
  #define FORCE_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER)
  #define FORCE_INLINE __forceinline
#else
  #define FORCE_INLINE inline
#endif
```

## 3. 命名规范

| 类别 | 规范 | 示例 |
| --- | --- | --- |
| 函数 | `snake_case` | `tensor_alloc()`, `matmul_f32()` |
| 类型/结构体 | 小写 + `_t` 后缀 | `tensor_t`, `shape_t` |
| 宏/常量 | `UPPER_SNAKE_CASE` | `MAX_TENSOR_DIMS`, `TENSOR_ALIGN` |
| 枚举 | 类型前缀 + `UPPER_SNAKE_CASE` | `DATA_TYPE_F32`, `DATA_TYPE_I32` |
| 全局变量 | `g_` 前缀 + `snake_case` | `g_default_allocator` |
| 静态变量 | `s_` 前缀 + `snake_case` | `s_ref_count` |
| 文件 | `snake_case` | `tensor_ops.c`, `platform_linux.c` |

### 命名禁忌

- 避免通用缩写（`tbl` → `table`, `calc` → `compute`），数学运算名除外（`add`, `mul`, `norm`）
- 类型名和函数名避免与 POSIX 或 C 标准库冲突（不要自定义 `size_t`, `int32_t`）
- 指针变量建议加 `_ptr` 后缀或 `p_` 前缀：`data_ptr`, `p_buffer`

## 4. 代码组织

### 文件结构

```
模块/
├── include/
│   └── module_name.h      # 公开 API 头文件
├── src/
│   ├── module_name.c      # CPU fallback 实现（C11）
│   ├── module_name_cuda.cu # CUDA kernel 实现（C++20）
│   ├── module_name_int.h  # 算子参数结构体（.c 和 .cu 共用）
│   └── platform/
│       ├── module_name_x86.c    # x86 平台实现
│       └── module_name_arm.c    # ARM 平台实现
└── test/
    └── test_module_name.c
```

> **`*_int.h` 模式**：算子专用参数类型（如 `conv_params_t`）定义在 `*_int.h` 中，供 `.c`（CPU）和 `.cu`（CUDA）共用，避免跨编译单元的类型定义重复。

### 头文件规范

- 所有头文件必须有 `#pragma once` 或头文件保护宏
- 头文件保护宏格式：`<MODULE>_<FILE>_H_`

```c
#pragma once

/* 或者传统方式 */
#ifndef TENSOR_TENSOR_OPS_H_
#define TENSOR_TENSOR_OPS_H_
/* ... */
#endif
```

- 头文件中的函数声明必须有完整参数名和注释
- `.c` 文件内部使用的函数声明为 `static`

### 函数规范

- 一个函数不超过 **60 行**（算子核心循环不计入）
- 参数不超过 **6 个**（超过用结构体传参）
- 每个函数返回错误码（成功返回 `0`，失败返回负值错误码）

```c
typedef struct {
    const float* input;
    float* output;
    int64_t numel;
    int32_t ndim;
    const int64_t* shape;
} tensor_reshape_params_t;

int tensor_reshape(const tensor_reshape_params_t* params);
```

## 5. 错误处理

- 函数返回 `int` 错误码，`0` 表示成功，负值表示具体错误类型
- 不使用 `setjmp`/`longjmp` 做异常流控制
- 资源分配采用 **goto cleanup** 模式集中释放

```c
int tensor_process(tensor_t* t) {
    void* buf = NULL;
    FILE* fp = NULL;
    int ret = 0;

    buf = malloc(t->size);
    if (!buf) {
        ret = -ENOMEM;
        goto cleanup;
    }

    fp = fopen(t->path, "rb");
    if (!fp) {
        ret = -ENOENT;
        goto cleanup;
    }

    /* 处理逻辑... */

cleanup:
    free(buf);
    if (fp) fclose(fp);
    return ret;
}
```

- 指针参数做非空断言（使用 `assert()` 用于调试，显式 `if` 用于发布）

```c
int tensor_get_shape(const tensor_t* t, int64_t* dims) {
    if (t == NULL || dims == NULL) return -EINVAL;
    /* ... */
}
```

## 6. 内存管理

- 谁分配谁释放，函数文档中标注返回指针的释放责任
- 对齐分配使用 `aligned_alloc` / `posix_memalign`（封装在平台适配层）
- 避免碎片化：频繁分配/释放的场景使用内存池或栈上分配

```c
/* 内存分配器接口 */
typedef struct {
    void* (*alloc)(size_t size, size_t alignment);
    void  (*free)(void* ptr);
} allocator_t;
```

## 7. 平台适配规范

- 平台差异代码通过 **预处理宏** 隔离，放在 `src/platform/` 目录下
- 统一提供 `platform.h` 抽象层，暴露平台无关接口

```c
/* include/platform.h */
#ifndef PLATFORM_H_
#define PLATFORM_H_

/* 平台检测 */
#if defined(__x86_64__) || defined(__i386__)
  #define PLATFORM_X86 1
#elif defined(__aarch64__) || defined(__arm__)
  #define PLATFORM_ARM 1
#elif defined(__riscv)
  #define PLATFORM_RISCV 1
#endif

/* 统一接口：各平台在 src/platform/ 下分别实现 */
int platform_get_cache_line_size(void);
int platform_get_core_count(void);
void* platform_alloc_aligned(size_t size, size_t alignment);

#endif /* PLATFORM_H_ */
```

## 8. 编译与构建

- 警告级别：GCC/Clang 使用 `-Wall -Wextra -Wpedantic`，MSVC 使用 `/W4 /WX`，警告视为错误
- 统一构建系统：CMake，支持跨平台交叉编译
- **关键：MSVC 编译选项必须通过 `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` 设置**，不能用 `add_compile_options()` — 后者会将 C 编译选项泄漏到 nvcc，导致 CUDA kernel 编译失败
- CUDA 专用选项通过 `CMAKE_CUDA_FLAGS` 或 `target_compile_options` 配合 `$<COMPILE_LANGUAGE:CUDA>` generator expression 设置
- `CUDA_ARCH` 格式为纯数字（如 `86`），不要加 `sm_` 前缀，CMake 会自动生成正确的 `compute_86` / `sm_86`

```cmake
# CMakeLists.txt 示例
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 正确：仅在 C/CXX 上启用警告
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   /W4 /WX")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")

# 正确：CUDA 架构目标
set(CMAKE_CUDA_ARCHITECTURES 86)

option(ENABLE_AVX2 "Enable AVX2 optimizations" OFF)
if(ENABLE_AVX2)
  target_compile_definitions(operator PRIVATE USE_AVX2)
  target_compile_options(operator PRIVATE -mavx2 -mfma)
endif()
```

## 9. 注释规范

- 公开 API 使用 Doxygen 风格

```c
/**
 * @brief 对张量执行 ReLU 激活函数（逐元素）
 * @param input  输入张量指针，不能为 NULL
 * @param output 输出张量指针，不能为 NULL
 * @param size   元素数量
 * @return 成功返回 0，失败返回负值错误码
 */
int relu_f32(const tensor_t* input, tensor_t* output, int64_t size);
```

- 内部函数用简单注释说明意图
- 性能关键路径标注优化原因：`/* 循环展开 4 路以利用 SIMD */`
- 禁止大段注释掉的死代码，改用 git 历史追溯

## 10. 测试规范

- 使用 C 单元测试框架（Unity/CMocka/Criterion）
- 每个算子的测试覆盖：
  - 正常输入（含边界形状：标量、0 维、大维度）
  - 异常输入（NULL 指针、形状不匹配、越界索引）
  - 平台特性分支（AVX2 启用/禁用、NEON 启用/禁用）

```c
/* 测试用例示例 */
static void test_relu_f32_basic(void) {
    tensor_t* t = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){2, 3});
    /* 初始化输入值... */
    int ret = relu_f32(t, t, 6);
    assert_int_equal(ret, 0);
    /* 验证输出... */
    tensor_destroy(t);
}
```

## 11. Git 提交规范

### 提交信息格式

```
<type>(<scope>): <简短描述>

<详细说明（可选）>
```

### type 类型

| type | 含义 |
| --- | --- |
| feat | 新增算子或功能 |
| fix | 修复 bug |
| perf | 性能优化 |
| port | 平台适配/移植 |
| refactor | 重构 |
| test | 测试相关 |
| docs | 文档 |
| chore | 构建/CI/工具链 |

### 提交原则

- 每个提交只做一件事
- 提交信息用中文描述**做了什么**和**为什么**
- 不提交有编译警告或测试失败的代码
