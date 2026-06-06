# 新增 CUDA 算子

## 背景

为 inference engine 添加一个 XXX 算子，需要同时提供 CUDA kernel 和 CPU fallback。

## 步骤

### 1. 创建文件

```text
src/operator/nn/
├── xxx.c              # CPU fallback（纯 C）
├── xxx_cuda.cu        # CUDA kernel
├── xxx_int.h          # 内部声明（shape check、常量等）
└── test/
    └── test_xxx.c     # 单元测试
```

### 2. CPU fallback 实现

在 `xxx.c` 中实现：
- 参数校验（NULL 检查 + shape 校验）
- 纯 C 实现（不做平台优化）
- 返回错误码

### 3. CUDA kernel 实现

在 `xxx_cuda.cu` 中实现：
- global kernel 函数（naive 版本）
- launch wrapper：网格/block 配置、error check
- 通过 `g_cuda` 接口访问 GPU 资源，不直接调用 CUDA API

### 4. 算子注册（X-macro 模式）

在 `src/operator/operator_registry.def` 中添加注册条目：

```c
/* CPU 算子 */
REGISTER_CPU(register_xxx_f32)

/* CUDA 算子 */
REGISTER_CUDA(register_xxx_f32_cuda)
```

- **不要修改 `operator_init.c`** — 宏自动展开生成前向声明和初始化调用
- 在 `xxx.c` 中实现 `register_xxx_f32()`，在 `xxx_cuda.cu` 中实现 `extern "C" register_xxx_f32_cuda()`
- 验证 `flags` 是否正确标注（是否 in-place、是否支持 alias）
- 运行 `scripts/check_registry.sh` 验证 .def 与源文件一致

### 5. 测试

- 正常输入测试（含边界形状）
- 异常输入测试（NULL、shape 不匹配）
- CPU vs GPU allclose 对比
- `compute-sanitizer` 通过（CUDA 12.0+ 替代 cuda-memcheck）

### 6. 构建验证

```bash
cmake -B build -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DCUDA_ARCH=86
cmake --build build -j$(nproc)
ctest --test-dir build -R test_xxx -V
compute-sanitizer ./build/Release/test_xxx.exe
```
