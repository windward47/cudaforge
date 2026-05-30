# CudaForge 测试文档

> 基于 2026-05-30 代码库状态。共 31 个测试 + 3 个 benchmark。

---

## 运行方式

```bash
# 构建所有测试
cmake -B build -G "Visual Studio 17 2022" -A x64 -DENABLE_CUDA=ON -DENABLE_TESTS=ON -DCUDA_ARCH=86
cmake --build build --config Release -j$(nproc)

# 运行全部测试
ctest --test-dir build -C Release -j$(nproc)

# 运行单个测试
./build/Release/test_bert_mha.exe

# GPU 内存检查
compute-sanitizer ./build/Release/test_bert_mha.exe

# 运行 benchmark
./build/Release/bench_bert_mha.exe
```

---

## 测试分类

### 1. 算子单元测试（`src/operator/nn/test/`）

每个算子有独立的单元测试文件，验证 CPU 参考实现和 CUDA kernel 的正确性。

| 测试名 | 源文件 | 测试内容 | 验证方式 |
|--------|------|---------|---------|
| test_relu | `nn/test/test_relu.c` | ReLU 激活 | CPU 精确值 + CUDA |
| test_matmul | `blas/test/test_matmul.c` | 矩阵乘法 | CPU 参考 + CUDA |
| test_conv | `nn/test/test_conv.c` | Conv2D | CPU + CUDA vs CPU |
| test_pooling | `nn/test/test_pooling.c` | MaxPool2D, AvgPool2D | CPU + CUDA |
| test_batchnorm | `nn/test/test_batchnorm.c` | BatchNorm | CPU + CUDA |
| test_add | `nn/test/test_add.c` | 逐元素 Add | CPU + CUDA |
| test_mul | `nn/test/test_mul.c` | 逐元素 Mul | CPU + CUDA |
| test_sub | `nn/test/test_sub.c` | 逐元素 Sub | CPU + CUDA |
| test_div | `nn/test/test_div.c` | 逐元素 Div | CPU + CUDA |
| test_reshape | `nn/test/test_reshape.c` | Reshape 形状变换 | CPU 验证 |
| test_transpose | `nn/test/test_transpose.c` | Transpose 维度交换 | CPU + CUDA |
| test_concat | `nn/test/test_concat.c` | Concat 拼接 | CPU + CUDA |
| test_resize | `nn/test/test_resize.c` | Resize (nearest + bilinear) | CPU + CUDA |
| test_slice | `nn/test/test_slice.c` | Slice 切片 | CPU + CUDA |
| test_split | `nn/test/test_split.c` | Split 分割 | CPU + CUDA |
| test_softmax | `nn/test/test_softmax.c` | Softmax | CPU + CUDA |
| test_globalavgpool | `nn/test/test_globalavgpool.c` | GlobalAvgPool | CPU + CUDA |
| test_layernorm | `nn/test/test_layernorm.c` | LayerNorm | CPU + CUDA |
| test_gather | `nn/test/test_gather.c` | Gather 索引取值 | CPU + CUDA |
| test_activations | `nn/test/test_activations.c` | Sigmoid, GELU, SiLU, Exp | CPU + CUDA |

### 2. 集成测试（`tests/`）

端到端模型推理测试，验证完整推理链路。

| 测试名 | 源文件 | 模型 | 输入 | 验证方式 |
|--------|------|------|------|---------|
| test_classification | `tests/test_classification.c` | MNIST CNN | 1×1×28×28 | CPU vs PyTorch (1e-3) |
| test_resnet | `tests/test_resnet.c` | ResNet-18 | 1×3×224×224 | CPU+CUDA vs PyTorch (1e-2) |
| test_yolo | `tests/test_yolo.c` | YOLOv8n | 1×3×640×640 | CPU+CUDA vs PyTorch (1e-2) |
| test_bert | `tests/test_bert.c` | BERT Phase A+B | 1×8 | CPU+CUDA vs PyTorch (1e-3) |
| test_bert_mha | `tests/test_bert_mha.c` | BERT MHA 融合 | B=1,S=4,H=2,d=8 | CPU vs ref + CUDA vs CPU |
| test_mha_decode | `tests/test_mha_decode.c` | Decode + KV-cache | B=1,S=4,H=2,d=4 | CPU 两步 + CUDA vs CPU |
| test_rope | `tests/test_rope.c` | RoPE 旋转编码 | S=4,H=2,d=4 | CPU ref + in-place + CUDA |

### 3. 框架测试（`tests/`）

测试 ONNX 解析器、计算图、CUDA 内核基础功能。

| 测试名 | 源文件 | 测试内容 |
|--------|------|---------|
| test_onnx | `tests/test_onnx.c` | ONNX protobuf 解析器（varint、嵌套消息、packed float） |
| test_graph | `tests/test_onnx.c` | 计算图构建、拓扑排序、执行 |
| test_reduce | `tests/test_reduce.c` | ReduceSum, ReduceMax 精确值验证 |
| test_cuda_kernels | `tests/test_cuda_kernels.cu` | CUDA 内核基础（ReLU/MatMul/Conv/Pool/BatchNorm/Sigmoid/GELU） |
| test_ops_scale | `tests/test_ops_scale.c` | ResNet 规模算子 CPU vs CUDA |
| test_conv_scale | `tests/test_conv_scale.c` | Conv2D ResNet 规模 CPU vs CUDA |

### 4. Benchmark（`tests/`）

| 名称 | 源文件 | 测试内容 | 指标 |
|------|------|---------|------|
| bench_bert_mha | `tests/bench_bert_mha.c` | BERT-base MHA 融合 kernel | CPU/CUDA/FP16 延迟 (ms/iter) |
| bench_mha_decode | `tests/bench_mha_decode.c` | 单 token decode + KV-cache | CPU/CUDA 在不同 cache_len 的延迟 |
| bench_operators | `tests/bench_operators.cu` | 核心算子性能 | MatMul/Conv/激活/Pooling/BatchNorm |

---

## 测试模型生成脚本

| 脚本 | 生成模型 | 用途 |
|------|---------|------|
| `tests/gen_classification_test.py` | MNIST CNN (1×1×28×28→10) | test_classification |
| `tests/gen_resnet_test.py` | ResNet-18 (1×3×224×224→1000) | test_resnet |
| `tests/gen_yolo_test.py` | YOLOv8n (1×3×640×640→1×84×8400) | test_yolo |
| `tests/gen_bert_test.py` | BERT Phase A+B (1×8) | test_bert |
| `tests/gen_bert_mha_test.py` | BERT MHA 单层 (1×8×768) | test_bert_mha |
| `tests/gen_bert_base_test.py` | BERT-base 完整 | test_bert |
| `tests/gen_gpt2_test.py` | GPT-2 小型 (hidden=64, 2层) | GPT-2 测试模型 |
| `tests/gen_onnx_test.py` | ONNX 通用测试模型 | test_onnx |

---

## 测试覆盖的算子

| 类别 | 算子 |
|------|------|
| 逐元素 | ReLU, Sigmoid, GELU, SiLU, Exp, Add, Mul, Sub, Div, Clip |
| 矩阵 | MatMul, Conv2D, Pad |
| 池化 | MaxPool2D, AvgPool2D, GlobalAvgPool |
| 归一化 | BatchNorm, LayerNorm |
| 形状 | Reshape, Transpose, Concat, Resize, Slice, Split, Squeeze, Unsqueeze |
| 索引 | Gather, ArgMax |
| 归约 | ReduceSum, ReduceMax |
| 注意力 | MHA_Fused (融合), MHA_Decode (KV-cache + GQA) |
| 掩码 | CausalMask |
| 位置编码 | RoPE |
| 类型转换 | Cast (F32↔I64, F32↔F16) |
| FP16 | ReLU, Sigmoid, GELU, SiLU, Exp, Add, Mul, Sub, Div, Conv2D, MatMul, BatchNorm, LayerNorm, Softmax, MHA_Fused |

---

## 精度阈值

| 验证类型 | 阈值 | 说明 |
|---------|------|------|
| CPU vs PyTorch 参考 | 1e-3 ~ 1e-6 | 取决于模型复杂度 |
| CUDA vs CPU | 1e-3 ~ 1e-6 | FP32 浮点舍入差异 |
| FP16 vs FP32 | 1e-2 | FP16 精度损失 |
| 相对误差 | 1e-6 ~ 1e-4 | 用于大数值范围场景 |

---

## 已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| test_conv 警告即错误 | 已知 | MSVC 将 C4100/C4189 视为错误 |
| cuda_device_free: invalid argument | 已知 | tensor_destroy 双释放或 stale device pointer |
| 单 token decode 慢于 CPU | 预期 | CUDA launch 开销主导，B=1 只用 1 个 SM |
