# CudaForge 测试文档

> 基于 2026-05-30 代码库状态。共 30 个测试 + 3 个 benchmark。

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

### 1. 单元测试（算子级）

| 测试名 | 文件 | 测试内容 | 验证方式 |
|--------|------|---------|---------|
| test_relu | test_onnx.c | ReLU 单层 + 链式 | CPU 参考 + CUDA vs CPU |
| test_matmul | test_cuda_kernels.cu | MatMul 2×3 * 3×2 | CUDA 数值验证 |
| test_conv | test_onnx.c + test_cuda_kernels.cu | Conv2D + ReLU 融合 | CPU + CUDA vs CPU |
| test_pooling | test_cuda_kernels.cu | MaxPool2D, AvgPool2D | CUDA 数值验证 |
| test_batchnorm | test_cuda_kernels.cu | BatchNorm | CUDA 数值验证 |
| test_add | test_ops_scale.c | 逐元素 Add (ResNet 规模) | CPU vs CUDA (1e-4) |
| test_mul | test_cuda_kernels.cu | 逐元素 Mul | CUDA 数值验证 |
| test_sub | test_reduce.c | 逐元素 Sub | CPU + CUDA |
| test_div | test_reduce.c | 逐元素 Div | CPU + CUDA |
| test_softmax | test_cuda_kernels.cu | Softmax | CUDA 数值验证 |
| test_reshape | test_onnx.c | Reshape 形状变换 | CPU 验证 |
| test_transpose | test_onnx.c | Transpose 维度交换 | CPU 验证 |
| test_concat | test_onnx.c | Concat 拼接 | CPU 验证 |
| test_resize | test_onnx.c | Resize 双线性插值 | CPU 验证 |
| test_slice | test_onnx.c | Slice 切片 | CPU 验证 |
| test_split | test_onnx.c | Split 分割 | CPU 验证 |
| test_gather | test_onnx.c | Gather 索引取值 | CPU 验证 |
| test_globalavgpool | test_ops_scale.c | GlobalAvgPool (512×7×7) | CPU vs CUDA (1e-4) |
| test_layernorm | test_bert.c | LayerNorm (BERT 维度) | CPU vs PyTorch 参考 |
| test_reduce | test_reduce.c | ReduceSum, ReduceMax | CPU 精确值 + CUDA (1e-3) |
| test_activations | test_cuda_kernels.cu | Sigmoid, GELU (含 in-place) | CUDA 数值验证 |

### 2. 集成测试（端到端模型）

| 测试名 | 文件 | 模型 | 输入 | 验证方式 |
|--------|------|------|------|---------|
| test_classification | test_classification.c | MNIST CNN | 1×1×28×28 | CPU vs PyTorch (1e-3) |
| test_resnet | test_resnet.c | ResNet-18 | 1×3×224×224 | CPU+CUDA vs PyTorch (1e-2), Top-5 |
| test_yolo | test_yolo.c | YOLOv8n | 1×3×640×640 | CPU+CUDA vs PyTorch (1e-2), 计时 |
| test_bert | test_bert.c | BERT Phase A+B | 1×8 | CPU+CUDA vs PyTorch (1e-3/5e-3) |
| test_bert_mha | test_bert_mha.c | BERT MHA (手动构建图) | B=1,S=4,H=2,d=8 | CPU vs unfused ref + CUDA vs CPU (1e-3) |
| test_mha_decode | test_mha_decode.c | Decode + KV-cache | B=1,S=4,H=2,d=4 | CPU 两步验证 + CUDA vs CPU (1e-3) + CausalMask |

### 3. 框架测试

| 测试名 | 文件 | 测试内容 |
|--------|------|---------|
| test_onnx | test_onnx.c | ONNX protobuf 解析器（varint、嵌套消息、packed float、边界情况） |
| test_graph | test_onnx.c | 计算图构建、拓扑排序、执行 |
| test_cuda_kernels | test_cuda_kernels.cu | CUDA 内核基础功能（ReLU/MatMul/Conv/Pool/BatchNorm/Sigmoid/GELU） |
| test_ops_scale | test_ops_scale.c | ResNet 规模算子 CPU vs CUDA 对比 |
| test_conv_scale | test_conv_scale.c | Conv2D ResNet layer1.0 规模 CPU vs CUDA |

### 4. Benchmark

| 名称 | 文件 | 测试内容 | 指标 |
|------|------|---------|------|
| bench_bert_mha | bench_bert_mha.c | BERT-base MHA 融合 kernel | CPU/CUDA/FP16 延迟 (ms/iter) |
| bench_mha_decode | bench_mha_decode.c | 单 token decode + KV-cache | CPU/CUDA 在 cache_len=0/32/128/256/511 的延迟 |
| bench_operators | bench_operators.cu | 核心算子性能 | MatMul/Conv/激活/Pooling/BatchNorm CPU vs CUDA |

---

## 测试模型生成脚本

| 脚本 | 生成模型 | 用途 |
|------|---------|------|
| gen_classification_test.py | MNIST CNN (1×1×28×28→10) | test_classification |
| gen_resnet_test.py | ResNet-18 (1×3×224×224→1000) | test_resnet |
| gen_yolo_test.py | YOLOv8n (1×3×640×640→1×84×8400) | test_yolo |
| gen_bert_test.py | BERT Phase A+B (1×8→1×8×256 / 1×257) | test_bert |
| gen_bert_mha_test.py | BERT MHA 单层 (1×8×768) | test_bert_mha |
| gen_gpt2_test.py | GPT-2 小型 (hidden=64, 2层) | GPT-2 测试模型 |
| gen_onnx_test.py | ONNX 通用测试模型 | test_onnx |

每个脚本执行流程：
1. 定义 PyTorch `nn.Module`
2. `torch.onnx.export()` 导出 ONNX
3. ONNX Runtime 生成参考输出
4. 写入 `.bin` 文件（raw f32）

---

## 测试覆盖的算子

| 类别 | 算子 |
|------|------|
| 逐元素 | ReLU, Sigmoid, GELU, SiLU, Exp, Add, Mul, Sub, Div |
| 矩阵 | MatMul, Conv2D |
| 池化 | MaxPool2D, AvgPool2D, GlobalAvgPool |
| 归一化 | BatchNorm, LayerNorm |
| 形状 | Reshape, Transpose, Concat, Resize, Slice, Split, Squeeze, Unsqueeze |
| 索引 | Gather, ArgMax |
| 归约 | ReduceSum, ReduceMax |
| 注意力 | MHA_Fused (融合), MHA_Decode (KV-cache + GQA) |
| 掩码 | CausalMask |
| 位置编码 | RoPE |
| 类型转换 | Cast (F32↔I64, F32↔F16) |
| FP16 | ReLU, Sigmoid, GELU, SiLU, Exp, Add, Mul, Sub, Div, MHA_Fused |

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
| test_conv 警告即错误 | 已知 | MSVC 将 C4100/C4189 视为错误，需 /wd4100 /wd4189 |
| cuda_device_free: invalid argument | 已知 | tensor_destroy 双释放或 stale device pointer，非测试相关 |
| 单 token decode 慢于 CPU | 预期 | CUDA launch 开销主导，B=1 只用 1 个 SM |
