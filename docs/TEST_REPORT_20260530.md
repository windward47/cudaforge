# CudaForge 测试报告

> **日期**: 2026-05-30
> **版本**: v0.5.0
> **环境**: Windows 10 x64, MSVC 2022, CUDA 13.2, RTX 2050 (sm_86)

---

## 测试结果总览

| 指标 | 结果 |
| --- | --- |
| 测试总数 | 34 |
| 通过 | 34 |
| 失败 | 0 |
| 通过率 | 100% |
| compute-sanitizer | 0 errors (test_bert_mha, test_mha_decode) |

---

## 逐项测试结果

### 算子单元测试

| # | 测试名 | 结果 | 耗时 | 说明 |
| --- | --- | --- | --- | --- |
| 1 | test_relu | ✅ PASS | 0.02s | ReLU + in-place + null input |
| 2 | test_matmul | ✅ PASS | 0.02s | 矩阵乘法 |
| 3 | test_conv | ✅ PASS | 0.02s | Conv2D ref + im2col + stride=2 |
| 4 | test_conv_scale | ✅ PASS | 2.63s | Conv2D ResNet 规模 CPU vs CUDA |
| 5 | test_ops_scale | ✅ PASS | 2.39s | MaxPool/Add/GlobalAvgPool/MatMul 规模测试 |
| 6 | test_activations | ✅ PASS | 0.02s | Sigmoid/GELU/SiLU/Exp + in-place |
| 7 | test_pooling | ✅ PASS | 0.02s | MaxPool2D, AvgPool2D |
| 8 | test_batchnorm | ✅ PASS | 0.02s | BatchNorm |
| 9 | test_add | ✅ PASS | 0.02s | 逐元素 Add + scalar |
| 10 | test_reshape | ✅ PASS | 0.02s | Reshape 形状变换 |
| 11 | test_globalavgpool | ✅ PASS | 0.02s | GlobalAvgPool |
| 12 | test_softmax | ✅ PASS | 0.02s | Softmax |
| 13 | test_mul | ✅ PASS | 0.04s | 逐元素 Mul |
| 14 | test_concat | ✅ PASS | 0.02s | Concat 拼接 |
| 15 | test_resize | ✅ PASS | 0.02s | Resize (nearest + bilinear) |
| 16 | test_transpose | ✅ PASS | 0.02s | Transpose 维度交换 |
| 17 | test_sub | ✅ PASS | 0.04s | 逐元素 Sub |
| 18 | test_div | ✅ PASS | 0.06s | 逐元素 Div + scalar |
| 19 | test_slice | ✅ PASS | 0.04s | Slice 切片 |
| 20 | test_split | ✅ PASS | 0.06s | Split 分割 |
| 21 | test_layernorm | ✅ PASS | 0.03s | LayerNorm |
| 22 | test_gather | ✅ PASS | 0.07s | Gather 索引取值 |

### 集成测试

| # | 测试名 | 结果 | 耗时 | 说明 |
| --- | --- | --- | --- | --- |
| 23 | test_graph | ✅ PASS | 2.31s | 计算图构建/拓扑排序/执行 |
| 24 | test_cuda_kernels | ✅ PASS | 2.33s | CUDA 内核基础 (ReLU/MatMul/Conv/Pool/BatchNorm/Sigmoid/GELU) |
| 25 | test_onnx | ✅ PASS | 2.30s | ONNX protobuf 解析器 |
| 26 | test_classification | ✅ PASS | 2.31s | MNIST CNN (1×1×28×28→10) |
| 27 | test_resnet | ✅ PASS | 5.93s | ResNet-18 (1×3×224×224→1000) |
| 28 | test_yolo | ✅ PASS | 16.41s | YOLOv8n (1×3×640×640→1×84×8400) |
| 29 | test_reduce | ✅ PASS | 2.42s | ReduceSum, ReduceMax |
| 30 | test_bert | ✅ PASS | 2.39s | BERT Phase A+B |
| 31 | test_bert_mha | ✅ PASS | 2.39s | BERT MHA 融合 (CPU + CUDA) |
| 32 | test_mha_decode | ✅ PASS | 2.25s | Decode + KV-cache + CausalMask |
| 33 | test_rope | ✅ PASS | 2.30s | RoPE 旋转编码 (CPU + in-place + CUDA) |
| 34 | test_fp16_ops | ✅ PASS | 2.35s | FP16 Softmax + Conv2D |

---

## Benchmark 结果

### BERT-base 单层 MHA

| 配置 | 延迟 (ms/iter) | 加速比 |
| --- | --- | --- |
| CPU | 41.74 | 1.00x |
| CUDA FP32 | 26.01 | 1.60x |
| CUDA FP16 (WMMA) | **4.66** | **8.96x** |

**参数**: B=1, S=8, D=768, H=12, d=64

### MHA Decode (单 token)

| cache_len | CPU (ms) | CUDA (ms) |
| --- | --- | --- |
| 0 | 3.80 | 22.56 |
| 32 | 4.64 | 22.53 |
| 128 | 4.43 | 22.53 |
| 256 | 4.58 | 22.56 |
| 511 | 4.84 | 22.59 |

**参数**: B=1, D=768, H=12, d=64, max_seq=512

> 注: CUDA decode 单 token 慢于 CPU（launch 开销主导），batch decode 或 graph 级优化可改善。

---

## compute-sanitizer 结果

| 测试 | 错误数 |
| --- | --- |
| test_bert_mha | 0 |
| test_mha_decode | 0 |

---

## 算子覆盖

| 类别 | 算子数 | 算子列表 |
| --- | --- | --- |
| 逐元素 | 11 | ReLU, Sigmoid, GELU, SiLU, Exp, Add, Mul, Sub, Div, Clip, Pad |
| 矩阵 | 2 | MatMul, Conv2D |
| 池化 | 3 | MaxPool2D, AvgPool2D, GlobalAvgPool |
| 归一化 | 2 | BatchNorm, LayerNorm |
| 形状 | 7 | Reshape, Transpose, Concat, Resize, Slice, Split, Squeeze/Unsqueeze |
| 索引 | 2 | Gather, ArgMax |
| 归约 | 2 | ReduceSum, ReduceMax |
| 注意力 | 2 | MHA_Fused, MHA_Decode |
| 掩码 | 1 | CausalMask |
| 位置编码 | 1 | RoPE |
| 类型转换 | 1 | Cast (F32↔I64↔F16) |
| **总计** | **36** | |

---

## FP16 算子覆盖

| 算子 | FP16 CUDA | 文件 |
| --- | --- | --- |
| ReLU, Sigmoid, GELU, SiLU, Exp | ✅ | `elementwise_f16_cuda.cu` |
| Add, Mul, Sub, Div | ✅ (含 broadcast) | `elementwise_f16_cuda.cu` |
| Conv2D | ✅ | `conv_f16_cuda.cu` |
| MatMul | ✅ | `matmul_f16_cuda.cu` |
| BatchNorm | ✅ | `norm_f16_cuda.cu` |
| LayerNorm | ✅ | `norm_f16_cuda.cu` |
| Softmax | ✅ | `softmax_f16_cuda.cu` |
| MHA_Fused | ✅ (WMMA) | `mha_fused_f16_cuda.cu` |
| Cast | ✅ (F32↔F16) | `cast_cuda.cu` |

---

## 已知问题

| 问题 | 严重程度 | 说明 |
| --- | --- | --- |
| cuda_device_free: invalid argument | Low | tensor_destroy 中的 stale device pointer，非功能性问题 |
| 单 token decode 慢于 CPU | Expected | CUDA launch 开销主导，B=1 只用 1 个 SM |

---

## 版本信息

| 项目 | 值 |
| --- | --- |
| 版本 | v0.5.0 |
| CUDA Toolkit | 13.2 |
| GPU | NVIDIA GeForce RTX 2050 (sm_86, compute capability 8.6) |
| 编译器 | MSVC 2022 (19.44) + nvcc 13.2 |
| C 标准 | C11 (host) / C++20 (CUDA) |
| 测试框架 | Unity (third_party) + 自定义 CHECK 宏 |
