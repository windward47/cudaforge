# 算子性能 Profile 基线

> 基于 `bench_profile.cu` 采集，CUDA event 精确计时。
> 参考 CUDAForge 的 NCU 硬件指标驱动优化思路。

## 环境

| 项 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2050 (sm_86) |
| CUDA | 13.2 |
| 日期 | 2026-06-07 |
| Warmup | 5 iterations |
| Measure | 20 iterations |

## 基线数据

### MatMul

| Shape | Latency (ms) |
| --- | --- |
| 128x128x128 | 0.024 |
| 256x256x256 | 0.142 |
| 512x512x512 | 0.692 |
| 1024x1024x1024 | 5.481 |

### Conv2D

| Shape (NxCxHxWxKxKH) | Latency (ms) |
| --- | --- |
| 1x3x32x32x16x3 | 0.023 |
| 1x3x64x64x32x3 | 0.090 |
| 1x64x128x128x64x3 | 10.322 |
| 1x64x56x56x128x3 | 4.046 |

### Element-wise (1M elements)

| Op | Latency (ms) |
| --- | --- |
| relu_f32 | 0.094 |
| sigmoid_f32 | 0.095 |
| gelu_f32 | 0.096 |
| silu_f32 | 0.091 |
| add_f32 | 0.130 |

### Softmax

| Shape | Latency (ms) |
| --- | --- |
| 32x512 | 0.024 |
| 128x1024 | 0.024 |
| 256x4096 | 0.185 |

### LayerNorm

| Shape | Latency (ms) |
| --- | --- |
| 32x768 | 0.028 |
| 128x768 | 0.022 |
| 128x3072 | 0.069 |

### Reduce

| Shape | Op | Latency (ms) |
| --- | --- | --- |
| 128x1024 | sum | 0.024 |
| 128x1024 | max | 0.025 |
| 256x4096 | sum | 0.052 |

### Pooling

| Shape | Op | Latency (ms) |
| --- | --- | --- |
| 128x128x2x2 | maxpool | 0.029 |
| 128x128x2x2 | avgpool | 0.028 |
| 256x256x3x3 | maxpool | 0.023 |

---

## 回归检测

使用 `scripts/check_perf_regression.sh` 对比当前 profile 与基线：

```bash
# 生成当前 profile
./build/Release/bench_profile.exe > /tmp/current_profile.csv

# 对比基线（偏差 >20% 告警）
./scripts/check_perf_regression.sh /tmp/current_profile.csv docs/PROFILE_BASELINE.csv
```

## 更新基线

当优化算子后，重新运行 benchmark 更新基线：

```bash
./build/Release/bench_profile.exe > docs/PROFILE_BASELINE.csv
```
