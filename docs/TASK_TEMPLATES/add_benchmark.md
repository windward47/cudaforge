# 为算子添加 Benchmark

## 背景

为 XXX 算子添加性能测试，对比 CPU fallback 和 CUDA kernel 的耗时，以及不同 shape 下的性能表现。

## 步骤

### 1. 创建 benchmark 文件

```text
tests/benchmark/
└── bench_xxx.c      # benchmark 源码
```

### 2. 测试维度设计

| 维度 | 取值 |
| --- | --- |
| 输入大小 | 小（1K）、中（1M）、大（10M~100M） |
| 数据类型 | f32、f16（如适用） |
| 特殊形状 | 非对齐大小、极端 aspect ratio |

### 3. benchmark 模板

```c
static void bench_xxx_f32(benchmark_t* b) {
    /* 准备数据 */
    tensor_t* input  = tensor_rand(DATA_TYPE_F32, 1, &b->size);
    tensor_t* output = tensor_like(input);

    /* Warmup */
    relu_f32_cuda(input, output, NULL);

    /* 计时 */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        relu_f32_cuda(input, output, NULL);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("avg: %.3f ms\n", ms / iters);

    /* 清理... */
}
```

### 4. 输出格式

```
Operator    | Shape     | CPU (ms) | CUDA (ms) | Speedup
------------|-----------|----------|-----------|--------
relu_f32    | 4x256     | 0.012    | 0.008     | 1.5x
relu_f32    | 128x1024  | 0.89     | 0.034     | 26.2x
relu_f32    | 1024x4096 | 28.4     | 0.52      | 54.6x
```
