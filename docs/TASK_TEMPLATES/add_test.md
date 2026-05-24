# 为算子添加单元测试

## 背景

为 XXX 算子编写完整的单元测试。

## 测试覆盖要求

### 必测场景

| 类别 | 用例 | 说明 |
| --- | --- | --- |
| 正常 | 小尺寸（1x1） | 边界：最小有效输入 |
| 正常 | 典型尺寸（4x256） | 常规推理 shape |
| 正常 | 大尺寸（1024x4096） | 验证大 tensor 无 overflow |
| 正常 | 标量（1） | 单元素 tensor |
| 异常 | NULL 输入 | 函数应返回 -EINVAL |
| 异常 | shape 不匹配 | 应返回负值错误码 |
| 精度 | CPU vs GPU allclose | GPU 结果 vs CPU fallback |

### 可选场景

| 类别 | 用例 | 说明 |
| --- | --- | --- |
| 边界 | 全零输入 | 浮点运算特殊值 |
| 边界 | 全负值 | ReLU 等应全零 |
| 边界 | NaN / Inf | 如有定义 |
| 性能 | compute-sanitizer | 无 memory error |
| 回归 | 已修复的 bug | 防止复发 |

## 测试文件模板

```c
#include "unity.h"
#include "operator/nn/relu.h"

static tensor_t* input  = NULL;
static tensor_t* output = NULL;

void setUp(void) {
    input  = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){4, 256});
    output = tensor_create(DATA_TYPE_F32, 2, (int64_t[]){4, 256});
    tensor_rand_fill(input);
}

void tearDown(void) {
    tensor_destroy(input);
    tensor_destroy(output);
    input  = NULL;
    output = NULL;
}

void test_relu_f32_normal(void) {
    /* CPU 推理 */
    int ret = relu_f32(input, output, input->numel);
    TEST_ASSERT_EQUAL_INT(0, ret);
    /* 验证：不应有负值 */
    for (int64_t i = 0; i < input->numel; i++) {
        TEST_ASSERT_FLOAT_WITHIN(1e-6, fmaxf(0, input->data_f32[i]),
                                            output->data_f32[i]);
    }
}

void test_relu_f32_null_input(void) {
    int ret = relu_f32(NULL, output, 0);
    TEST_ASSERT_TRUE(ret < 0);
}

/* CUDA 版本 */
void test_relu_f32_cuda(void) {
    tensor_copy_to_device(input);
    int ret = relu_f32_cuda(input, output, NULL);
    TEST_ASSERT_EQUAL_INT(0, ret);
    tensor_copy_to_host(output);
    /* 对比 CPU fallback */
    tensor_t* expected = tensor_like(input);
    relu_f32(input, expected, input->numel);
    TEST_ASSERT_TRUE(tensor_allclose(expected, output, 1e-5, 1e-5));
    tensor_destroy(expected);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_relu_f32_normal);
    RUN_TEST(test_relu_f32_null_input);
    RUN_TEST(test_relu_f32_cuda);
    return UNITY_END();
}
```
