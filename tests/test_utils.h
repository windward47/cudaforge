/**
 * test_utils.h — 公共测试工具库
 *
 * 提供随机填充、allclose 比较、多轮验证等常用测试辅助函数。
 * 用法：#include "test_utils.h"
 */
#ifndef TEST_UTILS_H_
#define TEST_UTILS_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/* ============================================================
 * 可复现伪随机填充（LCG 算法）
 * ============================================================ */
static inline void test_random_fill(float* data, int64_t n, unsigned seed) {
    unsigned state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = ((float)(state & 0x7FFFFFFF) / 2147483648.0f) - 1.0f;
    }
}

/* 填充指定范围的随机值 */
static inline void test_random_fill_range(float* data, int64_t n,
                                           unsigned seed, float lo, float hi) {
    unsigned state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        float t = (float)(state & 0x7FFFFFFF) / 2147483648.0f;
        data[i] = lo + t * (hi - lo);
    }
}

/* ============================================================
 * allclose 比较（NumPy 风格：|a-b| <= atol + rtol * max(|a|,|b|)）
 * 返回 1 表示全部通过，0 表示有不匹配
 * ============================================================ */
static inline int test_allclose(const float* a, const float* b, int64_t n,
                                 float rtol, float atol, int* mismatch_idx) {
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        float maxv = fmaxf(fabsf(a[i]), fabsf(b[i]));
        if (diff > atol && diff > rtol * maxv) {
            if (mismatch_idx) *mismatch_idx = (int)i;
            return 0;
        }
    }
    return 1;
}

/* ============================================================
 * max_abs_diff: 返回最大绝对误差
 * ============================================================ */
static inline float test_max_abs_diff(const float* a, const float* b, int64_t n) {
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > maxd) maxd = diff;
    }
    return maxd;
}

/* ============================================================
 * mean_abs_diff: 返回平均绝对误差
 * ============================================================ */
static inline float test_mean_abs_diff(const float* a, const float* b, int64_t n) {
    double sum = 0.0;
    for (int64_t i = 0; i < n; i++)
        sum += fabs((double)a[i] - (double)b[i]);
    return (float)(sum / (double)n);
}

/* ============================================================
 * 详细比较报告：打印 max_diff、mismatch 数量、首个不匹配位置
 * 返回不匹配数量
 * ============================================================ */
static inline int test_compare_report(const float* cpu, const float* cuda,
                                       size_t n, const char* name, float tol) {
    float max_diff = 0.0f;
    int mismatch = 0;
    int max_idx = -1;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf(cpu[i] - cuda[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = (int)i; }
        if (diff > tol) mismatch++;
    }
    fprintf(stderr, "%s: max_diff=%.6e mismatches>%.0e=%d/%zu",
            name, max_diff, tol, mismatch, n);
    if (max_idx >= 0)
        fprintf(stderr, " [idx=%d CPU=%.6f CUDA=%.6f]", max_idx, cpu[max_idx], cuda[max_idx]);
    fprintf(stderr, "\n");
    return mismatch;
}

/* ============================================================
 * 多轮验证宏
 *
 * 用法：
 *   static void test_relu_trial(int trial) {
 *       unsigned seed = 42 + trial * 1000;
 *       // ... 用 seed 生成随机输入，运行并比较
 *   }
 *   void test_relu_f32(void) {
 *       RUN_N_TRIALS(5, test_relu_trial);
 *   }
 * ============================================================ */
#define RUN_N_TRIALS(n, test_fn) \
    do { \
        for (int _trial = 0; _trial < (n); _trial++) { \
            test_fn(_trial); \
        } \
    } while(0)

/* ============================================================
 * TEST_CHECK: 断言失败时打印消息并继续（不 exit）
 * 适用于非 Unity 测试框架
 * ============================================================ */
#define TEST_CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL at %s:%d: %s\n", __FILE__, __LINE__, msg); \
        fflush(stderr); \
    } \
} while(0)

/* ============================================================
 * CUDA 测试辅助宏
 * ============================================================ */
#ifdef USE_CUDA

/* 分配 + H2D 传输 */
#define CUDA_ALLOC_AND_COPY(d_ptr, h_ptr, n_bytes) do { \
    d_ptr = (typeof(d_ptr))g_cuda.device_alloc(n_bytes); \
    g_cuda.memcpy_h2d(d_ptr, h_ptr, n_bytes, 0); \
} while(0)

/* D2H 传输 + 释放 */
#define CUDA_COPY_FREE(h_ptr, d_ptr, n_bytes) do { \
    g_cuda.memcpy_d2h(h_ptr, d_ptr, n_bytes, 0); \
    g_cuda.device_free(d_ptr); \
} while(0)

#endif /* USE_CUDA */

#endif /* TEST_UTILS_H_ */
