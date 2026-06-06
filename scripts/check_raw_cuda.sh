#!/bin/bash
# ============================================================
# check_raw_cuda.sh — 检测绕过 g_cuda 接口层的裸 CUDA API 调用
#
# 扫描 src/operator/ 下的 .cu 文件，查找直接调用 CUDA Runtime API
# 而非通过 g_cuda 函数指针表的代码。
#
# 允许的例外：
#   - cudaStream_t 类型声明（只是类型，不是 API 调用）
#   - cuda_runtime.h include
#   - CUDA_KERNEL_LAUNCH / cudaLaunchKernel（通过 cuda_ops.h 封装）
#   - 注释和字符串中的 CUDA API 名
#
# 用法：
#   ./scripts/check_raw_cuda.sh [--fix-suggestions]
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SHOW_SUGGESTIONS=0
if [ "${1:-}" = "--fix-suggestions" ]; then
    SHOW_SUGGESTIONS=1
fi

# 要检测的 CUDA Runtime API（内存管理 + 同步）
BLOCKED_APIS="cudaMalloc|cudaFree|cudaMemcpy[^A]|cudaMemcpyAsync|cudaMemset|cudaStreamCreate|cudaStreamDestroy|cudaStreamSynchronize|cudaEventCreate|cudaEventDestroy|cudaEventRecord|cudaEventSynchronize|cudaEventElapsedTime|cudaDeviceSynchronize|cudaGetLastError|cudaGetErrorString"

# 平台层目录（允许使用裸 CUDA API）
PLATFORM_DIR="src/platform/cuda"

# 算子目录（禁止使用裸 CUDA API）
OPERATOR_DIR="src/operator"

ERRORS=0
WARNINGS=0

echo "=== Checking for raw CUDA API calls in operator code ==="
echo ""

# 扫描算子目录下的 .cu 文件
for f in $(find "$PROJECT_ROOT/$OPERATOR_DIR" -name "*.cu" -not -path "*/test/*" 2>/dev/null); do
    # 跳过平台层文件
    if echo "$f" | grep -q "$PLATFORM_DIR"; then
        continue
    fi

    # 查找裸 CUDA API 调用（排除注释、字符串、类型声明）
    matches=$(grep -nE "\b($BLOCKED_APIS)\s*\(" "$f" 2>/dev/null | \
              grep -v "^\s*//" | \
              grep -v "^\s*\*" | \
              grep -v "cudaStream_t" | \
              grep -v "CUDA_KERNEL_LAUNCH" | \
              grep -v "cudaLaunchKernel" | \
              grep -v "// *allow-raw" | \
              grep -v "#.*include" || true)

    if [ -n "$matches" ]; then
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$f" 2>/dev/null || echo "$f")
        echo "  WARNING: Raw CUDA API in $rel_path:"
        echo "$matches" | while IFS= read -r line; do
            echo "    $line"
        done
        WARNINGS=$((WARNINGS + 1))

        if [ "$SHOW_SUGGESTIONS" = "1" ]; then
            echo "  Suggestion: Use g_cuda.memcpy_d2d() / g_cuda.device_alloc() instead"
            echo ""
        fi
    fi
done

# 检查 cudaGetErrorString 的使用（应使用 CUDA_CHECK 宏）
for f in $(find "$PROJECT_ROOT/$OPERATOR_DIR" -name "*.cu" -not -path "*/test/*" 2>/dev/null); do
    if echo "$f" | grep -q "$PLATFORM_DIR"; then
        continue
    fi

    matches=$(grep -nE "cudaGetErrorString" "$f" 2>/dev/null | \
              grep -v "^\s*//" | \
              grep -v "CUDA_CHECK" || true)

    if [ -n "$matches" ]; then
        rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$f" 2>/dev/null || echo "$f")
        echo "  WARNING: Direct cudaGetErrorString in $rel_path (use CUDA_CHECK macro instead)"
        WARNINGS=$((WARNINGS + 1))
    fi
done

echo ""
if [ "$WARNINGS" -gt 0 ]; then
    echo "Found $WARNINGS file(s) with raw CUDA API calls"
    echo "These should use g_cuda.* interface or CUDA_CHECK macro"
    exit 1
else
    echo "OK: No raw CUDA API calls found in operator code"
    exit 0
fi
