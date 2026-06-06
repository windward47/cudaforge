#!/bin/bash
# ============================================================
# run_sanitizer.sh — 批量运行 compute-sanitizer 并汇总结果
#
# 用法：
#   ./scripts/run_sanitizer.sh                   # 运行所有 CUDA 测试
#   ./scripts/run_sanitizer.sh --tool memcheck    # 指定 sanitizer 工具
#   ./scripts/run_sanitizer.sh --tests-only       # 只运行 test_*.exe
#   ./scripts/run_sanitizer.sh --bench-only       # 只运行 bench_*.exe
#   ./scripts/run_sanitizer.sh --target test_matmul  # 只运行指定目标
#   ./scripts/run_sanitizer.sh --save report.txt  # 保存报告
#
# 前置条件：
#   cmake --build build --config Release
#   compute-sanitizer 在 PATH 中
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build/Release"

TOOL="memcheck"
MODE="all"  # all, tests, bench
TARGET=""
SAVE_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --tool)        TOOL="$2"; shift 2 ;;
        --tests-only)  MODE="tests"; shift ;;
        --bench-only)  MODE="bench"; shift ;;
        --target)      TARGET="$2"; shift 2 ;;
        --save)        SAVE_FILE="$2"; shift 2 ;;
        *)             echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 检查 compute-sanitizer
if ! command -v compute-sanitizer &>/dev/null; then
    echo "ERROR: compute-sanitizer not found in PATH"
    echo "Add CUDA toolkit bin directory to PATH"
    exit 1
fi

# 收集目标
TARGETS=()
if [ -n "$TARGET" ]; then
    TARGETS=("$BUILD_DIR/${TARGET}.exe")
elif [ "$MODE" = "tests" ]; then
    TARGETS=("$BUILD_DIR"/test_*.exe)
elif [ "$MODE" = "bench" ]; then
    TARGETS=("$BUILD_DIR"/bench_*.exe)
else
    TARGETS=("$BUILD_DIR"/test_*.exe "$BUILD_DIR"/bench_*.exe)
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "ERROR: No executables found in $BUILD_DIR"
    exit 1
fi

echo "=== CudaForge compute-sanitizer Runner ==="
echo "Tool:     $TOOL"
echo "Targets:  ${#TARGETS[@]}"
echo ""

PASS=0
FAIL=0
ERRORS_TOTAL=0
REPORT=""

for exe in "${TARGETS[@]}"; do
    if [ ! -f "$exe" ]; then
        continue
    fi

    name=$(basename "$exe" .exe)
    printf "  %-35s " "$name"

    # 运行 compute-sanitizer，捕获输出
    output=$(compute-sanitizer --tool "$TOOL" "$exe" 2>&1) || true

    # 检查结果
    if echo "$output" | grep -q "ERROR SUMMARY: 0 errors"; then
        echo "PASS (0 errors)"
        PASS=$((PASS + 1))
    elif echo "$output" | grep -q "ERROR SUMMARY:"; then
        errors=$(echo "$output" | grep "ERROR SUMMARY:" | grep -oP '\d+' | head -1)
        echo "FAIL ($errors errors)"
        FAIL=$((FAIL + 1))
        ERRORS_TOTAL=$((ERRORS_TOTAL + errors))
        # 保存失败详情
        REPORT="$REPORT\n--- $name ($errors errors) ---\n"
        REPORT="$REPORT$(echo "$output" | grep -A2 "ERROR" | head -10)\n"
    else
        # 可能 sanitizer 本身出错（如超时）
        echo "SKIP (sanitizer error)"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Passed:  $PASS"
echo "Failed:  $FAIL"
echo "Total:   $((PASS + FAIL))"
echo "Errors:  $ERRORS_TOTAL"

if [ -n "$REPORT" ]; then
    echo ""
    echo "=== Error Details ==="
    echo -e "$REPORT"
fi

if [ -n "$SAVE_FILE" ]; then
    {
        echo "compute-sanitizer Report ($TOOL)"
        echo "Date: $(date)"
        echo "Passed: $PASS, Failed: $FAIL, Errors: $ERRORS_TOTAL"
        echo ""
        for exe in "${TARGETS[@]}"; do
            if [ ! -f "$exe" ]; then continue; fi
            name=$(basename "$exe" .exe)
            output=$(compute-sanitizer --tool "$TOOL" "$exe" 2>&1) || true
            echo "--- $name ---"
            echo "$output" | grep -E "ERROR SUMMARY|==.*ERROR|Program hit" | head -5
            echo ""
        done
    } > "$SAVE_FILE"
    echo "Report saved to: $SAVE_FILE"
fi

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
