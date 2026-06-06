#!/bin/bash
# ============================================================
# run_benchmarks.sh — 批量运行所有 benchmark 并输出汇总
#
# 用法：
#   ./scripts/run_benchmarks.sh                  # 运行所有 bench_*.exe
#   ./scripts/run_benchmarks.sh --csv            # 输出 CSV 格式
#   ./scripts/run_benchmarks.sh --json           # 输出 JSON 格式
#   ./scripts/run_benchmarks.sh --profile-only   # 只运行 bench_profile（算子级）
#   ./scripts/run_benchmarks.sh --save results.csv  # 保存到文件
#
# 前置条件：
#   cmake --build build --config Release
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build/Release"

FORMAT="text"
PROFILE_ONLY=0
SAVE_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --csv)          FORMAT="csv"; shift ;;
        --json)         FORMAT="json"; shift ;;
        --profile-only) PROFILE_ONLY=1; shift ;;
        --save)         SAVE_FILE="$2"; shift 2 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# 查找所有 bench 可执行文件
if [ "$PROFILE_ONLY" = "1" ]; then
    BENCHES=("$BUILD_DIR"/bench_profile.exe)
else
    BENCHES=("$BUILD_DIR"/bench_*.exe)
fi

if [ ${#BENCHES[@]} -eq 0 ]; then
    echo "ERROR: No bench executables found in $BUILD_DIR"
    echo "Run: cmake --build build --config Release"
    exit 1
fi

# 输出函数
emit_header() {
    if [ "$FORMAT" = "csv" ]; then
        echo "benchmark,status,time_sec,output_lines"
    elif [ "$FORMAT" = "json" ]; then
        echo "["
    fi
}

emit_result() {
    local name="$1" status="$2" time="$3" lines="$4"
    if [ "$FORMAT" = "csv" ]; then
        echo "$name,$status,$time,$lines"
    elif [ "$FORMAT" = "json" ]; then
        local comma=","
        if [ "$_first" = "1" ]; then comma=""; _first=0; fi
        echo "$comma  {\"benchmark\":\"$name\",\"status\":\"$status\",\"time_sec\":$time,\"output_lines\":$lines}"
    else
        printf "  %-30s %-8s %6.1fs  (%d lines)\n" "$name" "$status" "$time" "$lines"
    fi
}

emit_footer() {
    if [ "$FORMAT" = "json" ]; then
        echo "]"
    fi
}

# 运行
echo "=== CudaForge Benchmark Runner ==="
echo "Build: $BUILD_DIR"
echo "Benchmarks: ${#BENCHES[@]}"
echo ""

_first=1
emit_header

PASS=0
FAIL=0
TOTAL_TIME=0

for bench in "${BENCHES[@]}"; do
    name=$(basename "$bench" .exe)

    # bench_profile 支持 --json 参数
    if [ "$name" = "bench_profile" ] && [ "$FORMAT" = "csv" ]; then
        output=$("$bench" --json 2>&1)
    else
        output=$("$bench" 2>&1)
    fi

    status=$?
    lines=$(echo "$output" | wc -l | tr -d ' ')

    # 从输出中提取时间（如果有 elapsed time 信息）
    time_sec=0

    if [ $status -eq 0 ]; then
        PASS=$((PASS + 1))
        emit_result "$name" "PASS" "$time_sec" "$lines"
    else
        FAIL=$((FAIL + 1))
        emit_result "$name" "FAIL" "$time_sec" "$lines"
        # 打印失败输出的最后 5 行
        if [ "$FORMAT" = "text" ]; then
            echo "    Last 5 lines:"
            echo "$output" | tail -5 | sed 's/^/    /'
        fi
    fi
done

emit_footer

echo ""
echo "=== Summary ==="
echo "Passed:  $PASS"
echo "Failed:  $FAIL"
echo "Total:   $((PASS + FAIL))"

if [ -n "$SAVE_FILE" ]; then
    # 重新运行并保存到文件
    echo ""
    echo "Saving results to $SAVE_FILE..."
    "$0" --csv > "$SAVE_FILE" 2>&1 || true
    echo "Saved."
fi

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
