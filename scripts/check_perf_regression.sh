#!/bin/bash
# ============================================================
# check_perf_regression.sh — 性能回归检测
#
# 对比当前 profile 与基线，偏差超过阈值时告警。
# 用法：
#   ./scripts/check_perf_regression.sh <current.csv> [baseline.csv] [threshold_pct]
#
# 参数：
#   current.csv    — bench_profile.exe 输出的 CSV
#   baseline.csv   — 基线文件（默认 docs/PROFILE_BASELINE.csv）
#   threshold_pct  — 告警阈值百分比（默认 20）
#
# 退出码：
#   0 — 无回归
#   1 — 存在回归
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CURRENT="${1:?Usage: $0 <current.csv> [baseline.csv] [threshold_pct]}"
BASELINE="${2:-$PROJECT_ROOT/docs/PROFILE_BASELINE.csv}"
THRESHOLD="${3:-20}"

if [ ! -f "$CURRENT" ]; then
    echo "ERROR: $CURRENT not found"
    exit 1
fi

if [ ! -f "$BASELINE" ]; then
    echo "ERROR: $BASELINE not found"
    exit 1
fi

echo "=== Performance Regression Check ==="
echo "Current:  $CURRENT"
echo "Baseline: $BASELINE"
echo "Threshold: ${THRESHOLD}%"
echo ""

TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

REGRESSIONS=0
IMPROVEMENTS=0
CHECKED=0

# 跳过表头，逐行对比
tail -n +2 "$BASELINE" | while IFS=',' read -r op shape base_ms; do
    cur_line=$(grep "^${op},${shape}," "$CURRENT" 2>/dev/null || true)
    if [ -z "$cur_line" ]; then
        echo "  SKIP: $op ($shape) — not in current profile"
        continue
    fi

    cur_ms=$(echo "$cur_line" | cut -d',' -f3)

    result=$(awk -v base="$base_ms" -v cur="$cur_ms" -v thresh="$THRESHOLD" \
        'BEGIN {
            if (base <= 0) { print "skip"; exit }
            pct = (cur - base) / base * 100
            if (pct > thresh) printf "regress:%.2f", pct
            else if (pct < -10) printf "improve:%.2f", pct
            else printf "ok:%.2f", pct
        }')

    status=$(echo "$result" | cut -d':' -f1)
    pct=$(echo "$result" | cut -d':' -f2)

    case "$status" in
        regress)
            echo "  REGRESSION: $op ($shape) — baseline=${base_ms}ms, current=${cur_ms}ms (+${pct}%)"
            echo "regress" >> "$TMPFILE"
            ;;
        improve)
            echo "  IMPROVED:   $op ($shape) — baseline=${base_ms}ms, current=${cur_ms}ms (${pct}%)"
            echo "improve" >> "$TMPFILE"
            ;;
        ok)
            echo "ok" >> "$TMPFILE"
            ;;
    esac
done

REGRESSIONS=$(grep -c "regress" "$TMPFILE" 2>/dev/null || true)
REGRESSIONS=${REGRESSIONS:-0}
IMPROVEMENTS=$(grep -c "improve" "$TMPFILE" 2>/dev/null || true)
IMPROVEMENTS=${IMPROVEMENTS:-0}
CHECKED=$(wc -l < "$TMPFILE" 2>/dev/null || true)
CHECKED=${CHECKED:-0}
CHECKED=$(echo "$CHECKED" | tr -d '[:space:]')

echo ""
echo "Checked: $CHECKED entries"
echo "Regressions: $REGRESSIONS, Improvements: $IMPROVEMENTS"
if [ "$REGRESSIONS" -gt 0 ]; then
    echo "FAILED: $REGRESSIONS regression(s) detected (>${THRESHOLD}%)"
    exit 1
else
    echo "OK: No regressions detected"
    exit 0
fi
