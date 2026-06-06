#!/bin/bash
# ============================================================
# check_registry.sh — 验证 operator_registry.def 与实际源文件一致性
#
# 检查项：
#   1. .def 中每个 CPU 注册函数在其 .c 文件中有定义
#   2. .def 中每个 CUDA 注册函数在其 .cu 文件中有定义
#   3. 实际源文件中的注册函数是否都在 .def 中列出
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEF_FILE="$PROJECT_ROOT/src/operator/operator_registry.def"

if [ ! -f "$DEF_FILE" ]; then
    echo "ERROR: $DEF_FILE not found"
    exit 1
fi

ERRORS=0

echo "=== Checking operator_registry.def ==="

# ---- 1. 检查 .def 中的 CPU 注册函数是否有定义 ----
echo ""
echo "[1/3] Checking CPU registration functions in .c files..."
while IFS= read -r line; do
    # 提取 REGISTER_CPU(register_xxx) 中的函数名
    fn=$(echo "$line" | sed -n 's/.*REGISTER_CPU(\([^)]*\)).*/\1/p')
    if [ -z "$fn" ]; then
        continue
    fi

    # 在 .c 文件中查找函数定义
    matches=$(grep -rl "int ${fn}(" "$PROJECT_ROOT/src/operator/" --include="*.c" 2>/dev/null || true)
    if [ -z "$matches" ]; then
        echo "  ERROR: $fn() — not found in any .c file"
        ERRORS=$((ERRORS + 1))
    fi
done < "$DEF_FILE"

# ---- 2. 检查 .def 中的 CUDA 注册函数是否有定义 ----
echo ""
echo "[2/3] Checking CUDA registration functions in .cu files..."
while IFS= read -r line; do
    fn=$(echo "$line" | sed -n 's/.*REGISTER_CUDA(\([^)]*\)).*/\1/p')
    if [ -z "$fn" ]; then
        continue
    fi

    # 在 .cu 文件中查找 extern "C" 函数定义
    matches=$(grep -rl "int ${fn}(" "$PROJECT_ROOT/src/operator/" --include="*.cu" 2>/dev/null || true)
    if [ -z "$matches" ]; then
        echo "  ERROR: $fn() — not found in any .cu file"
        ERRORS=$((ERRORS + 1))
    fi
done < "$DEF_FILE"

# ---- 3. 检查实际源文件中的注册函数是否都在 .def 中 ----
echo ""
echo "[3/3] Checking for registration functions missing from .def..."

# 查找所有 .c 文件中的 register_*_f32() 定义
for f in $(find "$PROJECT_ROOT/src/operator" -name "*.c" -not -name "operator_*" -not -path "*/test/*"); do
    grep -oP 'int\s+(register_\w+)\s*\(' "$f" 2>/dev/null | while read -r match; do
        fn=$(echo "$match" | grep -oP 'register_\w+')
        if ! grep -q "REGISTER_CPU($fn)" "$DEF_FILE" 2>/dev/null; then
            echo "  WARNING: $fn() in $(basename "$f") — not in .def"
        fi
    done
done

# 查找所有 .cu 文件中的 register_*_cuda() 定义
for f in $(find "$PROJECT_ROOT/src/operator" -name "*.cu" -not -path "*/test/*"); do
    grep -oP 'int\s+(register_\w+)\s*\(' "$f" 2>/dev/null | while read -r match; do
        fn=$(echo "$match" | grep -oP 'register_\w+')
        if ! grep -q "REGISTER_CUDA($fn)" "$DEF_FILE" 2>/dev/null; then
            echo "  WARNING: $fn() in $(basename "$f") — not in .def"
        fi
    done
done

echo ""
if [ "$ERRORS" -gt 0 ]; then
    echo "FAILED: $ERRORS error(s) found"
    exit 1
else
    echo "OK: operator_registry.def is consistent with source files"
    exit 0
fi
