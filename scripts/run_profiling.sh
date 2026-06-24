#!/bin/bash
# ============================================================
# run_profiling.sh — nsys + compute-sanitizer 综合性能分析
#
# 用法：
#   ./scripts/run_profiling.sh                      # 默认: bench_bert_mha
#   ./scripts/run_profiling.sh bench_bert_mha        # 指定 benchmark
#   ./scripts/run_profiling.sh --sanitizer-only      # 只跑 compute-sanitizer
#   ./scripts/run_profiling.sh --nsys-only           # 只跑 nsys
#
# 详见 docs/PROFILING_GUIDE.md
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build/Release"

# 工具路径（Windows）
NSYS="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/nsys.exe"
SQLITE="C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.6.3/target-windows-x64/sqlite3.exe"
SANITIZER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/compute-sanitizer.bat"

TARGET="bench_bert_mha"
RUN_SANITIZER=1
RUN_NSYS=1

while [ $# -gt 0 ]; do
    case "$1" in
        --sanitizer-only) RUN_NSYS=0; shift ;;
        --nsys-only)      RUN_SANITIZER=0; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  TARGET="$1"; shift ;;
    esac
done

EXE="$BUILD_DIR/$TARGET.exe"
if [ ! -f "$EXE" ]; then
    echo "ERROR: $EXE not found. Run cmake --build first."
    exit 1
fi

# ============================================================
# Step 1: compute-sanitizer (correctness gate)
# ============================================================
if [ $RUN_SANITIZER -eq 1 ]; then
    echo "=========================================="
    echo "Step 1: compute-sanitizer (memcheck)"
    echo "=========================================="
    "$SANITIZER" --tool memcheck "$EXE" 2>&1 | grep -E "ERROR SUMMARY|PASS|FAIL" || true
    echo ""
fi

# ============================================================
# Step 2: nsys profile (kernel time + smem/regs breakdown)
# ============================================================
if [ $RUN_NSYS -eq 1 ]; then
    echo "=========================================="
    echo "Step 2: nsys profile"
    echo "=========================================="
    REP="$PROJECT_ROOT/prof_${TARGET}"
    "$NSYS" profile --stats=true --trace=cuda --force-overwrite=true -o "$REP" "$EXE" 2>&1 | grep -E "cuda_gpu_kern_sum|cuda_gpu_mem" -A 2 || true

    echo ""
    echo "--- Kernel time / smem / regs breakdown ---"
    "$SQLITE" -header -column "${REP}.sqlite" "
    SELECT substr(s.value, 1, 55) as kernel, COUNT(*) as calls,
           printf('%.3f', SUM(k.end-k.start)/1e6) as total_ms,
           printf('%.3f', AVG(k.end-k.start)/1e6) as avg_ms,
           k.registersPerThread as regs,
           printf('%.1f', (k.dynamicSharedMemory+k.staticSharedMemory)/1024.0) as smem_kb
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
    GROUP BY s.value ORDER BY total_ms DESC LIMIT 12;" 2>&1

    echo ""
    echo "--- Occupancy estimate (RTX 2050: 100KB smem/SM, 65536 regs/SM) ---"
    "$SQLITE" -header -column "${REP}.sqlite" "
    SELECT substr(s.value, 1, 45) as kernel,
           printf('%.1f', (k.dynamicSharedMemory+k.staticSharedMemory)/1024.0) as smem_kb,
           k.registersPerThread as regs,
           k.blockX*k.blockY*k.blockZ as threads,
           CASE WHEN (k.dynamicSharedMemory+k.staticSharedMemory) > 0
                THEN 102400 / (k.dynamicSharedMemory+k.staticSharedMemory)
                ELSE 16 END as smem_blocks_per_sm
    FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
    WHERE s.value LIKE 'mha_flash%' OR s.value LIKE 'mha_preloaded%'
    GROUP BY s.value;" 2>&1
fi

echo ""
echo "Done. Report: ${REP}.nsys-rep (open with nsys-ui)"
