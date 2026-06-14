#!/usr/bin/env bash
# ============================================================
# run.sh — 批量运行 experiments/2/run.py 角度扫描实验
# 支持在 Git Bash / WSL / Linux / macOS 等 Unix Shell 中执行
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIZES=(8 12 16 20 40 80 100 148 200 1000)

for size in "${SIZES[@]}"; do
    echo "========================================"
    echo "  SIZE = ${size} × ${size}"
    echo "========================================"
    python "${SCRIPT_DIR}/2/run.py" "${size}"
    echo ""
done

echo "全部实验完成！"
