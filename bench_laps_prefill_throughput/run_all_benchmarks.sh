#!/bin/bash
# Run all model benchmarks sequentially: 7B(2GPU) -> 14B(8GPU) -> 32B(8GPU) -> 72B(8GPU)
#
# Usage:
#   bash run_all_benchmarks.sh          # run all 4
#   bash run_all_benchmarks.sh 7b       # run only 7B
#   bash run_all_benchmarks.sh 14b 32b  # run 14B and 32B
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_all_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse which benchmarks to run
if [ $# -eq 0 ]; then
    BENCHMARKS="7b 14b 32b 72b"
else
    BENCHMARKS="$*"
fi

declare -A SCRIPTS=(
    ["7b"]="bench_7b_2gpu.sh"
    ["14b"]="bench_14b_8gpu.sh"
    ["32b"]="bench_32b_8gpu.sh"
    ["72b"]="bench_72b_8gpu.sh"
)

declare -A GPUS=(
    ["7b"]="2 GPUs (TP=1)"
    ["14b"]="8 GPUs (TP=4)"
    ["32b"]="8 GPUs (TP=4)"
    ["72b"]="8 GPUs (TP=4)"
)

echo "============================================================"
echo "  Run All Benchmarks"
echo "  Time:  $(date)"
echo "  Tasks: ${BENCHMARKS}"
echo "============================================================"

FAILED=""
PASSED=""

for bench in $BENCHMARKS; do
    script="${SCRIPTS[$bench]:-}"
    if [ -z "$script" ]; then
        echo "[ERROR] Unknown benchmark: ${bench} (valid: 7b, 14b, 32b, 72b)"
        FAILED="${FAILED} ${bench}"
        continue
    fi

    script_path="${SCRIPT_DIR}/${script}"
    if [ ! -f "$script_path" ]; then
        echo "[ERROR] Script not found: ${script_path}"
        FAILED="${FAILED} ${bench}"
        continue
    fi

    log_file="${LOG_DIR}/${bench}_${TIMESTAMP}.log"

    echo ""
    echo "============================================================"
    echo "  Starting: ${bench} (${GPUS[$bench]})"
    echo "  Script:   ${script}"
    echo "  Log:      ${log_file}"
    echo "  Time:     $(date)"
    echo "============================================================"

    if bash "$script_path" 2>&1 | tee "$log_file"; then
        echo ""
        echo "[DONE] ${bench} completed successfully at $(date)"
        PASSED="${PASSED} ${bench}"
    else
        echo ""
        echo "[FAIL] ${bench} failed at $(date). Check log: ${log_file}"
        FAILED="${FAILED} ${bench}"
    fi
done

echo ""
echo "============================================================"
echo "  All Benchmarks Summary"
echo "  Time:   $(date)"
echo "  Passed: ${PASSED:-none}"
echo "  Failed: ${FAILED:-none}"
echo "  Logs:   ${LOG_DIR}/"
echo "============================================================"

[ -z "$FAILED" ] && exit 0 || exit 1
