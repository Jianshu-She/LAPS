#!/bin/bash
# Run all LAPS A100 benchmarks: 0.5B, 3B, 7B on 2 GPUs.
#
# Usage:
#   bash run_a100_all.sh              # run all 3 models
#   bash run_a100_all.sh 0.5b 7b      # run specific models
#
# Each model runs 3 settings x 7 concurrency levels = 21 data points.
# Total: 63 data points across 3 models.
#
# Estimated time: ~75 min total (0.5B ~10 min, 3B ~25 min, 7B ~40 min)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="0.5b 3b 7b"
fi

PASSED=0
FAILED=0

for model in $MODELS; do
    echo ""
    echo "############################################################"
    echo "  Starting: ${model}"
    echo "  Time: $(date)"
    echo "############################################################"

    if bash "${SCRIPT_DIR}/bench_a100_2gpu.sh" "$model"; then
        echo "  ${model}: PASSED"
        PASSED=$((PASSED + 1))
    else
        echo "  ${model}: FAILED (check logs)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "############################################################"
echo "  All done. Passed: ${PASSED}, Failed: ${FAILED}"
echo "  Time: $(date)"
echo "############################################################"
