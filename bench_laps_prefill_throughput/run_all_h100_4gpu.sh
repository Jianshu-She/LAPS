#!/bin/bash
# Run LAPS benchmarks on 4x H100 GPUs (no InfiniBand required).
#
# Uses nixl backend (UCX/shared memory) instead of mooncake RDMA.
# Runs: 7B (TP=1, 2 GPUs) → 14B (TP=2, 4 GPUs)
#
# Usage:
#   bash run_all_h100_4gpu.sh          # run both models
#   bash run_all_h100_4gpu.sh 7b       # run only 7B
#   bash run_all_h100_4gpu.sh 14b      # run only 14B
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="7b 14b"
fi

PASSED=0
FAILED=0

for model in $MODELS; do
    echo ""
    echo "############################################################"
    echo "  Starting: ${model} (nixl backend)"
    echo "  Time: $(date)"
    echo "############################################################"

    if bash "${SCRIPT_DIR}/scripts/bench_h100_4gpu.sh" "$model"; then
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
