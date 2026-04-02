#!/bin/bash
# Run all 4 benchmark scripts sequentially.
# Results go to: results_7b_2gpu/, results_14b_8gpu/, results_32b_8gpu/, results_72b_8gpu/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "  Starting all benchmarks"
echo "  $(date)"
echo "========================================"

for script in bench_7b_2gpu.sh bench_14b_8gpu.sh bench_32b_8gpu.sh bench_72b_8gpu.sh; do
    echo ""
    echo "########################################"
    echo "  Running: ${script}"
    echo "  $(date)"
    echo "########################################"
    bash "${SCRIPT_DIR}/scripts/${script}"
    echo ""
    echo "  Finished: ${script} at $(date)"
done

echo ""
echo "========================================"
echo "  All benchmarks complete!"
echo "  $(date)"
echo "========================================"
