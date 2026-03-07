#!/bin/bash
# Install script for LAPS (Length-Aware Prefill Scheduling)
#
# Creates a conda environment and installs all dependencies needed to run
# the LAPS benchmarks (e.g., bench_32b_concurrency.sh).
#
# Usage:
#   bash install.sh              # use defaults (env name: laps)
#   ENV_NAME=my_env bash install.sh  # custom env name
#
# Prerequisites:
#   - conda (miniconda or anaconda)
#   - NVIDIA GPU with CUDA drivers installed
set -euo pipefail

ENV_NAME="${ENV_NAME:-laps}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  LAPS Environment Setup"
echo "  Env name:       ${ENV_NAME}"
echo "  Python version: ${PYTHON_VERSION}"
echo "============================================================"

# ── Check prerequisites ──────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install miniconda or anaconda first."
    exit 1
fi

if ! nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# ── Create conda environment ─────────────────────────────────────
if conda env list | grep -qw "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "[1/3] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

PYTHON="$(conda run -n "${ENV_NAME}" which python)"
PIP="$(conda run -n "${ENV_NAME}" which pip)"
echo "  Python: ${PYTHON}"

# ── Install sglang (LAPS fork) in editable mode ─────────────────
echo "[2/3] Installing LAPS sglang package (editable mode)..."
"${PIP}" install -e "${SCRIPT_DIR}/python"

# ── Install additional runtime dependencies ──────────────────────
echo "[3/3] Installing mooncake-transfer-engine and sglang-router..."
"${PIP}" install mooncake-transfer-engine sglang-router

# ── Verify installation ─────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"

"${PYTHON}" -c "
import sglang;        print(f'  sglang:         {sglang.__version__}')
import mooncake;      print(f'  mooncake:        OK')
import sglang_router; print(f'  sglang_router:   OK')
import torch;         print(f'  torch:           {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import sgl_kernel;    print(f'  sgl_kernel:      OK')
import flashinfer;    print(f'  flashinfer:      OK')
import transformers;  print(f'  transformers:    {transformers.__version__}')
"

echo ""
echo "============================================================"
echo "  Installation complete!"
echo ""
echo "  To activate:  conda activate ${ENV_NAME}"
echo "  Python path:  ${PYTHON}"
echo ""
echo "  To run benchmarks:"
echo "    bash bench_laps_prefill_throughput/bench_32b_concurrency.sh"
echo "============================================================"
