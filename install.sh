#!/bin/bash
# Install script for LAPS (Length-Aware Prefill Scheduling)
#
# Creates a conda environment and installs all dependencies needed to run
# the LAPS benchmarks (e.g., bench_7b_2gpu.sh).
#
# Usage:
#   bash install.sh              # use defaults (env name: laps)
#   ENV_NAME=my_env bash install.sh  # custom env name
#
# Prerequisites:
#   - conda (miniconda or anaconda)
#   - NVIDIA GPU with CUDA drivers installed (Driver >= 550)
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

# ── [0/6] Install system libraries (libibverbs) ─────────────────
# mooncake-transfer-engine requires libibverbs.so even when using
# TCP or nixl backends, because its shared library links against it
# at load time.
echo ""
echo "[0/6] Checking system libraries (libibverbs)..."
if ldconfig -p 2>/dev/null | grep -q libibverbs; then
    echo "  libibverbs: found"
else
    echo "  libibverbs: NOT found"
    echo "  mooncake requires libibverbs.so.1 even for TCP/nixl mode."
    if command -v apt-get &>/dev/null; then
        echo "  Attempting to install via apt-get..."
        if sudo -n apt-get update -qq && sudo -n apt-get install -y libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev 2>/dev/null; then
            echo "  libibverbs: installed successfully"
        else
            echo "  WARNING: Could not install libibverbs (need sudo)."
            echo "  Please run manually: sudo apt-get install -y libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev"
        fi
    elif command -v yum &>/dev/null; then
        echo "  Attempting to install via yum..."
        if sudo -n yum install -y libibverbs libibverbs-devel librdmacm librdmacm-devel 2>/dev/null; then
            echo "  libibverbs: installed successfully"
        else
            echo "  WARNING: Could not install libibverbs (need sudo)."
            echo "  Please run manually: sudo yum install -y libibverbs libibverbs-devel librdmacm librdmacm-devel"
        fi
    else
        echo "  WARNING: Unknown package manager. Please install libibverbs manually."
    fi
fi

# ── [1/6] Create conda environment ──────────────────────────────
if conda env list | grep -qw "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo ""
    echo "[1/6] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

CONDA_PREFIX_PATH="$(conda run -n "${ENV_NAME}" printenv CONDA_PREFIX)"
PYTHON="${CONDA_PREFIX_PATH}/bin/python"
PIP="${CONDA_PREFIX_PATH}/bin/pip"
echo "  Python:       ${PYTHON}"
echo "  CONDA_PREFIX: ${CONDA_PREFIX_PATH}"

# ── [2/6] Install CUDA Toolkit if nvcc is missing ───────────────
echo ""
echo "[2/6] Checking CUDA Toolkit (nvcc)..."
if conda run -n "${ENV_NAME}" which nvcc &>/dev/null; then
    echo "  nvcc: found ($(conda run -n "${ENV_NAME}" nvcc --version 2>&1 | grep release | awk '{print $NF}'))"
else
    echo "  nvcc: NOT found"
    echo "  flashinfer JIT compilation requires nvcc. Installing CUDA Toolkit via conda..."
    # Detect PyTorch CUDA version to match
    TORCH_CUDA=$("${PYTHON}" -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "12.8")
    CUDA_MAJOR_MINOR=$(echo "${TORCH_CUDA}" | cut -d. -f1,2)
    echo "  PyTorch CUDA version: ${TORCH_CUDA}, installing cuda-toolkit matching ${CUDA_MAJOR_MINOR}..."
    # Try specific version channel first, fallback to generic
    conda install -n "${ENV_NAME}" -c "nvidia/label/cuda-${CUDA_MAJOR_MINOR}.0" cuda-toolkit -y 2>/dev/null \
        || conda install -n "${ENV_NAME}" -c nvidia cuda-toolkit -y
    echo "  nvcc: installed"
fi

# ── [3/6] Install sglang (LAPS fork) in editable mode ───────────
echo ""
echo "[3/6] Installing LAPS sglang package (editable mode)..."
# Force-remove stale sglang if it exists without RECORD (e.g. leftover
# editable install from a different directory). Without this, pip fails
# with "Cannot uninstall sglang: no RECORD file".
SITE_PACKAGES="${CONDA_PREFIX_PATH}/lib/python${PYTHON_VERSION}/site-packages"
if "${PIP}" show sglang &>/dev/null; then
    echo "  Existing sglang found, removing first..."
    "${PIP}" uninstall sglang -y 2>/dev/null || true
    # If pip uninstall failed (no RECORD), nuke all sglang artifacts manually
    rm -rf "${SITE_PACKAGES}"/sglang-*.dist-info 2>/dev/null || true
    rm -f "${SITE_PACKAGES}"/sglang.egg-link 2>/dev/null || true
    rm -f "${SITE_PACKAGES}"/__editable__.sglang-*.pth 2>/dev/null || true
    echo "  Cleaned up stale sglang"
fi
"${PIP}" install -e "${SCRIPT_DIR}/python"

# ── [4/6] Install runtime dependencies ───────────────────────────
echo ""
echo "[4/6] Installing mooncake-transfer-engine, sglang-router, and nixl..."
"${PIP}" install mooncake-transfer-engine sglang-router nixl matplotlib

# ── [5/6] Fix lib64 symlinks for flashinfer JIT ─────────────────
# flashinfer's ninja build searches -L$CONDA_PREFIX/lib64 and
# -L$CONDA_PREFIX/lib64/stubs for -lcudart and -lcuda. Conda
# installs these in lib/ and targets/.../stubs/ instead.
echo ""
echo "[5/6] Creating lib64 symlinks for flashinfer JIT compilation..."
mkdir -p "${CONDA_PREFIX_PATH}/lib64" "${CONDA_PREFIX_PATH}/lib64/stubs"
if [ -f "${CONDA_PREFIX_PATH}/lib/libcudart.so" ]; then
    ln -sf "${CONDA_PREFIX_PATH}/lib/libcudart.so" "${CONDA_PREFIX_PATH}/lib64/libcudart.so"
    echo "  lib64/libcudart.so -> lib/libcudart.so"
else
    echo "  WARNING: lib/libcudart.so not found, skipping symlink"
fi
if [ -f "${CONDA_PREFIX_PATH}/targets/x86_64-linux/lib/stubs/libcuda.so" ]; then
    ln -sf "${CONDA_PREFIX_PATH}/targets/x86_64-linux/lib/stubs/libcuda.so" "${CONDA_PREFIX_PATH}/lib64/stubs/libcuda.so"
    echo "  lib64/stubs/libcuda.so -> targets/.../stubs/libcuda.so"
else
    echo "  WARNING: targets/.../stubs/libcuda.so not found, skipping symlink"
fi
# Clear flashinfer JIT cache to force recompilation with correct paths
rm -rf "${HOME}/.cache/flashinfer" 2>/dev/null || true
echo "  Cleared flashinfer JIT cache"

# Set up conda activate.d so CUDA_HOME and LD_LIBRARY_PATH are auto-set
ACTIVATE_DIR="${CONDA_PREFIX_PATH}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX_PATH}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/laps_env.sh" << ACTIVATE_EOF
#!/bin/bash
export CUDA_HOME="\${CONDA_PREFIX}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH:-}"
export _LAPS_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
ACTIVATE_EOF

cat > "${DEACTIVATE_DIR}/laps_env.sh" << DEACTIVATE_EOF
#!/bin/bash
unset CUDA_HOME
if [ -n "\${_LAPS_OLD_LD_LIBRARY_PATH:-}" ]; then
    export LD_LIBRARY_PATH="\${_LAPS_OLD_LD_LIBRARY_PATH}"
else
    unset LD_LIBRARY_PATH
fi
unset _LAPS_OLD_LD_LIBRARY_PATH
DEACTIVATE_EOF

echo "  Configured conda activate.d for CUDA_HOME and LD_LIBRARY_PATH"

# ── [6/6] Verify installation ───────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"

# Set LD_LIBRARY_PATH for verification (same as runtime)
export LD_LIBRARY_PATH="${CONDA_PREFIX_PATH}/lib:${CONDA_PREFIX_PATH}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

"${PYTHON}" -c "
import sglang;        print(f'  sglang:         {sglang.__version__}')
import mooncake;      print(f'  mooncake:        OK')
import sglang_router; print(f'  sglang_router:   OK')
import torch;         print(f'  torch:           {torch.__version__} (CUDA: {torch.cuda.is_available()})')
import sgl_kernel;    print(f'  sgl_kernel:      OK')
import flashinfer;    print(f'  flashinfer:      OK')
import transformers;  print(f'  transformers:    {transformers.__version__}')
try:
    import nixl;      print(f'  nixl:            OK')
except ImportError:   print(f'  nixl:            NOT installed (optional, needed without InfiniBand)')
"

echo ""
echo "============================================================"
echo "  Environment Info"
echo "============================================================"
if nvidia-smi &>/dev/null; then
    echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "  GPU count:     $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    echo "  Driver:        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
fi
"${PYTHON}" -c "import torch; print(f'  CUDA (torch):  {torch.version.cuda}')" 2>/dev/null || true
conda run -n "${ENV_NAME}" nvcc --version 2>/dev/null | grep release | sed 's/^/  nvcc:          /' || true

# Detect InfiniBand and recommend backend
HAS_IB=false
if command -v ibstat &>/dev/null && ibstat &>/dev/null 2>&1; then
    HAS_IB=true
    echo "  InfiniBand:    available (mooncake RDMA backend recommended)"
else
    echo "  InfiniBand:    NOT detected"
fi

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "  To activate the environment:"
echo ""
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  To run benchmarks (7B, 2 GPUs):"
echo ""
if [ "${HAS_IB}" = true ]; then
    echo "    bash bench_laps_prefill_throughput/bench_7b_2gpu.sh"
else
    echo "    # Your system has no InfiniBand. Use nixl backend instead of mooncake:"
    echo "    BACKEND=nixl bash bench_laps_prefill_throughput/bench_7b_2gpu.sh"
    echo ""
    echo "    # WARNING: Do NOT use 'MOONCAKE_PROTOCOL=tcp' — mooncake TCP mode"
    echo "    # causes ephemeral port exhaustion under high throughput and will"
    echo "    # fail after ~200-300 requests. Use nixl (UCX/shared memory) instead."
fi
echo ""
echo "  For more details, see:"
echo "    bench_laps_prefill_throughput/README.md"
echo "============================================================"
