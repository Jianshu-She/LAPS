# LAPS Artifact Evaluation (A100 Branch)

> **You are on the `a100-eval` branch.** This branch is for evaluating LAPS on **2x NVIDIA A100 GPUs** (40GB or 80GB). If you have 8x H200/H100 GPUs with InfiniBand, use the [`main` branch](https://github.com/Jianshu-She/LAPS/tree/main) instead.

This branch provides **one-click** artifact evaluation with all A100-specific fixes pre-applied, including support for LAPS Batch Prefill CUDA Graph on the FlashInfer attention backend (the main branch only supports Flash Attention 3 on Hopper GPUs).

## Quick Start

```bash
# 1. Clone and checkout
git clone https://github.com/Jianshu-She/LAPS.git
cd LAPS
git checkout a100-eval

# 2. Install (handles all A100-specific dependencies automatically)
bash install.sh
conda activate laps

# 3. Run all benchmarks (~75 min total)
bash bench_laps_prefill_throughput/run_a100_all.sh

# Or run a single model (~10 min)
bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 0.5b
```

## 1. Software Dependencies

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 2x NVIDIA A100 (40GB or 80GB) |
| NVIDIA Driver | >= 550 |
| CUDA (driver level) | >= 12.4 |
| System RAM | >= 32 GB |
| Disk | >= 30 GB free (for model weights and conda env) |

### Software Prerequisites

The following must be available **before** running `install.sh`:

| Dependency | Version | How to check | Notes |
|-----------|---------|-------------|-------|
| conda | any | `conda --version` | [Miniconda](https://docs.anaconda.com/miniconda/) or Anaconda |
| NVIDIA Driver | >= 550 | `nvidia-smi` | Must support CUDA 12.4+ |
| sudo access | - | `sudo echo ok` | Needed for `libibverbs` install; see manual install below if unavailable |

### Packages Installed by `install.sh`

`install.sh` creates a conda environment (`laps`) and installs everything automatically. Here is the full dependency list for reference:

**System libraries** (installed via `apt-get`/`yum`, requires sudo):

| Package | Version | Why |
|---------|---------|-----|
| `libibverbs1`, `libibverbs-dev` | any | mooncake links against `libibverbs.so.1` at import time, even when using nixl backend |
| `librdmacm1`, `librdmacm-dev` | any | RDMA connection manager, required by mooncake |

If sudo is unavailable, install manually or ask your sysadmin:
```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev

# RHEL/CentOS
sudo yum install -y libibverbs libibverbs-devel librdmacm librdmacm-devel
```

**Python environment** (conda, Python 3.12):

| Package | Version | Why |
|---------|---------|-----|
| Python | 3.12 | Required by SGLang |
| PyTorch | 2.9.1+cu128 | Deep learning framework |
| sgl-kernel | 0.3.21 | SGLang CUDA kernels |
| flashinfer | 0.6.3 | FlashInfer attention backend (used on A100) |
| transformers | 4.57.1 | HuggingFace model loading |
| mooncake-transfer-engine | latest | KV cache transfer (RDMA/TCP) |
| sglang-router | latest | Request routing for PD disaggregation |
| nixl | latest | KV cache transfer via UCX/shared memory (used instead of mooncake on non-IB systems) |
| matplotlib | latest | Plotting benchmark results |

**CUDA Toolkit** (installed via conda if system `nvcc` is missing):

| Package | Version | Why |
|---------|---------|-----|
| cuda-toolkit | matches PyTorch CUDA (12.8) | FlashInfer JIT-compiles CUDA kernels at runtime, requires `nvcc` |

**Post-install fixups** (handled automatically by `install.sh`):

| Fix | Why |
|-----|-----|
| `lib64/libcudart.so` symlink | FlashInfer's ninja build searches `$CONDA_PREFIX/lib64/` but conda puts libs in `lib/` |
| `lib64/stubs/libcuda.so` symlink | Same issue for CUDA driver stub library |
| Clear `~/.cache/flashinfer` | Force recompilation with correct CUDA paths |
| `conda activate.d/laps_env.sh` | Auto-set `CUDA_HOME` and `LD_LIBRARY_PATH` on `conda activate laps` |

### Models (downloaded automatically on first run)

| Model | HuggingFace ID | Size | Min VRAM/GPU |
|-------|---------------|------|-------------|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | ~1 GB | ~5 GB |
| Qwen2.5-3B | `Qwen/Qwen2.5-3B` | ~6 GB | ~10 GB |
| Qwen2.5-7B | `Qwen/Qwen2.5-7B` | ~14 GB | ~20 GB |

Models are downloaded from HuggingFace Hub to `~/.cache/huggingface/` on first run. Ensure network access or pre-download with:
```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B
huggingface-cli download Qwen/Qwen2.5-3B
huggingface-cli download Qwen/Qwen2.5-7B
```

## 2. Installation

```bash
bash install.sh
conda activate laps
```

The install script runs 6 steps:

| Step | What it does |
|------|-------------|
| [0/6] | Check/install system libraries (`libibverbs`, `librdmacm`) |
| [1/6] | Create conda environment with Python 3.12 |
| [2/6] | Check/install CUDA Toolkit (`nvcc`) via conda |
| [3/6] | Install SGLang (LAPS fork) in editable mode |
| [4/6] | Install mooncake, sglang-router, nixl |
| [5/6] | Create lib64 symlinks, clear flashinfer cache, configure `conda activate` env vars |
| [6/6] | Verify all imports work |

## 3. Dataset

The benchmarks use LMSYS-Chat-1M prompts at `data/lmsys_chat_10k.jsonl` (included in the repo). Scripts locate it automatically.

## 4. Run Benchmarks

### Run All Models (Recommended)

```bash
bash bench_laps_prefill_throughput/run_a100_all.sh
```

This runs 0.5B -> 3B -> 7B sequentially. Estimated total time: ~75 minutes.

### Run Individual Models

```bash
bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 0.5b   # ~10 min
bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 3b     # ~25 min
bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 7b     # ~40 min
```

### Customize

```bash
# Fewer prompts for a quick sanity check
NUM_PROMPTS=100 bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 0.5b

# Use mooncake backend (if you have InfiniBand)
BACKEND=mooncake bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 7b
```


## 5. Results

Each run creates a timestamped results directory:

```
results_7b_2gpu_a100_20260330_143022/
├── summary.txt          # Start here: req/s, TTFT tables, speedup
├── plots/               # PNG charts for each metric
│   ├── request_throughput_req_s.png
│   ├── mean_ttft_ms.png
│   └── p90_ttft_ms.png
└── raw/                 # Per-setting, per-concurrency JSON + logs
    ├── vanilla_sglang_cc1.json
    ├── laps_cc64.json
    └── ...
```

### Metrics

| Metric | Description |
|--------|-------------|
| `request_throughput_req_s` | Requests completed per second (higher is better) |
| `mean_ttft_ms` | Mean time-to-first-token in milliseconds (lower is better) |
| `p90_ttft_ms` | 90th percentile TTFT (lower is better) |

## 6. A100-Specific Notes

### Why nixl instead of mooncake?

The default KV cache transfer backend is `mooncake` with RDMA, which requires InfiniBand. On systems without InfiniBand, mooncake falls back to TCP, but this causes **ephemeral port exhaustion** under high throughput (~300 requests exhaust the ~28K port range). The `nixl` backend uses UCX with shared memory/CUDA IPC and avoids this issue entirely.

### FlashInfer Backend Support

The main branch only supports Batch Prefill CUDA Graph with the Flash Attention 3 backend (Hopper GPUs). This branch adds support for the **FlashInfer backend** (used by default on Ampere/A100 GPUs) by implementing the missing interface methods:

- `FlashInferAttnBackend.init_batch_prefill_cuda_graph_state()`
- Correct parameter passing for `init_forward_metadata_capture/replay_batch_prefill_cuda_graph()`
- Proper buffer pre-allocation with `max_context_len` to avoid flashinfer's `_max_total_num_rows` overflow

### Server Startup Order

The benchmark script starts the **decode server first**, waits for it to be healthy, then starts the **prefill server**. This is required because nixl's bootstrap mechanism needs the decode instance to be ready before the prefill instance can register.

### Environment Variables

`install.sh` configures `conda activate laps` to auto-set:
- `CUDA_HOME=$CONDA_PREFIX` — for flashinfer to find `nvcc`
- `LD_LIBRARY_PATH` — includes `$CONDA_PREFIX/lib` and nvidia cuda_runtime lib path

The benchmark script also auto-detects these if not set. No manual `export` needed.
