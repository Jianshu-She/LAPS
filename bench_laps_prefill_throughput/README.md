# LAPS Artifact Evaluation (A100 Branch)

This branch provides **one-click** artifact evaluation on **2x NVIDIA A100 GPUs** (40GB or 80GB).

It includes fixes to enable LAPS Batch Prefill CUDA Graph on A100 GPUs via the FlashInfer attention backend (the main branch only supports Flash Attention 3 on Hopper GPUs).

## Quick Start

```bash
# 1. Install (handles all A100-specific dependencies automatically)
bash install.sh
conda activate laps

# 2. Run all benchmarks (~75 min total)
bash bench_laps_prefill_throughput/run_a100_all.sh

# Or run a single model (~10 min)
bash bench_laps_prefill_throughput/bench_a100_2gpu.sh 0.5b
```

## 1. Installation

```bash
bash install.sh
conda activate laps
```

The install script automatically handles all A100-specific dependencies:

| Step | What it does | Why |
|------|-------------|-----|
| System libs | Installs `libibverbs`, `librdmacm` | Required by mooncake at import time, even with nixl backend |
| CUDA Toolkit | Installs `nvcc` via conda if missing | FlashInfer JIT-compiles CUDA kernels at runtime |
| lib64 symlinks | Links `libcudart.so` to `lib64/` | FlashInfer's ninja build searches `$CONDA_PREFIX/lib64/` |
| nixl | `pip install nixl` | KV cache transfer backend (replaces mooncake RDMA on non-IB systems) |
| flashinfer cache | Clears `~/.cache/flashinfer` | Forces recompilation with correct CUDA paths |

**Troubleshooting**: If `install.sh` fails on system libraries (requires sudo), install them manually:
```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs1 libibverbs-dev librdmacm1 librdmacm-dev

# RHEL/CentOS
sudo yum install -y libibverbs libibverbs-devel librdmacm librdmacm-devel
```

## 2. Dataset

The benchmarks use LMSYS-Chat-1M prompts at `data/lmsys_chat_10k.jsonl` (included in the repo). Scripts locate it automatically.

## 3. Run Benchmarks

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

### What Each Setting Means

Each model is benchmarked under 3 settings:

| Setting | CLI Flags | Description |
|---------|-----------|-------------|
| **Vanilla SGLang** | (none) | Baseline: disaggregated PD serving without LAPS |
| **Dual-Queue** | `--enable-laps-scheduler` | LAPS Feature 2: length-aware dual-queue scheduling |
| **LAPS** | `--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph` | LAPS Feature 1: batch prefill CUDA graph (main speedup source) |

## 4. Results

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

## 5. GPU Requirements

| Model | Min VRAM per GPU | Notes |
|-------|-----------------|-------|
| Qwen2.5-0.5B | ~5 GB | Fastest, good for verifying setup |
| Qwen2.5-3B | ~10 GB | Medium |
| Qwen2.5-7B | ~20 GB | Matches paper's 2-GPU config |

All models use TP=1 with 1 GPU for prefill and 1 GPU for decode.

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
