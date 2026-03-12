# LAPS Artifact Evaluation

This directory contains everything needed to reproduce the LAPS benchmark results.

## 1. Installation

From the **repo root**, run the install script to set up the environment:

```bash
bash install.sh
conda activate laps
```

This creates a conda environment with SGLang (LAPS fork), mooncake-transfer-engine, and sglang-router.

**Prerequisites:**
- conda (miniconda or anaconda)
- NVIDIA GPUs with CUDA drivers
- InfiniBand device (for mooncake KV transfer)

## 2. Dataset

The benchmarks use LMSYS-Chat-1M prompts. The dataset file is at `data/lmsys_chat_10k.jsonl` (relative to repo root). The scripts locate it automatically.

## 3. Run Benchmarks

Each benchmark launches a PD disaggregation cluster (prefill server + decode server + router), runs 3 settings (Vanilla SGLang, Disaggregation, LAPS) across 7 concurrency levels, and generates a `summary.txt` with throughput/latency tables.

### Quick Evaluation (One Command)

Run all 4 model sizes sequentially:

```bash
bash run_all_benchmarks.sh
```

This runs: 7B (2 GPU) → 14B (8 GPU) → 32B (8 GPU) → 72B (8 GPU).

You can also select specific models:

```bash
bash run_all_benchmarks.sh 7b          # only 7B
bash run_all_benchmarks.sh 14b 32b     # only 14B and 32B
```

### Run Individual Models

| Script | Model | GPUs Required | TP Size |
|---|---|---|---|
| `bench_7b_2gpu.sh` | Qwen2.5-7B | 2 | 1 |
| `bench_14b_8gpu.sh` | Qwen2.5-14B | 8 | 4 |
| `bench_32b_8gpu.sh` | Qwen2.5-32B-Instruct | 8 | 4 |
| `bench_72b_8gpu.sh` | Qwen2.5-72B-Instruct | 8 | 4 |

Example:

```bash
bash bench_7b_2gpu.sh     # ~2 GPUs, fastest to run
bash bench_72b_8gpu.sh    # ~8 GPUs, largest model
```

## 4. Results

Each run creates a timestamped results directory:

```
results_7b_2gpu_20260312_143022/
├── summary.txt          # <-- Start here: throughput, latency, speedup tables
└── raw/                 # Per-setting, per-concurrency JSON/TXT + server logs
    ├── vanilla_sglang_cc1.json
    ├── vanilla_sglang_cc1.txt
    ├── laps_cc64.json
    ├── ...
    └── laps_prefill.log
```

The `summary.txt` contains:
- Request throughput (req/s) across all concurrency levels
- Prefill throughput (tok/s)
- Mean and P99 TTFT latency (ms)
- Speedup vs Vanilla SGLang

## 5. What Each Setting Tests

| Setting | Description | Flags |
|---|---|---|
| `vanilla_sglang` | Baseline SGLang with PD disaggregation, no scheduling optimizations | (none) |
| `disaggregation` | + LAPS dual-queue scheduler (separates short/long prefills) | `--enable-laps-scheduler` |
| `laps` | + Piecewise CUDA graph + Batch prefill CUDA graph + LAPS scheduler | `--enable-laps` |
