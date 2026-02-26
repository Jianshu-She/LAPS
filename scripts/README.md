# Scripts

Test and benchmark scripts for LAPS and batch prefill CUDA graph features.

## Test Scripts

### Correctness Tests

| Script | Description | Requirements |
|---|---|---|
| `test_bp_v2.py` | Quick smoke test for batch prefill CUDA graph v2. Launches an engine with attention-in-graph, runs single and batch requests, prints outputs. | 1 GPU |
| `test_bp_v2_compare.py` | Correctness validation: compares batch prefill v2 outputs against piecewise-only baseline using greedy decoding. Reports MATCH/MISMATCH per prompt. | 1 GPU (launches two engines sequentially) |
| `test_batch_prefill.py` | Sends concurrent short requests to a **running** server to verify batch prefill graph replay. Includes burst (8 requests) and sequential tests. Check server logs for `[BatchPrefill]` messages. | Running server on `localhost:30010` |

### End-to-End (PD Disaggregation) Tests

| Script | Description | Requirements |
|---|---|---|
| `test_bp_v2_disagg.py` | Full PD disaggregation test: launches prefill server (with batch prefill graph + mooncake), decode server, and router. Sends requests through the router and checks logs for graph capture/replay. | 2 GPUs, mooncake backend |
| `test_laps_dynamic_alloc.py` | Tests LAPS dynamic GPU allocation: launches 2 prefill servers + 1 decode server + router with `--enable-laps-alloc`. Sends mixed short/long requests and verifies `/laps_status` endpoint for group assignments and rebalancing. | 3 GPUs, mooncake backend |

## Benchmark Scripts

### Offline (Engine API) Benchmarks

| Script | Description | Requirements |
|---|---|---|
| `bench_bp_v2.py` | Latency microbenchmark: piecewise-only vs batch prefill v2 (attention-in-graph). Reports trimmed mean, p50, min, max across multiple batch sizes and prompt lengths. | 1 GPU |
| `bench_bp_v2_3way.py` | 3-way latency comparison: no CUDA graph vs piecewise vs attention-in-graph. Same methodology as `bench_bp_v2.py` but adds the no-graph baseline. | 1 GPU |

### Online (Server) Benchmarks

| Script | Description | Requirements |
|---|---|---|
| `benchmark_batch_prefill.py` | Comprehensive server-based benchmark: launches baseline and batch-prefill servers on separate GPUs, runs correctness checks, then measures concurrent request latency at bs=1/4/8. | 2 GPUs |
| `bench_pd_cuda_graph.sh` | PD disaggregation serving benchmark: launches prefill + decode + router via mooncake, runs `sglang.bench_serving` at multiple prompt lengths and request rates. Outputs throughput/TTFT/ITL summary. | 2 GPUs, mooncake backend |

### Prefill Throughput Benchmarks (Paper Results)

These are the benchmarks used to produce the results in the LAPS paper. Results are in `results/qwen2.5-32b-h200/`.

| Script | Description | Requirements |
|---|---|---|
| `bench_prefill_only.py` | Async prefill throughput client. Sends JSONL prompts to a router with `max_new_tokens=1`, measures TTFT and prefill throughput. Supports `--concurrency` (semaphore) and `--request-rate` (Poisson) modes. | Running PD disagg cluster |
| `bench_prefill_throughput.sh` | Runs `bench_prefill_only.py` under 3 CUDA graph settings (none / piecewise / batch-prefill) at fixed request rate + infinite rate. Uses Qwen2.5-0.5B. | 2 GPUs, mooncake backend |
| `bench_32b_concurrency.sh` | **Main paper benchmark.** Sweeps concurrency levels (1, 2, 4, 8, 16, 32, 64, 128) across 3 CUDA graph settings on **Qwen2.5-32B / H200**. Generates the throughput and TTFT tables in the README. | 2 GPUs (H200), mooncake backend |

## Usage

All scripts assume the model path is hardcoded. Edit the `MODEL_PATH` / `model_path` / `MODEL` variable at the top of each script before running.

```bash
# Quick correctness check
python scripts/test_bp_v2.py

# Output comparison against baseline
python scripts/test_bp_v2_compare.py

# Latency microbenchmark (3-way)
python scripts/bench_bp_v2_3way.py

# Full PD disaggregation test
python scripts/test_bp_v2_disagg.py

# Server-based benchmark (needs 2 GPUs)
python scripts/benchmark_batch_prefill.py --model-path /path/to/model

# LAPS dynamic allocation test (needs 3 GPUs)
python scripts/test_laps_dynamic_alloc.py

# Paper benchmark: Qwen2.5-32B concurrency sweep (needs 2x H200)
bash scripts/bench_32b_concurrency.sh
```
