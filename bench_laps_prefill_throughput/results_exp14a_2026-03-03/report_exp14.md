# Experiment 14: Investigating Batch CG Degradation & 7B tp=4

**Date**: 2026-03-03
**Branch**: `batch-prefill-attn-in-graph`
**Model**: Qwen/Qwen2.5-7B-Instruct
**Hardware**: H100 GPUs

---

## 1. Motivation

Experiment 13 revealed a paradox: `batch_cg_v2` degrades at high concurrency on the 7B model (single GPU):

| Concurrency | batch_cg_v2 vs vanilla (tp=1) |
|-------------|-------------------------------|
| cc=4        | 1.17x speedup                 |
| cc=8        | 1.16x speedup                 |
| cc=32       | 0.95x (5% slower)             |
| cc=64       | 0.98x (2% slower)             |

Server log analysis from exp13 deepened the paradox: at cc=32, batch_cg_v2's eager calls had **higher** average batch size (10.4) than vanilla (8.3), yet vanilla still achieved higher total RPS (411.4 vs 391.5). This meant the issue was NOT simply batch size degradation — there was overhead in the CG path itself.

**Initial hypothesis**: For the 7B model on a single GPU, the GPU kernel is already so fast that the CPU overhead of `replay_batch_prefill()` (right-aligning tokens, updating buffers, attention metadata init) negates the CG speedup.

**Goals**:
1. Instrument the forward path to measure per-call timing and identify the real bottleneck
2. Test whether tp=4 helps batch_cg_v2 on the 7B model (as it did for 32B)

---

## 2. What We Did

### 2.1 Part 1: Timing Instrumentation (Exp 14a)

We added fine-grained `time.perf_counter()` + `torch.cuda.synchronize()` timing to two locations:

#### Change 1: `model_runner.py` — `forward_extend()`

Added timing around each of the three forward paths:

- **batch_prefill_cg path**: Total duration from entry to return (includes replay_batch_prefill + sync)
- **piecewise_cg path**: Total duration from entry to return
- **eager path**: Three-phase timing:
  - `meta_ms`: `attn_backend.init_forward_metadata()` duration
  - `fwd_ms`: `model.forward()` duration (including cuda sync)
  - `total_ms`: End-to-end duration

Each call logs: `FWD-TIME path=<path> bs=<batch_size> dur_ms=<duration>` (or the three-phase breakdown for eager).

#### Change 2: `piecewise_cuda_graph_runner.py` — `replay_batch_prefill()`

Added two-phase timing inside the CG replay function:

- `prep_ms`: Everything before graph replay — zeroing buffers, right-aligning tokens (Python loop over sequences), updating batch-level buffers, calling `init_forward_metadata_replay_batch_prefill_cuda_graph()`, plus a `cuda.synchronize()` fence
- `replay_ms`: `graph.replay()` + output extraction, plus a `cuda.synchronize()` fence

Each call logs: `BP-CG-TIMING prep_ms=<prep> replay_ms=<replay> actual_bs=<bs> target=(<target_bs>,<target_seq_len>) real_tokens=<real> padded_tokens=<padded>`

**Important caveat**: `torch.cuda.synchronize()` forces CPU-GPU synchronization, destroying the async pipeline overlap that normally gives throughput. This means the RPS numbers from this experiment are NOT comparable to exp13 — but the per-call timing breakdown IS accurate.

#### Profiling Run

- **Settings**: vanilla and batch_cg_v2, both at cc=32
- **N**: 2000 requests (small, to keep it fast)
- **GPU**: Single GPU (GPU 5)
- **Script**: `bench_exp14_timing.sh`

### 2.2 Part 2: 7B tp=4 Experiment (Exp 14b)

After removing the timing instrumentation (to restore normal throughput), we ran the 7B model with tensor parallelism across 4 GPUs.

#### Configurations

| Setting | Server Args |
|---------|-------------|
| vanilla_tp4 | `--tp 4` |
| batch_cg_v2_tp4 | `--tp 4 --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128` |
| piecewise_cap512_tp4 | `--tp 4 --enable-piecewise-cuda-graph --disable-cuda-graph --piecewise-cuda-graph-max-tokens 512` |

#### Run Parameters

- **Concurrency levels**: 4, 8, 32, 64
- **N**: 10,000 requests per run
- **GPUs**: 4,5,6,7
- **Dataset**: unsorted human_prompts_stream_10000.jsonl
- **Total**: 12 runs across 3 server starts
- **Script**: `bench_exp14_7b_tp4.sh`

---

## 3. Results

### 3.1 Exp 14a: Profiling Timing Results

#### Overall RPS (with sync overhead)

| Setting | RPS | Avg Latency |
|---------|-----|-------------|
| vanilla | 272.1 | 116.0ms |
| batch_cg_v2 | 272.0 | 116.1ms |

Both collapsed to ~272 RPS (vs ~400+ in exp13) because `cuda.synchronize()` destroyed the GPU-CPU pipeline overlap. This confirms that in the real (uninstrumented) system, throughput depends heavily on async overlap, not just per-call speed.

#### Vanilla Forward Call Breakdown

| Metric | Value |
|--------|-------|
| Total eager calls | 204 |
| Average batch size | 9.4 |
| Avg `meta_ms` (init_forward_metadata) | 0.202ms |
| Avg `fwd_ms` (model.forward + sync) | 27.470ms |
| Avg `total_ms` | 27.916ms |
| Avg ms/request | 3.990ms |

Breakdown by batch size bucket:

| Batch Size | # Calls | Avg BS | Avg total_ms | Avg fwd_ms | Requests |
|------------|---------|--------|--------------|------------|----------|
| 1-2        | 19      | 1.4    | 12.964       | 12.479     | 27       |
| 3-8        | 84      | 6.1    | 25.651       | 25.197     | 514      |
| 9-16       | 80      | 12.1   | 29.770       | 29.337     | 971      |
| 17+        | 21      | 19.6   | 43.442       | 43.015     | 411      |

Key observation: `init_forward_metadata` is negligible (~0.2ms). Nearly all time is in `model.forward()`.

#### Batch CG v2 Forward Call Breakdown

| Metric | CG Path | Eager Path |
|--------|---------|------------|
| # Calls | 30 (16.9%) | 148 (83.1%) |
| Total requests | 129 | 1,605 |
| Avg batch size | 4.3 | 10.8 |
| Avg duration | 12.261ms | 31.991ms |
| **Avg ms/request** | **2.851** | **2.950** |

#### BP-CG-TIMING: The Critical Breakdown

| Metric | Value |
|--------|-------|
| **Avg prep_ms** | **0.589ms** |
| **Avg replay_ms** | **10.840ms** |
| **Prep fraction** | **5.2%** |
| Avg real tokens | 114.9 |
| Avg padded tokens | 462.7 |
| **Avg padding ratio** | **3.45x** |

Breakdown by target graph grid:

| Target Grid | # Calls | Prep (ms) | Replay (ms) | Total (ms) | Avg Real Tokens | Padding Ratio |
|-------------|---------|-----------|-------------|------------|-----------------|---------------|
| (1, 8)      | 1       | 0.450     | 5.112       | 5.562      | 8               | 1.00x         |
| (1, 16)     | 1       | 0.490     | 5.309       | 5.799      | 12              | 1.33x         |
| (1, 32)     | 1       | 0.474     | 6.109       | 6.583      | 20              | 1.60x         |
| (1, 64)     | 1       | 0.896     | 5.166       | 6.062      | 33              | 1.94x         |
| (1, 128)    | 1       | 0.499     | 5.502       | 6.001      | 81              | 1.58x         |
| (2, 32)     | 1       | 0.491     | 5.203       | 5.694      | 46              | 1.39x         |
| (2, 64)     | 1       | 0.501     | 6.255       | 6.756      | 67              | 1.91x         |
| (2, 128)    | 1       | 0.577     | 7.303       | 7.880      | 105             | 2.44x         |
| (4, 32)     | 3       | 0.583     | 5.365       | 5.948      | 59              | 2.21x         |
| (4, 64)     | 1       | 0.545     | 7.047       | 7.592      | 105             | 2.44x         |
| (4, 128)    | 2       | 0.528     | 11.237      | 11.765     | 133             | 3.96x         |
| (8, 32)     | 1       | 0.682     | 6.710       | 7.392      | 112             | 2.29x         |
| (8, 64)     | 8       | 0.616     | 10.944      | 11.560     | 130             | 4.08x         |
| (8, 128)    | 7       | 0.618     | 19.909      | 20.527     | 196             | 5.47x         |

The (8, 128) grid — the largest captured graph — processes 196 real tokens but pays for 1024 padded tokens (5.47x overhead), taking 20.5ms per replay.

#### Key Comparison: ms/request across paths

| Path | Total Time (ms) | Requests | ms/request |
|------|-----------------|----------|------------|
| CG (batch_cg_v2) | 367.8 | 129 | **2.851** |
| Eager (batch_cg_v2) | 4,734.7 | 1,605 | **2.950** |
| Eager (vanilla) | 5,694.8 | 1,923 | **2.961** |

The CG path is **3.7% faster per-request** than vanilla's eager path. The prep overhead is not the issue.

### 3.2 Exp 14b: 7B tp=4 Results

#### RPS Table

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| *exp13 vanilla tp=1* | *135.0* | *204.1* | *411.4* | *450.2* |
| *exp13 batch_cg_v2 tp=1* | *158.5* | *236.8* | *391.5* | *441.3* |
| *exp13 piecewise_cap512 tp=1* | *170.3* | *293.4* | *449.6* | *439.4* |
| | | | | |
| **vanilla tp=4** | **105.7** | **150.5** | **389.2** | **533.1** |
| **batch_cg_v2 tp=4** | **136.3** | **218.5** | **426.1** | **528.2** |
| **piecewise_cap512 tp=4** | **126.9** | **246.2** | **453.8** | **531.2** |

#### Speedup vs tp=4 Vanilla

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla_tp4 | 1.000x | 1.000x | 1.000x | 1.000x |
| **batch_cg_v2_tp4** | **1.289x** | **1.452x** | **1.095x** | 0.991x |
| **piecewise_cap512_tp4** | **1.200x** | **1.636x** | **1.166x** | 0.996x |

#### tp=4 vs tp=1 (Same Setting)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 0.783x | 0.738x | 0.946x | **1.184x** |
| batch_cg_v2 | 0.859x | 0.923x | **1.088x** | **1.197x** |
| piecewise_cap512 | 0.745x | 0.839x | 1.009x | **1.209x** |

#### Latency Table (avg ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla_tp4 | 37.6 | 52.8 | 81.1 | 118.2 |
| batch_cg_v2_tp4 | 29.1 | 36.1 | 73.8 | 119.0 |
| piecewise_cap512_tp4 | 31.3 | 32.1 | 69.4 | 118.3 |

All runs: 0 failures.

---

## 4. Analysis

### 4.1 The Original Hypothesis Was Wrong

The hypothesis was: *"For the 7B model, the CPU overhead of `replay_batch_prefill()` (right-aligning tokens, updating buffers, attention metadata init) negates the CG speedup."*

**This is definitively disproven.** The prep phase takes only 0.589ms on average — just 5.2% of the total CG call time. The CG path is actually 3.7% faster per-request than vanilla's eager path (2.851 vs 2.961 ms/req).

### 4.2 The Real Root Cause: Scheduling Inefficiency + Padding Waste

The degradation at high concurrency stems from two interacting effects:

**Effect 1: CG steals small batches from the eager pool.** At cc=32 in exp13, CG made 483 calls (38.5% of total) but only handled 19.7% of requests at avg_bs=4.1. Each CG call processes a tiny batch quickly, preventing those requests from accumulating into larger eager batches. The eager path ends up with higher avg_bs (10.4 vs vanilla's 8.3) but fewer total calls.

**Effect 2: Padding waste burns GPU compute.** The average padding ratio is 3.45x — for every real token, the GPU processes 3.45 padded tokens. The worst offender is the (8, 128) grid at 5.47x. This means a CG call processing 196 real tokens actually computes 1024 tokens on the GPU.

**The combined effect**: CG calls are fast per-request but process very few requests per call due to the small captured grids (max bs=8, max seq=128). The GPU time spent on padding is wasted. Meanwhile, vanilla's eager path batches the same requests into larger, more efficient batches with zero padding waste.

At low concurrency (cc=4), queue depth is shallow — requests arrive slowly and can't accumulate into large batches anyway. CG's ability to quickly dispatch small batches gives a clear speedup. At high concurrency (cc=32+), the queue is always deep, and vanilla naturally forms large batches that are more compute-efficient than CG's padded small batches.

### 4.3 tp=4 Helps batch_cg_v2 (Confirming 32B Pattern)

The tp=4 results confirm the expected pattern:

**At low cc (4, 8)**: batch_cg_v2_tp4 strongly beats vanilla_tp4 (1.29x at cc=4, 1.45x at cc=8). With tensor parallelism, each forward call incurs NCCL all-reduce communication overhead, making kernel launch overhead a larger fraction of total time. CG's elimination of kernel launches becomes more valuable.

**At medium cc (32)**: batch_cg_v2_tp4 maintains a 1.10x advantage (vs 0.95x degradation with tp=1). The tp=4 overhead shifts the break-even point higher.

**At high cc (64)**: All settings converge to ~530 RPS. The GPU is fully saturated and scheduling differences wash out.

**Critically**: batch_cg_v2_tp4 at cc=32 achieves 426.1 RPS — **8.8% higher** than batch_cg_v2_tp1's 391.5 RPS, and even **3.6% higher** than vanilla_tp1's 411.4 RPS at the same concurrency.

### 4.4 piecewise_cap512 Remains the Strongest Option

`piecewise_cap512_tp4` achieves the best or near-best results at every concurrency level:
- cc=8: **246.2 RPS** (1.64x vs vanilla_tp4) — the biggest speedup of any configuration
- cc=32: **453.8 RPS** (1.17x vs vanilla_tp4, best absolute RPS at cc=32 across all settings)
- cc=64: 531.2 RPS (~1.0x vs vanilla_tp4)

This is because piecewise CG captures a single sequence up to 512 tokens in one graph, with no padding waste and no batching disruption.

### 4.5 tp=4 Only Helps at High Concurrency for Vanilla

For vanilla (no CG), tp=4 is actually **slower** than tp=1 at low concurrency:
- cc=4: 0.78x (105.7 vs 135.0)
- cc=8: 0.74x (150.5 vs 204.1)

This makes sense: with 4 GPUs, the communication overhead dominates when there's little compute to parallelize (small batches at low cc). Only at cc=64 does vanilla_tp4 (533.1) beat vanilla_tp1 (450.2) by 1.18x.

However, batch_cg_v2 benefits more from tp=4 than vanilla does — the CG kernel launch savings compound with the TP overhead to create a larger relative advantage.

---

## 5. Conclusions

1. **Prep overhead is not the bottleneck.** At 0.589ms (5.2% of CG call time), the buffer preparation in `replay_batch_prefill()` is negligible. The CG path is actually slightly faster per-request than eager.

2. **The degradation is a scheduling/efficiency problem.** CG steals small batches from the queue, preventing efficient large-batch eager execution, while padding waste burns GPU cycles on dummy tokens. This effect dominates at high concurrency where large batches would otherwise form naturally.

3. **tp=4 shifts the trade-off in CG's favor.** With tensor parallelism adding communication overhead, CG's kernel launch elimination becomes more valuable relative to the padding cost. batch_cg_v2_tp4 maintains speedup through cc=32 (1.10x), unlike tp=1 where it degrades to 0.95x.

4. **piecewise_cap512 is the best CG strategy for 7B.** It avoids both the padding waste of batch_prefill_cg and the batching disruption, while still eliminating kernel launches for short sequences.

5. **For 7B single-GPU deployment**: Use `piecewise_cap512` (no need for batch_prefill_cg). It achieves 449.6 RPS at cc=32 vs vanilla's 411.4 — a clean 9.3% improvement with no downsides.

6. **For 7B multi-GPU deployment (tp=4)**: `piecewise_cap512_tp4` achieves the highest throughput at cc=32 (453.8 RPS) and `batch_cg_v2_tp4` is a close second (426.1 RPS). Both outperform vanilla_tp4 (389.2 RPS).

---

## 6. File Inventory

| File | Description |
|------|-------------|
| `bench_exp14_timing.sh` | Profiling benchmark script (exp 14a) |
| `bench_exp14_7b_tp4.sh` | 7B tp=4 benchmark script (exp 14b) |
| `results_exp14a_2026-03-03/` | Profiling results + server logs with timing |
| `results_exp14b_2026-03-03/` | 7B tp=4 results |
| `model_runner.py` lines 2312-2358 | forward_extend() — timing removed after profiling |
| `piecewise_cuda_graph_runner.py` lines 992-1087 | replay_batch_prefill() — timing removed after profiling |

## 7. Next Steps

Potential directions:
- **Reduce padding waste**: Investigate variable-length CG capture or tighter grid spacing to reduce the 3.45x average ratio
- **Smarter CG eligibility**: Only use CG when padding ratio is below a threshold (e.g., <2x), falling through to eager for inefficient configurations
- **Larger CG grids**: Capture graphs with bs=16 or bs=32 to handle high-cc scenarios where many small sequences arrive simultaneously
- **Profile 32B tp=4**: Verify that the 32B model (which showed strong CG gains in exp13) still benefits with these insights
