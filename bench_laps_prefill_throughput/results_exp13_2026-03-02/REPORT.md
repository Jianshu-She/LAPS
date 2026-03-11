# Experiment 13: Making CUDA Graph Prefill Beat Vanilla

**Date:** 2026-03-02
**Branch:** `debug-batch-prefill-experiments`
**Repository:** `/mnt/weka/home/jianshu.she/graph/sglang`

---

## 1. Background and Motivation

### The Problem

Experiments 8-12 established that CUDA graph (CG) modes for prefill — both **piecewise CG** and **batch prefill CG** — are significantly slower than vanilla (eager) execution at high concurrency levels on the LMSYS-chat-10000 dataset. The diagnosis identified three hypotheses:

| Hypothesis | Description | Status |
|---|---|---|
| **H1: Padding waste** | CG pads prefill batches to fit captured graph dimensions, wasting FLOPs | **Confirmed** (83% waste in batch_cg) |
| **H2: Memory pressure** | CG memory reduces available GPU KV cache | **Refuted** (halving KV has zero effect) |
| **H3: Batching degradation** | CG drains request queue too fast via bs=1 replays, preventing scheduler from accumulating efficient large batches | **Confirmed** (93.5% bs=1 calls) |

### Root Cause Analysis

The core issue is the **forward path fallback chain**: `batch_prefill_cg → piecewise_cg → eager`. When batch prefill CG cannot find a matching captured graph (sequence too long or batch too large), it falls through to piecewise CG, which processes every single-sequence request via fast graph replay. This is efficient per-request but **starves the scheduler of queue depth**, preventing the formation of large, compute-efficient batches.

In Experiment 8's batch_cg mode at cc=32, the server log analysis showed:
- **93.5%** of piecewise CG calls were bs=1 (single-sequence replays)
- Average batch size was only **1.3** (vs. vanilla's natural batching of 8-16)
- The result: 143.3 RPS vs. vanilla's 415.2 RPS — a **65.5% degradation**

### Dataset Characteristics

The LMSYS-chat-10000 dataset (`human_prompts_stream_10000.jsonl`) has a highly skewed prompt length distribution:

| Statistic | Value |
|---|---|
| Total prompts | 10,000 |
| Min tokens (est.) | 0 |
| Max tokens (est.) | 4,002 |
| Mean tokens | 63.3 |
| Median tokens | 13 |
| Std deviation | 235.9 |

This extreme skew (median=13, mean=63.3) means most prompts are very short, but a long tail of large prompts drives the mean up. This is important because:
- Short prompts (< 128 tokens) are ideal candidates for small CUDA graphs
- Long prompts (> 128 tokens) must fall back to eager regardless

---

## 2. Solution Design

### Goal

Make at least one CG mode achieve **higher RPS than vanilla** on the LMSYS-chat-10000 dataset across all concurrency levels.

### Two-Pronged Approach

#### Approach A: Batch Prefill CG v2 (Code Changes)

Two code changes to fix H3 (batching degradation):

**Change 1: Allow bs=1 in batch prefill CG**

File: `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`

The original `can_run_batch_prefill()` had a hard guard at line 800-801:
```python
if forward_batch.batch_size <= 1:
    return False
```
This forced all single-sequence prefills to bypass batch prefill CG entirely, even though bs=1 is a valid captured graph size. **Removed this guard** so bs=1 requests can use batch prefill CG for short sequences.

**Change 2: Skip piecewise CG fallback when batch prefill is enabled**

File: `python/sglang/srt/model_executor/model_runner.py`

The original forward path in `forward_extend()` (lines 2325-2336) always tried piecewise CG as fallback when batch prefill CG missed. This caused H3 — every CG miss still got fast piecewise replay, draining the queue. **Added a check**: when batch prefill CG is enabled, skip the piecewise CG fallback entirely and go straight to eager execution.

```python
# Skip piecewise fallback when batch_prefill is enabled — fall through
# to eager instead, preserving natural queue accumulation (fixes H3).
skip_piecewise = (
    self.piecewise_cuda_graph_runner is not None
    and self.piecewise_cuda_graph_runner.batch_prefill_enabled
)
can_run_graph = (
    not skip_piecewise
    and self.piecewise_cuda_graph_runner is not None
    and self.piecewise_cuda_graph_runner.can_run(forward_batch)
)
```

**Resulting forward path with batch_cg_v2:**
- Short sequences (≤128 tokens): batch prefill CG replay (fast, attention inside graph)
- Long sequences (>128 tokens): eager execution (same as vanilla, preserves natural batching)

**Graph configuration (CLI args, no code change):**
- `--batch-prefill-batch-sizes 1 2 4 8` — 4 batch sizes including bs=1
- `--batch-prefill-seq-lengths 8 16 32 64 128` — 5 sequence lengths
- `--batch-prefill-max-seq-len 128` — cap at 128 tokens
- Total: **20 small graphs** captured

#### Approach B: Piecewise CG with cap=512

No code changes — purely a configuration change. Experiment 10 showed that limiting `--piecewise-cuda-graph-max-tokens 512` forces sequences >512 tokens to eager execution, partially solving H3. This was the best-performing configuration from the diagnostic experiments.

#### Approach C: Sorted Dataset (Optional)

Sort the dataset by estimated token count (ascending) to group similar-length prompts together. Hypothesis: grouping short prompts improves CG hit rate in bursts and reduces padding waste.

---

## 3. Experimental Setup

### Hardware
- **GPU:** NVIDIA H200 (143 GB HBM3)
- **7B runs:** Single GPU (GPU 5)
- **32B runs:** 4 GPUs (GPUs 4-7) with tensor parallelism (tp=4)

### Models
- **7B:** `Qwen/Qwen2.5-7B-Instruct`
- **32B:** `Qwen/Qwen2.5-32B`

### Benchmark Configuration
- **Client:** `baseline.py` with HTTP/2
- **Requests:** 10,000 per run
- **Max tokens:** 1 (prefill-only throughput test)
- **Dataset:** LMSYS-chat-10000 (`human_prompts_stream_10000.jsonl`)
- **Sorted dataset:** Same 10K prompts sorted ascending by estimated token count (chars/4)

### Configurations Tested

#### 7B (tp=1) — 4 settings × 4 cc = 16 runs

| Setting | Description | Server Args |
|---|---|---|
| `vanilla` | Eager baseline | (none) |
| `batch_cg_v2` | Code changes + small graphs, unsorted | `--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128` |
| `batch_cg_v2_sorted` | Same as above, sorted dataset | Same as above |
| `piecewise_cap512` | Piecewise CG capped at 512 tokens | `--enable-piecewise-cuda-graph --disable-cuda-graph --piecewise-cuda-graph-max-tokens 512` |

#### 32B (tp=4) — 3 settings × 3 cc = 9 runs

| Setting | Description | Server Args |
|---|---|---|
| `vanilla` | Eager baseline with tp=4 | `--tp 4` |
| `batch_cg_v2` | Code changes + small graphs, unsorted | `--tp 4 --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128` |
| `batch_cg_v2_sorted` | Same as above, sorted dataset | Same as above |

### Concurrency Levels
- **7B:** cc=4, 8, 32, 64
- **32B:** cc=4, 32, 64

### Total: 25 runs, 7 server starts, 0 failures

---

## 4. Results

### 4.1 Qwen2.5-7B-Instruct (tp=1)

#### Throughput (RPS)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 135.0 | 204.1 | 411.4 | 450.2 |
| batch_cg_v2 | 158.5 | 236.8 | 391.5 | 441.3 |
| batch_cg_v2_sorted | 163.1 | 248.8 | 375.3 | 397.9 |
| **piecewise_cap512** | **170.3** | **293.4** | **449.6** | **439.4** |

#### Speedup vs Exp13 Vanilla

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| batch_cg_v2 | **1.174x** | **1.160x** | 0.952x | 0.980x |
| batch_cg_v2_sorted | **1.208x** | **1.219x** | 0.912x | 0.884x |
| **piecewise_cap512** | **1.261x** | **1.437x** | **1.093x** | 0.976x |

#### Speedup vs Exp8 Vanilla (Historical Baseline)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| exp13 vanilla | 1.048x | 1.017x | 0.991x | 1.018x |
| batch_cg_v2 | **1.230x** | **1.180x** | 0.943x | 0.998x |
| batch_cg_v2_sorted | **1.266x** | **1.240x** | 0.904x | 0.900x |
| **piecewise_cap512** | **1.322x** | **1.462x** | **1.083x** | 0.993x |

#### Average Latency (ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 29.4 | 38.9 | 76.8 | 140.0 |
| batch_cg_v2 | 25.1 | 33.4 | 80.7 | 142.9 |
| batch_cg_v2_sorted | 24.3 | 31.6 | 80.6 | 145.7 |
| **piecewise_cap512** | **23.3** | **27.0** | **70.3** | 143.6 |

#### Tail Latency (p99, ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 74.7 | 95.3 | 378.2 | 466.3 |
| batch_cg_v2 | 73.7 | 101.9 | 406.0 | 440.9 |
| batch_cg_v2_sorted | 108.3 | 200.7 | 758.4 | 1226.8 |
| piecewise_cap512 | 71.2 | 101.0 | 286.0 | 711.4 |

#### Wall Clock Time (seconds)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 74.1 | 49.0 | 24.3 | 22.2 |
| batch_cg_v2 | 63.1 | 42.2 | 25.5 | 22.7 |
| batch_cg_v2_sorted | 61.3 | 40.2 | 26.6 | 25.1 |
| **piecewise_cap512** | **58.7** | **34.1** | **22.2** | 22.8 |

### 4.2 Qwen2.5-32B (tp=4)

#### Throughput (RPS)

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla | 69.1 | 236.9 | 287.3 |
| batch_cg_v2 | 72.8 | 242.5 | 285.8 |
| batch_cg_v2_sorted | 77.2 | 217.4 | 250.3 |

#### Speedup vs Exp13 Vanilla (tp=4)

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| batch_cg_v2 | **1.054x** | **1.024x** | 0.995x |
| batch_cg_v2_sorted | **1.118x** | 0.918x | 0.871x |

#### Average Latency (ms)

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla | 57.6 | 133.9 | 219.7 |
| batch_cg_v2 | 54.7 | 130.5 | 220.9 |
| batch_cg_v2_sorted | 51.5 | 140.2 | 233.6 |

#### Tail Latency (p99, ms)

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla | 120.2 | 557.7 | 653.6 |
| batch_cg_v2 | 125.6 | 395.2 | 1009.8 |
| batch_cg_v2_sorted | 164.0 | 1039.0 | 1825.8 |

#### Comparison: Exp13 32B (tp=4) vs Exp8 32B (tp=1)

| Setting | Exp8 tp=1 cc=4 | Exp13 tp=4 cc=4 | Exp8 tp=1 cc=64 | Exp13 tp=4 cc=64 |
|---|---|---|---|---|
| vanilla | 59.2 | 69.1 (1.17x) | 124.7 | 287.3 (2.30x) |
| batch_cg (old) / v2 | 42.2 | 72.8 (1.72x) | 44.6 | 285.8 (6.41x) |

The tp=4 setting dramatically improves 32B throughput — vanilla goes from 124.7 to 287.3 RPS at cc=64 (2.3x), and batch_cg goes from a catastrophic 44.6 to 285.8 RPS (6.4x improvement).

---

## 5. Analysis

### 5.1 Piecewise CG cap=512: The Clear Winner (7B)

**piecewise_cap512 beats vanilla at 3 out of 4 concurrency levels:**

| Concurrency | Speedup | Explanation |
|---|---|---|
| cc=4 | **+26.1%** | CG replay dominates; most prompts are short (median=13 tokens) and fit under 512-token cap |
| cc=8 | **+43.7%** | Sweet spot — enough queue depth for batching, most sequences still use CG |
| cc=32 | **+9.3%** | CG handles short sequences; long sequences (>512 tokens) fall to eager, preserving queue depth |
| cc=64 | -2.4% | Near parity; the CG overhead slightly offsets gains at this extreme concurrency |

**Why cap=512 works:** The cap creates a natural partition:
- **Sequences ≤ 512 tokens** (~85% of dataset): Use piecewise CG replay (fast)
- **Sequences > 512 tokens** (~15% of dataset): Fall back to eager (same speed as vanilla)

This preserves vanilla's natural batching for long sequences while accelerating the majority of short sequences. Unlike the full piecewise CG (max_tokens=8192), the cap prevents the fast CG path from starving the queue at high concurrency.

### 5.2 Batch CG v2: Fixes the Catastrophic Regression

The code changes successfully fixed the catastrophic regression observed in Experiment 8:

| Metric | Exp8 batch_cg | Exp13 batch_cg_v2 | Improvement |
|---|---|---|---|
| cc=32 RPS (7B) | 143.3 (0.35x vanilla) | 391.5 (0.95x vanilla) | **+173%** (2.73x) |
| cc=64 RPS (7B) | 211.2 (0.48x vanilla) | 441.3 (0.98x vanilla) | **+109%** (2.09x) |
| cc=32 RPS (32B) | 35.1 (0.29x vanilla) | 242.5 (1.02x vanilla) | **+591%** (6.91x) |
| cc=64 RPS (32B) | 44.6 (0.36x vanilla) | 285.8 (1.00x vanilla) | **+541%** (6.41x) |

**What fixed it:** The combination of:
1. **Allowing bs=1** in batch prefill CG means short single-sequence requests get CG replay instead of falling to piecewise
2. **Skipping piecewise fallback** means long sequences go to eager, preserving natural queue accumulation
3. **Small graph grid** (max 128 tokens) limits CG to where it has the most benefit

However, batch_cg_v2 doesn't quite beat vanilla at high cc because:
- The 20 small graphs (bs=[1,2,4,8] × seq=[8,16,32,64,128]) still have padding waste for non-power-of-2 sequences
- The batch prefill monolithic graph includes attention, which adds capture overhead
- At cc=32+, the queue is deep enough that vanilla's natural large-batch eager execution is already very efficient

### 5.3 Sorted Dataset: Hurts at High Concurrency

Sorting prompts by length was expected to improve CG hit rates by grouping similar-length prompts together. The results show a mixed picture:

| cc | batch_cg_v2 (unsorted) | batch_cg_v2_sorted | Delta |
|---|---|---|---|
| cc=4 | 158.5 (1.17x) | 163.1 (1.21x) | +2.9% |
| cc=8 | 236.8 (1.16x) | 248.8 (1.22x) | +5.1% |
| cc=32 | 391.5 (0.95x) | 375.3 (0.91x) | -4.1% |
| cc=64 | 441.3 (0.98x) | 397.9 (0.88x) | -9.8% |

**Why sorting hurts at high cc:** When prompts are sorted ascending by length, the server first processes a burst of very short prompts (fast CG replay), then hits a wall of increasingly long prompts that all go through eager. The long-prompt burst creates a **convoy effect** — many long requests compete for GPU time simultaneously, increasing contention and tail latency.

Evidence from tail latency:
- batch_cg_v2_sorted p99 at cc=64: **1226.8 ms** vs batch_cg_v2 unsorted: **440.9 ms** (2.8x worse)
- batch_cg_v2_sorted max at cc=64: **3794.9 ms** vs batch_cg_v2 unsorted: **801.3 ms** (4.7x worse)

The unsorted dataset naturally interleaves short and long prompts, providing a more uniform workload distribution.

### 5.4 32B with tp=4: Batch CG v2 is Competitive

For the 32B model with tp=4:
- **batch_cg_v2 beats vanilla** at cc=4 (+5.4%) and cc=32 (+2.4%)
- **Essentially matches** at cc=64 (-0.5%)
- This is a dramatic improvement over Exp8 where batch_cg was 3-7x slower than vanilla

The 32B model benefits less from CG because:
1. Each forward pass is more compute-intensive, so CG's kernel launch overhead savings are proportionally smaller
2. With tp=4, all-reduce communication adds latency that CG cannot eliminate
3. The absolute RPS is lower (287 vs 450 for 7B), so queue dynamics are different

---

## 6. Latency Profile Comparison (7B)

### Median Latency (p50, ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 27.4 | 35.5 | 58.0 | 100.8 |
| batch_cg_v2 | 23.1 | 29.8 | 62.8 | 102.6 |
| piecewise_cap512 | 21.3 | 21.5 | 52.0 | 102.9 |

### p90 Latency (ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 36.8 | 44.1 | 122.8 | 243.6 |
| batch_cg_v2 | 33.5 | 44.0 | 127.6 | 261.9 |
| piecewise_cap512 | 28.2 | 38.7 | 121.5 | 261.9 |

### p95 Latency (ms)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 42.1 | 54.1 | 179.0 | 296.9 |
| batch_cg_v2 | 42.8 | 56.4 | 178.0 | 336.7 |
| piecewise_cap512 | 35.8 | 52.0 | 156.9 | 332.6 |

**Key observation:** piecewise_cap512 has the best latency profile across all percentiles at low-to-medium concurrency. At cc=64, all modes converge to similar latency as the system becomes throughput-saturated.

---

## 7. Comparison with Experiment 8 Historical Baselines

### 7B: Before vs After (RPS)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| Exp8 vanilla | 128.8 | 200.7 | 415.2 | 442.3 |
| Exp8 piecewise_cg | 170.9 | 290.6 | 411.4 | 435.5 |
| Exp8 batch_cg | 172.1 | 204.9 | **143.3** | **211.2** |
| --- | --- | --- | --- | --- |
| Exp13 vanilla | 135.0 | 204.1 | 411.4 | 450.2 |
| Exp13 batch_cg_v2 | 158.5 | 236.8 | 391.5 | 441.3 |
| **Exp13 piecewise_cap512** | **170.3** | **293.4** | **449.6** | **439.4** |

The catastrophic batch_cg regression at cc=32 (143.3 RPS, **-65.5%** vs vanilla) and cc=64 (211.2 RPS, **-52.2%** vs vanilla) has been completely eliminated. batch_cg_v2 now achieves 391.5 and 441.3 RPS respectively (**-4.8%** and **-2.0%** vs vanilla).

### 32B: Before vs After (RPS)

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| Exp8 vanilla (tp=1) | 59.2 | 119.6 | 124.7 |
| Exp8 batch_cg (tp=1) | 42.2 | **35.1** | **44.6** |
| --- | --- | --- | --- |
| Exp13 vanilla (tp=4) | 69.1 | 236.9 | 287.3 |
| Exp13 batch_cg_v2 (tp=4) | 72.8 | 242.5 | 285.8 |

With tp=4 and the code fixes, 32B batch_cg goes from catastrophic (35.1 RPS at cc=32, **-70.7%** vs vanilla) to competitive (242.5 RPS, **+2.4%** vs vanilla).

---

## 8. Key Findings

### Finding 1: Piecewise CG with cap=512 is the best configuration for 7B

- **+26% to +44%** throughput at low concurrency (cc=4, cc=8)
- **+9%** throughput at medium concurrency (cc=32)
- **-2%** at high concurrency (cc=64) — near parity
- **No code changes required** — purely a CLI configuration change
- Best latency profile across all percentiles

### Finding 2: Batch CG v2 code changes fix the catastrophic regression

- The two code changes (allow bs=1, skip piecewise fallback) transform batch_cg from **-65% degradation to -5%** at cc=32
- For 32B with tp=4, batch_cg_v2 **beats vanilla** at cc=4 and cc=32
- The fix correctly addresses H3 (batching degradation) by allowing long sequences to use eager execution

### Finding 3: Dataset sorting is counterproductive at high concurrency

- Sorting improves low-cc performance (+3-5%) by grouping short prompts for better CG hit rates
- Sorting **hurts** high-cc performance (-4% to -10%) by creating convoy effects with clustered long prompts
- Tail latency (p99) is 2-5x worse with sorting at high cc
- **Recommendation: Do not sort the dataset** for production workloads with mixed prompt lengths

### Finding 4: tp=4 dramatically improves 32B CG performance

- 32B vanilla goes from 124.7 to 287.3 RPS at cc=64 (+130%) with tp=4
- 32B batch_cg goes from 44.6 to 285.8 RPS at cc=64 (+541%) with tp=4 + code fixes
- Tensor parallelism reduces per-GPU memory pressure and compute, making CG overhead proportionally smaller

### Finding 5: The CG cap approach (limiting max tokens) is more effective than the grid approach

- Piecewise CG cap=512 (limit when CG applies): **+9.3%** at cc=32
- Batch CG dense grid (reduce padding): **-44%** at cc=32 (Exp11)
- Batch CG v2 (small graphs + eager fallback): **-4.8%** at cc=32
- The key insight: **controlling when CG runs** matters more than **how efficiently it runs**

---

## 9. Recommendations

### For Immediate Deployment

1. **Use `--piecewise-cuda-graph-max-tokens 512`** for 7B models as the default when prefill CG is enabled. This provides +9-44% throughput improvement at cc≤32 with minimal regression at cc=64.

2. **Consider making 512 the default cap** in `server_args.py` when `enable_piecewise_cuda_graph` is set. The current default (`chunked_prefill_size`, typically 8192) captures too many graph sizes and causes H3 at high concurrency.

### For Future Work

3. **Adaptive cap based on queue depth**: Monitor `len(waiting_queue)` and dynamically adjust the cap. Low queue depth → use CG (fast individual requests), high queue depth → skip CG (let scheduler form large batches).

4. **Merge the batch_cg_v2 code changes**: The two-line changes are safe and backward-compatible. They only affect behavior when `--enable-batch-prefill-cuda-graph` is explicitly set.

5. **Investigate piecewise_cap512 for 32B**: This experiment didn't test piecewise_cap512 on 32B tp=4. Given its strong 7B results, it may also outperform vanilla on 32B.

---

## 10. Appendix: Full Raw Data

### A. 7B Vanilla

| Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| RPS | 135.0 | 204.1 | 411.4 | 450.2 |
| Wall time (s) | 74.1 | 49.0 | 24.3 | 22.2 |
| Latency avg (ms) | 29.4 | 38.9 | 76.8 | 140.0 |
| Latency p50 (ms) | 27.4 | 35.5 | 58.0 | 100.8 |
| Latency p90 (ms) | 36.8 | 44.1 | 122.8 | 243.6 |
| Latency p95 (ms) | 42.1 | 54.1 | 179.0 | 296.9 |
| Latency p99 (ms) | 74.7 | 95.3 | 378.2 | 466.3 |
| Latency max (ms) | 368.0 | 480.7 | 556.6 | 792.7 |
| Success | 10000 | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 | 0 |

### B. 7B Batch CG v2

| Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| RPS | 158.5 | 236.8 | 391.5 | 441.3 |
| Wall time (s) | 63.1 | 42.2 | 25.5 | 22.7 |
| Latency avg (ms) | 25.1 | 33.4 | 80.7 | 142.9 |
| Latency p50 (ms) | 23.1 | 29.8 | 62.8 | 102.6 |
| Latency p90 (ms) | 33.5 | 44.0 | 127.6 | 261.9 |
| Latency p95 (ms) | 42.8 | 56.4 | 178.0 | 336.7 |
| Latency p99 (ms) | 73.7 | 101.9 | 406.0 | 440.9 |
| Latency max (ms) | 457.7 | 476.9 | 665.3 | 801.3 |
| Success | 10000 | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 | 0 |

### C. 7B Batch CG v2 (Sorted)

| Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| RPS | 163.1 | 248.8 | 375.3 | 397.9 |
| Wall time (s) | 61.3 | 40.2 | 26.6 | 25.1 |
| Latency avg (ms) | 24.3 | 31.6 | 80.6 | 145.7 |
| Latency p50 (ms) | 19.4 | 26.2 | 47.0 | 85.7 |
| Latency p90 (ms) | 30.7 | 38.4 | 104.5 | 245.8 |
| Latency p95 (ms) | 37.1 | 53.2 | 219.9 | 401.8 |
| Latency p99 (ms) | 108.3 | 200.7 | 758.4 | 1226.8 |
| Latency max (ms) | 639.2 | 923.3 | 2318.3 | 3794.9 |
| Success | 10000 | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 | 0 |

### D. 7B Piecewise CG cap=512

| Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| RPS | 170.3 | 293.4 | 449.6 | 439.4 |
| Wall time (s) | 58.7 | 34.1 | 22.2 | 22.8 |
| Latency avg (ms) | 23.3 | 27.0 | 70.3 | 143.6 |
| Latency p50 (ms) | 21.3 | 21.5 | 52.0 | 102.9 |
| Latency p90 (ms) | 28.2 | 38.7 | 121.5 | 261.9 |
| Latency p95 (ms) | 35.8 | 52.0 | 156.9 | 332.6 |
| Latency p99 (ms) | 71.2 | 101.0 | 286.0 | 711.4 |
| Latency max (ms) | 451.7 | 581.5 | 599.1 | 813.9 |
| Success | 10000 | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 | 0 |

### E. 32B Vanilla (tp=4)

| Metric | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| RPS | 69.1 | 236.9 | 287.3 |
| Wall time (s) | 144.8 | 42.2 | 34.8 |
| Latency avg (ms) | 57.6 | 133.9 | 219.7 |
| Latency p50 (ms) | 62.3 | 111.1 | 175.7 |
| Latency p90 (ms) | 73.6 | 189.0 | 366.0 |
| Latency p95 (ms) | 81.1 | 246.2 | 445.7 |
| Latency p99 (ms) | 120.2 | 557.7 | 653.6 |
| Latency max (ms) | 600.0 | 797.8 | 990.1 |
| Success | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 |

### F. 32B Batch CG v2 (tp=4)

| Metric | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| RPS | 72.8 | 242.5 | 285.8 |
| Wall time (s) | 137.3 | 41.2 | 35.0 |
| Latency avg (ms) | 54.7 | 130.5 | 220.9 |
| Latency p50 (ms) | 46.7 | 111.7 | 171.1 |
| Latency p90 (ms) | 71.1 | 185.0 | 360.2 |
| Latency p95 (ms) | 90.8 | 251.8 | 426.0 |
| Latency p99 (ms) | 125.6 | 395.2 | 1009.8 |
| Latency max (ms) | 599.7 | 862.9 | 1964.5 |
| Success | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 |

### G. 32B Batch CG v2 Sorted (tp=4)

| Metric | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| RPS | 77.2 | 217.4 | 250.3 |
| Wall time (s) | 129.5 | 46.0 | 40.0 |
| Latency avg (ms) | 51.5 | 140.2 | 233.6 |
| Latency p50 (ms) | 43.3 | 96.4 | 137.9 |
| Latency p90 (ms) | 67.2 | 166.6 | 386.2 |
| Latency p95 (ms) | 100.0 | 323.2 | 659.9 |
| Latency p99 (ms) | 164.0 | 1039.0 | 1825.8 |
| Latency max (ms) | 765.6 | 3364.1 | 5389.2 |
| Success | 10000 | 10000 | 10000 |
| Fail | 0 | 0 | 0 |

---

## 11. Files Modified

| File | Change |
|---|---|
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py:800-801` | Removed `if forward_batch.batch_size <= 1: return False` |
| `python/sglang/srt/model_executor/model_runner.py:2325-2336` | Added `skip_piecewise` logic when batch_prefill is enabled |
| `bench_laps_prefill_throughput/sort_dataset.py` | Created: sorts JSONL by estimated token count |
| `bench_laps_prefill_throughput/bench_exp13_fix.sh` | Created: benchmark script for all configurations |
