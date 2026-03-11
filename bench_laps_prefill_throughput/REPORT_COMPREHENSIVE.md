# Comprehensive Report: SGLang Batch Prefill CUDA Graph Optimization

**Date**: 2026-03-05
**Repository**: `/mnt/weka/home/jianshu.she/graph/sglang` (branch: `batch-prefill-attn-in-graph`)
**Model**: Qwen/Qwen2.5-7B-Instruct (TP=1, unless noted)
**Dataset**: lmsys-chat-1m human prompts (10,000 variable-length prompts, max_tokens=1 prefill-only)
**Hardware**: Single GPU (A100/H100), mem-fraction-static=0.85

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Background & Motivation](#2-background--motivation)
3. [Experiment 7: Piecewise CG Baseline](#3-experiment-7-piecewise-cg-baseline)
4. [Experiment 8: All Configurations × 2 Models](#4-experiment-8-all-configurations--2-models)
5. [Experiments 9–12: Diagnostic Suite](#5-experiments-912-diagnostic-suite)
6. [Experiment 13: Making Batch Prefill CG Work](#6-experiment-13-making-batch-prefill-cg-work)
7. [Experiment 14a/14b: Timing & Multi-GPU](#7-experiment-14a14b-timing--multi-gpu)
8. [Experiment 15: Hybrid Mode & Cumulative Fixes](#8-experiment-15-hybrid-mode--cumulative-fixes)
9. [Experiment 16: Queue-Aware Gating](#9-experiment-16-queue-aware-gating)
10. [Experiment 17: Length-Aware Batch Construction](#10-experiment-17-length-aware-batch-construction)
11. [Experiment 18: Eager vs Piecewise Fallback](#11-experiment-18-eager-vs-piecewise-fallback)
12. [Experiment 19: CG Grid Optimization](#12-experiment-19-cg-grid-optimization)
13. [Final Conclusions & Recommendations](#13-final-conclusions--recommendations)
14. [Appendix: File References](#14-appendix-file-references)

---

## 1. Executive Summary

This report documents a systematic 13-experiment campaign to optimize CUDA graph (CG) usage for SGLang's prefill path. The work progressed through four phases:

1. **Baseline & Discovery** (Exp 7–8): Established that piecewise CG helps at low concurrency (1.25–1.46×) but batch prefill CG causes catastrophic regression at high concurrency (0.29–0.48×).

2. **Diagnosis** (Exp 9–12): Identified three hypotheses. H1 (padding waste) partially confirmed, H2 (memory pressure) refuted, H3 (batching degradation) confirmed as the primary cause.

3. **Fixes & Optimization** (Exp 13–16): Iteratively fixed batch prefill CG, introduced hybrid mode (piecewise fallback), and developed queue-depth gating — the final recommended configuration.

4. **Diminishing Returns** (Exp 17–19): Explored length-aware batching, eager fallback, and CG grid optimization. All showed minimal or negative impact, confirming the current solution is near-optimal.

**Final Recommended Configuration** (Approach B, queue-depth gate):
```
--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph
--batch-prefill-batch-sizes 1 2 4 8
--batch-prefill-seq-lengths 8 16 32 64 128
--batch-prefill-max-seq-len 128
--batch-prefill-max-padding-ratio 2.0
--piecewise-extend-max-bs 9999
--piecewise-max-waiting-reqs 8
```

**Result vs vanilla** (Qwen2.5-7B, TP=1):

| Concurrency | Vanilla RPS | Optimized RPS | Delta |
|:-----------:|:-----------:|:-------------:|:-----:|
| cc=4 | 133.9 | 179.2 | **+33.8%** |
| cc=8 | 200.9 | 290.4 | **+44.6%** |
| cc=32 | 409.5 | 419.0 | +2.3% |
| cc=64 | 460.9 | 439.6 | -4.6% |

---

## 2. Background & Motivation

### What are CUDA Graphs?

CUDA graphs capture a sequence of GPU operations (kernel launches, memory copies) into a replayable graph object. Replaying a captured graph eliminates CPU-side kernel launch overhead, which can be significant when GPU kernels are small and fast (as in decode steps or short prefills).

SGLang supports two CG modes for the prefill (extend) path:

- **Piecewise CG**: Captures graphs for individual sequences at various lengths. During forward, each sequence in a batch is replayed independently (bs=1). Benefits small batches but prevents the scheduler from accumulating large batches.

- **Batch Prefill CG**: Captures graphs for batched prefill at fixed (batch_size, seq_length) grid points. Sequences are padded to a common length using right-aligned padding for flash attention compatibility. Benefits: preserves batch structure. Cost: padding waste.

### The Problem

Initial testing showed piecewise CG worked well at low concurrency but batch prefill CG caused severe throughput regression at high concurrency. This report documents the systematic investigation and resolution.

---

## 3. Experiment 7: Piecewise CG Baseline

**Script**: `bench_exp7_piecewise_baseline.sh`
**Date**: 2026-03-02
**Objective**: Establish clean baseline comparing vanilla (no CG) vs piecewise CG using the HTTP/2 load tester.

### Configuration

| Setting | Server Args |
|---------|------------|
| vanilla | (none) |
| piecewise_cg | `--enable-piecewise-cuda-graph --disable-cuda-graph` |

### Results

| Setting | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---------|------|------|------|-------|-------|-------|--------|
| vanilla | 98.9 | 136.4 | 197.7 | 344.2 | 414.6 | 434.9 | 378.3 |
| piecewise_cg | 88.8 | 170.0 | 288.9 | 357.2 | 410.5 | 437.2 | 365.2 |
| **Speedup** | 0.90× | **1.25×** | **1.46×** | 1.04× | 0.99× | 1.01× | 0.97× |

### Analysis

- **Sweet spot at cc=4–8**: Piecewise CG eliminates kernel launch overhead when batch sizes are small (1–4 sequences). The GPU spends less time idle between kernel launches.
- **Diminishing returns at cc≥16**: As concurrency grows, the scheduler builds larger batches. Piecewise CG processes these one-at-a-time (bs=1 replay), which is less efficient than native batched attention.
- **Slight regression at cc=2**: CG capture/replay overhead exceeds the benefit when there's only one sequence at a time.

### Conclusion

Piecewise CG is valuable at low-to-medium concurrency. The challenge is to either (a) extend this benefit to high concurrency, or (b) gracefully disable CG when it's not helpful.

---

## 4. Experiment 8: All Configurations × 2 Models

**Script**: `bench_exp8_all_configs.sh`
**Date**: 2026-03-02
**Objective**: Compare all four server configurations across both 7B and 32B models.

### Configurations

| Setting | Description |
|---------|------------|
| vanilla | No CUDA graphs |
| piecewise_cg | Per-sequence CG replay |
| batch_cg | Batched prefill CG (bs=[1..32] × seq=[1..512], 60 graphs) |
| laps | batch_cg + LAPS scheduler (auto-disabled in non-disagg mode) |

### Results — Qwen2.5-7B-Instruct (TP=1)

| Setting | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 |
|---------|------|------|-------|-------|-------|
| vanilla | 128.8 | 200.7 | 329.5 | 415.2 | 442.3 |
| piecewise_cg | 170.9 (1.33×) | 290.6 (1.45×) | 357.8 (1.09×) | 411.4 (0.99×) | 435.5 (0.98×) |
| batch_cg | 172.1 (1.34×) | 204.9 (1.02×) | 170.0 (0.52×) | 143.3 (**0.35×**) | 211.2 (**0.48×**) |
| laps | 175.8 (1.36×) | 198.5 (0.99×) | 176.0 (0.53×) | 149.9 (0.36×) | 210.6 (0.48×) |

### Results — Qwen2.5-32B (TP=4)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 59.2 | 82.4 | 119.6 | 124.7 |
| piecewise_cg | 52.8 (0.89×) | 79.1 (0.96×) | 112.8 (0.94×) | 116.3 (0.93×) |
| batch_cg | 42.2 (0.71×) | 43.7 (0.53×) | 35.1 (**0.29×**) | 44.6 (**0.36×**) |

### Key Findings

1. **Batch prefill CG is catastrophic at high concurrency**: Only 29–48% of vanilla throughput at cc=32/64.
2. **LAPS auto-disables in non-disagg mode**: Results identical to batch_cg.
3. **Piecewise CG pattern confirmed**: Good at cc=4–8, neutral at cc=32+.
4. **32B model**: CG is less helpful overall; piecewise slightly hurts at all cc levels.

### Conclusion

Batch prefill CG has a fundamental problem at high concurrency. Need to diagnose whether it's padding waste (H1), memory pressure (H2), or batching degradation (H3).

---

## 5. Experiments 9–12: Diagnostic Suite

**Script**: `bench_exp9_12_diagnose.sh`
**Date**: 2026-03-02
**Objective**: Test three hypotheses for batch prefill CG's catastrophic regression.

### Experiment 9: Uniform Dataset (H1 — Padding Waste)

**Idea**: If padding waste causes the regression, using uniform-length prompts (40–80 tokens) should eliminate it.

| Setting | cc=4 | cc=32 | cc=64 |
|---------|------|-------|-------|
| vanilla | 155.1 | 565.4 | 632.7 |
| piecewise_cg | 184.7 (1.19×) | 636.3 (1.13×) | 649.6 (1.03×) |
| batch_cg | 199.8 (1.29×) | 480.6 (**0.85×**) | 575.8 (**0.91×**) |

**Verdict**: Batch CG *still* regresses at cc=32/64 even with uniform data. **H1 is a contributing factor but not the root cause.**

### Experiment 10: Piecewise CG Token Cap (H3 — Batching Degradation)

**Idea**: If piecewise CG drains the queue too fast (bs=1 replays), capping the max sequence length should force longer sequences through eager mode, allowing batch accumulation.

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 128.8 | 200.7 | 415.2 | 442.3 |
| piecewise (uncapped) | 170.9 | 290.6 | 411.4 | 435.5 |
| piecewise_cap256 | 167.7 | 284.7 | 430.3 | 443.9 |
| **piecewise_cap512** | **172.2** | **294.4** | **450.4** | **445.2** |
| piecewise_cap1024 | 174.1 | 296.1 | 442.4 | 435.4 |

**Verdict**: cap512 is the sweet spot — preserves low-cc gains while improving cc=32 to 1.08×. **H3 confirmed: controlling CG scope improves batch accumulation.**

### Experiment 11: CG Grid Density (H1 — Padding Waste, continued)

**Idea**: If padding waste matters, denser CG grids (more captured shapes) should reduce it.

| Setting | Graphs | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|--------|------|------|-------|-------|
| batch_cg (exp8) | 60 | 172.1 | 204.9 | 143.3 | 211.2 |
| batch_cg_dense | 132 | 183.5 | 227.6 | 182.5 | 246.3 |
| batch_cg_vdense | 224 | 182.7 | 241.9 | 197.4 | 257.5 |

**Verdict**: Denser grids help (+10–30% improvement) but batch_cg still far below vanilla at cc=32/64. **H1 partially confirmed: padding matters, but H3 is dominant.**

### Experiment 12: KV Cache Reduction (H2 — Memory Pressure)

**Idea**: If CG memory reduces available KV cache, artificially limiting KV cache should reproduce the regression.

| Setting | cc=4 | cc=32 | cc=64 |
|---------|------|-------|-------|
| vanilla (default KV) | 59.2 | 119.6 | 124.7 |
| vanilla_kv180k | 62.1 | 119.8 | 124.6 |
| vanilla_kv120k | 62.2 | 119.8 | 122.4 |

**Verdict**: Reducing KV cache by 33% has no effect on throughput. **H2 refuted.**

### Diagnostic Summary

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1: Padding waste | Partially confirmed | Denser grids help 10-30% but don't fix the core issue |
| H2: Memory pressure | **Refuted** | KV cache reduction has zero effect |
| H3: Batching degradation | **Confirmed (primary)** | Uniform data doesn't fix it; token cap helps |

---

## 6. Experiment 13: Making Batch Prefill CG Work

**Date**: 2026-03-02
**Objective**: Fix the catastrophic batch_cg regression using insights from experiments 9–12.

### Changes Made

1. **Batch CG v2** (code changes in `piecewise_cuda_graph_runner.py`):
   - Allow bs=1 batch prefill replays (previously skipped to piecewise)
   - Skip piecewise CG fallback when batch_prefill is enabled
   - Result: Fixes catastrophic regression (cc=32: 143.3 → 391.5 RPS, **+173%**)

2. **Piecewise cap=512** (configuration only):
   - `--piecewise-cuda-graph-max-tokens 512`
   - Limits CG scope to sequences ≤512 tokens
   - Result: Best overall performance

3. **Sorted dataset** (test only):
   - Pre-sorted prompts by length
   - Result: Moderate improvement in CG hit rates

### Results — 7B (TP=1)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 135.0 | 204.1 | 411.4 | 450.2 |
| **piecewise_cap512** | **170.3 (+26%)** | **293.4 (+44%)** | **449.6 (+9.3%)** | 439.4 (-2.4%) |
| batch_cg_v2 | 166.9 (+24%) | 280.8 (+38%) | 391.5 (-4.8%) | 441.3 (-2.0%) |
| sorted + piecewise | 168.2 (+25%) | 283.5 (+39%) | 428.3 (+4.1%) | 433.9 (-3.6%) |

### Key Insight

The critical fix was understanding that batch_cg's original code path forced ALL sequences through piecewise CG when batch prefill missed — even when eager mode would have been better. By fixing the fallback logic and capping CG scope, performance recovered to near-vanilla at high cc while preserving large gains at low cc.

---

## 7. Experiment 14a/14b: Timing & Multi-GPU

**Date**: 2026-03-03

### Exp 14a: Timing Instrumentation (7B, TP=1)

**Objective**: Add detailed timing to understand where batch_cg_v2 spends time vs vanilla at cc=32.

| Setting | RPS |
|---------|-----|
| vanilla | 272.1 |
| batch_cg_v2 | 272.0 |

**Verdict**: At cc=32, batch_cg_v2 and vanilla are essentially identical. The CG overhead and CG benefit cancel out.

### Exp 14b: Multi-GPU Scaling (7B, TP=4)

**Objective**: Test if CG benefits scale with tensor parallelism.

| Setting | cc=4 | cc=64 |
|---------|------|-------|
| vanilla_tp1 | 135.0 | 450.2 |
| vanilla_tp4 | — | 533.1 (+18.4%) |
| batch_cg_v2_tp4 | — | 528.2 (+19.7% vs tp1) |
| piecewise_cap512_tp4 | — | 531.2 (+21.0% vs tp1) |

**Verdict**: TP=4 improves throughput significantly, and CG benefits roughly maintain their relative position (slightly worse than vanilla at high cc, better at low cc).

---

## 8. Experiment 15: Hybrid Mode & Cumulative Fixes

**Script**: `bench_exp15.sh`
**Date**: 2026-03-03
**Objective**: Systematically apply cumulative fixes and find the best batch prefill CG configuration.

### Three Phases

**Phase A — Padding Ratio Threshold (2.0×)**:
- Gate: Reject batch prefill CG when padding ratio > 2.0
- Effect: Fixes cc=32/64 regression, but without piecewise fallback, cc=4/8 regresses

**Phase B — Hybrid Mode (batch_prefill CG + piecewise CG fallback)** ← Winner:
- When batch prefill CG misses (padding too high, seq too long), fall back to piecewise CG instead of eager
- Result: Best-ever low-cc performance

**Phase C — Expanded CG Grid (45 graphs)**:
- More (bs, seq) combinations: bs=[1,2,4,8,16] × seq=[8,16,32,48,64,96,128,192,256]
- Result: No measurable improvement over 20-graph grid

### Results — Phase B (v3_hybrid)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 135.0 | 204.1 | 411.4 | 450.2 |
| v3_hybrid | **176.9 (+31%)** | **283.8 (+39%)** | 414.7 (+0.8%) | **430.3 (-4.4%)** |

### Problem Identified

At cc=64, v3_hybrid regresses by 4.4%. Root cause: piecewise CG fallback absorbs ALL work (even when queue is deep), preventing batch accumulation. The scheduler never sees a large enough queue to build efficient batches.

### Conclusion

Hybrid mode is the best approach, but piecewise CG fallback needs to be gated based on system load. This leads directly to Experiment 16.

---

## 9. Experiment 16: Queue-Aware Gating

**Script**: `bench_exp16.sh` + `bench_exp16_final.sh`
**Date**: 2026-03-04
**Objective**: Gate piecewise CG fallback based on runtime conditions to eliminate cc=64 regression.

### Three Approaches Tested

**Approach A — Batch-Size Gate** (`--piecewise-extend-max-bs 4`):
- Skip piecewise CG when forward batch_size > 4
- Result: Helps cc=64 (+1.5%) but hurts cc=8 (-13.4%). Too aggressive.

**Approach B — Queue-Depth Gate** (`--piecewise-max-waiting-reqs 8`) ← **RECOMMENDED**:
- Skip piecewise CG when waiting_queue > 8 requests
- Intuition: At low cc, queue is usually empty → use piecewise CG for speed. At high cc, queue builds up → skip piecewise CG to allow batch accumulation.

**Approach C — Combined** (batch-size gate + adaptive padding ratio):
- Relax padding threshold for large batches
- Result: Best cc=64 (+1.1% vs vanilla) but hurts cc=8 (-6.3%)

### Final Results — Approach B (Definitive, Exp16_final)

| Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|--------|------|------|-------|-------|
| **Vanilla RPS** | 133.9 | 200.9 | 409.5 | 460.9 |
| **Approach B RPS** | **179.2** | **290.4** | **419.0** | 439.6 |
| **Delta** | **+33.8%** | **+44.6%** | **+2.3%** | -4.6% |
| Vanilla latency (avg) | 29.7ms | 39.6ms | 77.1ms | 137.0ms |
| Approach B latency (avg) | **22.2ms** | **27.2ms** | **75.4ms** | 143.5ms |

### Why Approach B Works

The queue-depth gate creates an adaptive two-mode system:

- **Low load (queue ≤ 8)**: Piecewise CG processes sequences individually with minimal launch overhead. Perfect for cc=4/8 where batches are naturally small.
- **High load (queue > 8)**: Piecewise CG is skipped, sequences go through eager mode. The scheduler accumulates large batches for efficient native attention. Preserves cc=32/64 performance.

The threshold of 8 was chosen because:
- Max CG batch size is 8, so queue > 8 means the scheduler has enough to build a full batch
- At cc=4, queue rarely exceeds 8, so CG is used ~100% of the time
- At cc=64, queue frequently exceeds 8, so CG is used selectively

---

## 10. Experiment 17: Length-Aware Batch Construction

**Script**: `bench_exp17.sh`
**Date**: 2026-03-04
**Objective**: Test whether sorting the waiting queue by prompt length can improve CG hit rates by grouping similar-length sequences.

### Motivation

At cc=32/64, batch prefill CG frequently misses due to high padding ratios. Example:
- FIFO batch: [len=120, len=15, len=95, len=8] → pad to 128 → ratio = 2.15 → **MISS**
- Length-sorted: [8, 15] → pad to 16 → ratio 1.39 → **HIT**; [95, 120] → pad to 128 → ratio 1.19 → **HIT**

### Implementation

Added `--prefill-sort-window N` CLI argument. Before building each prefill batch, sort the first N requests in the waiting queue by `len(origin_input_ids)`. Stable sort preserves FIFO within equal-length groups.

### Results — B+D (window=64) vs B (queue-gate only)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 134.3 | 202.9 | 413.7 | 446.0 |
| B: queue_gate | 179.5 | 286.0 | 422.0 | 444.7 |
| B+D: + length_sort | 183.5 | 283.1 | 414.9 | 444.9 |
| **B+D vs B** | +2.2% | -1.0% | -1.7% | +0.0% |

### CG Hit Rate Analysis

| Metric | B only | B+D |
|--------|--------|-----|
| Total forward calls | ~14,700 | ~14,700 |
| BP-CG-REPLAY (hits) | 5,703 (38.8%) | 6,243 (42.5%) |
| BP-CG-MISS | 5,422 | 4,882 |
| Piecewise CG | 3,575 | 3,575 |

CG hit rate improved by +3.7 percentage points, but throughput was unchanged.

### Why It Didn't Work

1. **Dominant miss reason is `seq_too_long` (34.8%)**: These sequences exceed 128 tokens and cannot be helped by sorting — they miss regardless of batch composition.
2. **Piecewise CG fallback is nearly as fast**: When batch_prefill CG misses, piecewise CG handles the work at similar throughput. Converting a miss to a hit doesn't save much time.
3. **Marginal computation saving**: Even when CG hits, padded attention is only slightly faster than native flash attention for variable-length sequences.

### Decision

**Approach D not shipped.** Changes reverted from `server_args.py` and `scheduler.py`.

---

## 11. Experiment 18: Eager vs Piecewise Fallback

**Script**: `bench_exp18.sh`
**Date**: 2026-03-04
**Objective**: When batch prefill CG misses, is eager (native flash attention) or piecewise CG a better fallback?

### Motivation

The queue-depth gate (Approach B) already skips piecewise CG at high load. But what about the fallback path when batch_prefill CG misses at low load? Currently, misses fall through to piecewise CG. Would skipping piecewise CG entirely (eager fallback) be better?

### Implementation

Modified `model_runner.py` to skip piecewise CG when batch_prefill is enabled:
```python
can_run_graph = (... and not self.piecewise_cuda_graph_runner.batch_prefill_enabled)
```

### Results

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla (exp17) | 134.3 | 202.9 | 413.7 | 446.0 |
| B: piecewise_fb (exp17) | 179.5 | 286.0 | 422.0 | 444.7 |
| E: eager_fb | 149.2 | 228.1 | 413.2 | 454.2 |
| **E vs B** | **-16.9%** | **-20.2%** | -2.1% | **+2.1%** |

### Analysis

- **cc=4/8**: Eager fallback loses badly. At low concurrency, most batches are bs=1–2, and piecewise CG is genuinely faster than eager for these small batches.
- **cc=64**: Eager fallback wins slightly (+2.1%). At high concurrency, native batched attention is more efficient, and piecewise CG's bs=1 replays are wasteful.
- **Queue-depth gate already handles this**: Approach B skips piecewise CG when queue > 8, achieving the same adaptive behavior without code changes.

### Decision

**Eager fallback not shipped.** The existing queue-depth gate (Approach B) already provides the right adaptive behavior. Change reverted from `model_runner.py`.

---

## 12. Experiment 19: CG Grid Optimization

**Script**: `bench_exp19.sh`
**Date**: 2026-03-05
**Objective**: Optimize the (batch_size, seq_length) grid to maximize CG hit rates and minimize padding waste.

### Motivation

The current grid captures 20 graphs: bs=[1,2,4,8] × seq=[8,16,32,64,128]. CG hit rate is only ~39%. Can we design a better grid based on the actual dataset distribution?

### Dataset Analysis

Tokenized 10K prompts from lmsys-chat-1m with Qwen2.5-7B tokenizer:

| Percentile | Token Count |
|:----------:|:-----------:|
| Min | 1 |
| P25 | 9 |
| Median | 17 |
| P75 | 47 |
| P90 | 128 |
| P95 | 270 |
| P99 | 881 |
| Max | 13,208 |

Key: 71% of prompts ≤32 tokens, 90.5% ≤128 tokens. Heavy concentration at very short lengths.

### Grids Tested

| Grid | Batch Sizes | Seq Lengths | Max Seq | Ratio | Graphs |
|------|------------|-------------|---------|-------|--------|
| Current | [1,2,4,8] | [8,16,32,64,128] | 128 | ≤2.0 | 20 |
| G (fine_seq) | [1,2,4,8] | [4,8,12,16,20,24,32,48,64,96,128] | 128 | ≤2.0 | 44 |
| L (optimized) | [1,2,3,4,6,8,12,16] | [4,8,12,16,20,24,32,48,64,96,128,192,256,384,512] | 512 | ≤3.0 | 120 |

### Results

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---------|------|------|-------|-------|
| vanilla | 130.2 | 199.2 | 408.3 | 452.4 |
| current (20) | **179.5 (+37.9%)** | **285.0 (+43.1%)** | **419.7 (+2.8%)** | 444.7 (-1.7%) |
| G: fine_seq (44) | 174.2 (+33.8%) | 284.9 (+43.0%) | 414.9 (+1.6%) | 443.4 (-2.0%) |
| L: optimized (120) | 173.4 (+33.2%) | 267.9 (+34.5%) | 394.6 (-3.4%) | 430.3 (-4.9%) |

### The Surprising Finding

**More graphs = WORSE throughput.** This is counterintuitive but explained by:

1. **CG capture time**: 120 graphs take significantly longer to capture at startup, increasing warmup time.
2. **Padded attention is inherently slower**: When batch_prefill CG hits, it runs padded flash attention. For variable-length sequences, the padded computation wastes FLOPs even with a tight grid. Native flash attention handles variable lengths natively with zero waste.
3. **The current grid's benefit comes from piecewise CG, not batch prefill CG**: At cc=4/8, most of the +34–44% gain comes from piecewise CG processing bs=1 batches. Batch prefill CG contributes minimally.
4. **Grid L forces more work through padded CG**: With 80.7% hit rate (vs 39.2% for current), Grid L forces more batches through padded attention instead of efficient native attention — making things worse.

### CG Hit Rate vs Throughput (cc=32)

| Grid | CG Hit Rate | RPS | Delta vs vanilla |
|------|:-----------:|:---:|:----------------:|
| Current (20) | 39.2% | 419.7 | +2.8% |
| G (44) | 47.9% | 414.9 | +1.6% |
| L (120) | 80.7% | 394.6 | **-3.4%** |

**Higher CG hit rate correlates with LOWER throughput.** This definitively proves that batch prefill CG with padding is not beneficial for variable-length prefill workloads.

### Decision

**Keep current 20-graph grid.** The gains at low cc come from piecewise CG, not batch prefill CG. The current grid provides the best balance.

---

## 13. Final Conclusions & Recommendations

### What Works

1. **Piecewise CG at low concurrency**: +26–44% throughput improvement at cc=4/8. The kernel launch overhead elimination is genuinely valuable for small batches.

2. **Queue-depth gating** (`--piecewise-max-waiting-reqs 8`): Adaptive two-mode system that uses piecewise CG when the system is lightly loaded and disables it when the queue builds up. Provides the best overall profile across all concurrency levels.

3. **Current CG grid** (20 graphs): The minimal grid is sufficient. More graphs don't help because the benefit comes from piecewise CG, not batch prefill CG.

### What Doesn't Work

1. **Batch prefill CG with padding**: For variable-length sequences, padded flash attention is inherently slower than native flash attention. Higher CG hit rates → worse throughput.

2. **Length-sorted batching**: Improves CG hit rate by +3.7pp but has zero throughput impact because the dominant miss reasons (seq_too_long) are not addressed, and CG hits aren't faster than the fallback.

3. **Dense CG grids**: More captured shapes increase hit rates but also increase capture time and force more work through padded attention.

4. **Eager fallback**: Hurts at low cc where piecewise CG is genuinely valuable. Queue-depth gating already handles the transition.

### Fundamental Insight

The primary value of CUDA graphs for prefill is **eliminating CPU kernel launch overhead**, not **avoiding recomputation**. This overhead is significant only when batch sizes are small (1–4). At higher batch sizes, the GPU is already saturated and CG provides no benefit — in fact, padding makes things worse.

The optimal strategy is therefore simple: **use CG when batches are small (low load), disable when batches are large (high load)**. The queue-depth gate achieves exactly this.

### Recommended Production Configuration

```bash
# For Qwen2.5-7B-Instruct (TP=1)
--enable-piecewise-cuda-graph \
--enable-batch-prefill-cuda-graph \
--disable-cuda-graph \
--batch-prefill-batch-sizes 1 2 4 8 \
--batch-prefill-seq-lengths 8 16 32 64 128 \
--batch-prefill-max-seq-len 128 \
--batch-prefill-max-padding-ratio 2.0 \
--piecewise-extend-max-bs 9999 \
--piecewise-max-waiting-reqs 8
```

### Performance Summary

| Concurrency | Vanilla | Optimized (B) | Improvement |
|:-----------:|:-------:|:-------------:|:-----------:|
| cc=4 | 133.9 req/s | 179.2 req/s | **+33.8%** |
| cc=8 | 200.9 req/s | 290.4 req/s | **+44.6%** |
| cc=32 | 409.5 req/s | 419.0 req/s | +2.3% |
| cc=64 | 460.9 req/s | 439.6 req/s | -4.6% |

The optimization delivers substantial gains at low-to-medium concurrency (the common case for interactive serving) with minimal regression at high concurrency (the batch processing case where throughput is already high).

---

## 14. Appendix: File References

### Benchmark Scripts

| Experiment | Script |
|-----------|--------|
| Exp 7 | `bench_laps_prefill_throughput/bench_exp7_piecewise_baseline.sh` |
| Exp 8 | `bench_laps_prefill_throughput/bench_exp8_all_configs.sh` |
| Exp 9–12 | `bench_laps_prefill_throughput/bench_exp9_12_diagnose.sh` |
| Exp 13 | `bench_laps_prefill_throughput/bench_exp13_batch_cg_v2.sh` |
| Exp 14a | `bench_laps_prefill_throughput/bench_exp14a_timing.sh` |
| Exp 14b | `bench_laps_prefill_throughput/bench_exp14b_tp4.sh` |
| Exp 15 | `bench_laps_prefill_throughput/bench_exp15.sh` |
| Exp 16 | `bench_laps_prefill_throughput/bench_exp16.sh` |
| Exp 16 final | `bench_laps_prefill_throughput/bench_exp16_final.sh` |
| Exp 17 | `bench_laps_prefill_throughput/bench_exp17.sh` |
| Exp 18 | `bench_laps_prefill_throughput/bench_exp18.sh` |
| Exp 19 | `bench_laps_prefill_throughput/bench_exp19.sh` |

### Result Directories

| Experiment | Results Directory |
|-----------|------------------|
| Exp 7 | `bench_laps_prefill_throughput/results_exp7_baseline_2026-03-02/` |
| Exp 8 | `bench_laps_prefill_throughput/results_exp8_2026-03-02/` |
| Exp 9–12 | `bench_laps_prefill_throughput/results_exp9_12_2026-03-02/` |
| Exp 13 | `bench_laps_prefill_throughput/results_exp13_2026-03-02/` |
| Exp 14a | `bench_laps_prefill_throughput/results_exp14a_2026-03-03/` |
| Exp 14b | `bench_laps_prefill_throughput/results_exp14b_2026-03-03/` |
| Exp 15 | `bench_laps_prefill_throughput/results_exp15_2026-03-03/` |
| Exp 16 | `bench_laps_prefill_throughput/results_exp16_2026-03-04/` |
| Exp 16 final | `bench_laps_prefill_throughput/results_exp16_final_2026-03-04/` |
| Exp 17 | `bench_laps_prefill_throughput/results_exp17_2026-03-04/` |
| Exp 18 | `bench_laps_prefill_throughput/results_exp18_2026-03-04/` |
| Exp 19 | `bench_laps_prefill_throughput/results_exp19_2026-03-05/` |

### Key Source Files Modified

| File | Purpose |
|------|---------|
| `python/sglang/srt/server_args.py` | CLI argument definitions |
| `python/sglang/srt/managers/scheduler.py` | Batch construction logic |
| `python/sglang/srt/model_executor/model_runner.py` | Forward path selection (CG vs eager) |
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | CG capture/replay, batch prefill CG logic |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Right-aligned padding for flash attention |
