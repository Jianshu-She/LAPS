# Experiment 17: Length-Aware Batch Construction (Approach D)

**Date**: 2026-03-04
**Model**: Qwen/Qwen2.5-7B-Instruct (TP=1)
**GPU**: Single GPU (ID=5)
**Dataset**: lmsys-chat-1m (10,000 prompts, max_tokens=1)
**Concurrency levels**: 4, 8, 32, 64

---

## 1. Hypothesis

At high concurrency (cc=32/64), batch_prefill CUDA graphs frequently miss due to
high padding ratios. FIFO batch construction groups mixed-length sequences, and
padding all to the max length inflates the ratio above the 2.0 threshold.

**Approach D** sorts a bounded window (64 requests) of the waiting queue by prompt
length before building prefill batches. This groups similar-length sequences together,
reducing padding waste and improving CG hit rates.

Example:
- FIFO: `[len=120, len=15, len=95, len=8]` → pad to 128 → ratio = 2.15 → **MISS**
- Sorted: `[8, 15]` → pad 16 → ratio 1.39 → **HIT**; `[95, 120]` → pad 128 → ratio 1.19 → **HIT**

---

## 2. Implementation

Three changes:

| File | Change |
|------|--------|
| `server_args.py` | Added `--prefill-sort-window N` CLI arg (default 0 = disabled) |
| `scheduler.py` | 7-line stable sort of first `min(N, queue_len)` requests by `len(origin_input_ids)` after `calc_priority()` |
| `bench_exp17.sh` | Benchmark script: vanilla vs B vs B+D |

Settings tested:

| Setting | Extra Server Args |
|---------|-------------------|
| **vanilla** | (none) |
| **B: queue_gate** | `--piecewise-max-waiting-reqs 8` + CG flags |
| **B+D: queue_gate + length_sort** | `--piecewise-max-waiting-reqs 8 --prefill-sort-window 64` + CG flags |

CG flags: `--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128 --batch-prefill-max-padding-ratio 2.0`

---

## 3. Results

### 3.1 Throughput (req/s)

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| vanilla | 129.8 | 200.9 | 414.4 | 458.8 |
| B: queue_gate | 177.6 | 286.4 | 420.4 | 435.3 |
| B+D: length_sort | 181.6 | 283.6 | 413.2 | 435.5 |

### 3.2 RPS Delta vs Vanilla

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| B: queue_gate | +36.8% | +42.6% | +1.5% | -5.1% |
| B+D: length_sort | +39.9% | +41.2% | -0.3% | -5.1% |

### 3.3 RPS Delta: B+D vs B (isolated effect of Approach D)

| cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|
| **+2.2%** | -1.0% | -1.7% | +0.0% |

All deltas within noise (< 2.5%). **Approach D provides no meaningful throughput improvement.**

### 3.4 Latency (ms)

| Setting | Metric | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|---|
| vanilla | avg | 30.6 | 39.6 | 76.2 | 137.3 |
| B: queue_gate | avg | 22.4 | 27.6 | 75.1 | 145.0 |
| B+D: length_sort | avg | 21.9 | 27.9 | 76.4 | 144.9 |
| | | | | | |
| vanilla | p50 | 28.6 | 36.0 | 57.5 | 95.7 |
| B: queue_gate | p50 | 19.1 | 22.7 | 51.0 | 106.4 |
| B+D: length_sort | p50 | 18.8 | 23.0 | 51.9 | 106.0 |
| | | | | | |
| vanilla | p99 | 74.0 | 98.2 | 357.3 | 454.6 |
| B: queue_gate | p99 | 74.7 | 108.6 | 400.8 | 536.5 |
| B+D: length_sort | p99 | 75.3 | 108.8 | 355.2 | 469.0 |

Notable: B+D shows slightly better p99 at cc=32 (355ms vs 401ms) and cc=64 (469ms vs 537ms)
compared to B. This suggests length sorting may improve tail latency consistency, even though
median throughput is unchanged.

### 3.5 Cross-reference with Exp16 Final

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| **Exp16** vanilla | 133.9 | 200.9 | 409.5 | 460.9 |
| **Exp17** vanilla | 129.8 | 200.9 | 414.4 | 458.8 |
| **Exp16** B: queue_gate | 179.2 | 290.4 | 419.0 | 439.6 |
| **Exp17** B: queue_gate | 177.6 | 286.4 | 420.4 | 435.3 |

Exp16 and Exp17 results are consistent within ~3% noise, confirming reproducibility.

---

## 4. CUDA Graph Hit/Miss Analysis

### 4.1 Overall CG Statistics (aggregated across all 4 cc levels)

| Metric | B: queue_gate | B+D: length_sort | Delta |
|---|---|---|---|
| Total CG attempts | 11,120 | 11,566 | +446 (+4.0%) |
| CG hits (BP-CG-REPLAY) | 4,316 | 4,911 | **+595 (+13.8%)** |
| CG misses (BP-CG-MISS) | 6,804 | 6,655 | -149 (-2.2%) |
| **Hit rate** | **38.8%** | **42.5%** | **+3.7pp** |

Length sorting **does** improve the CG hit rate by ~3.7 percentage points (from 38.8% to 42.5%).

### 4.2 Miss Reason Breakdown

| Miss Reason | B: queue_gate | B+D: length_sort | Delta |
|---|---|---|---|
| padding_ratio (> 2.0) | 3,752 (55.1%) | 3,609 (54.2%) | **-143 (-3.8%)** |
| seq_too_long (> 128) | 2,367 (34.8%) | 2,367 (35.6%) | 0 (unchanged) |
| bs_too_large (> 8) | 685 (10.1%) | 679 (10.2%) | -6 |

Key observations:
- **padding_ratio misses decreased by 143** (3.8% reduction) — this is the mechanism D was designed to improve, and it works as expected.
- **seq_too_long misses are identical** (2,367) — expected, since sorting doesn't change which sequences exceed 128 tokens. This is the dominant miss category that D cannot address.
- **bs_too_large misses are unchanged** — expected, since sorting doesn't change batch sizes.

### 4.3 Forward Path Distribution

| Path | B: queue_gate | B+D: length_sort |
|---|---|---|
| batch_prefill_cg | 4,316 | 4,911 (+13.8%) |
| piecewise_cg | 6,800 | 6,655 (-2.1%) |
| eager | 4 | 0 |

More batches take the faster `batch_prefill_cg` path with length sorting. When batch_prefill CG
misses, the system falls back to `piecewise_cg` (per-sequence CG), which is slower but still
faster than eager.

---

## 5. Why CG Hit Rate Improved But Throughput Didn't

The core puzzle: **+13.8% more CG hits, but ~0% throughput change.** Three explanations:

### 5.1 The dominant miss reason is seq_too_long, not padding_ratio

2,367 out of 6,804 misses (34.8%) are because sequences exceed 128 tokens — the max
captured CG sequence length. Length sorting cannot help with these. To address this,
we would need to capture CG graphs for longer sequences (256, 512).

### 5.2 Piecewise CG fallback is nearly as fast

When batch_prefill CG misses, the system falls to `piecewise_cg` — individual per-sequence
CUDA graphs. This is slower than batch_prefill CG but much faster than eager execution.
The gap between batch_prefill_cg and piecewise_cg is small enough that converting 595
batches from piecewise→batch_prefill doesn't measurably move the needle.

### 5.3 The improvement concentrates at low cc where it matters least

At cc=4/8, the queue is shallow (1-4 requests) and batches are already small and
homogeneous. Sorting has little to reorder. At cc=32/64 where sorting could help most,
the dominant misses are seq_too_long and bs_too_large — categories sorting can't fix.

---

## 6. Interesting Finding: P99 Latency Improvement

Despite no throughput gain, B+D shows notably better tail latency:

| cc | B p99 | B+D p99 | Improvement |
|---|---|---|---|
| 32 | 400.8 ms | 355.2 ms | -11.4% |
| 64 | 536.5 ms | 469.0 ms | -12.6% |

Hypothesis: length sorting groups long sequences together. Without sorting, a batch
with one very long sequence pads all others wastefully, delaying them. With sorting,
long sequences are batched together (all need similar time) while short sequences
get their own fast batches. This reduces variance without changing the mean.

---

## 7. Conclusions

| Verdict | Detail |
|---|---|
| CG hit rate | Improved: 38.8% → 42.5% (+3.7pp). Mechanism works as designed. |
| Throughput | No improvement: all B+D vs B deltas within noise (< 2.5%). |
| Tail latency | Potential improvement: p99 reduced ~12% at cc=32/64 (needs confirmation). |
| Root cause | Dominant miss reasons (seq_too_long, piecewise_cg fallback) limit the throughput impact. |
| Recommendation | **Do not ship Approach D alone** — the complexity isn't justified by the results. |

### 7.1 Next Steps to Consider

1. **Extend CG grid to longer sequences** (256, 512 tokens): Would eliminate 2,367 seq_too_long
   misses (34.8% of all misses). This could unlock the throughput gain that D was designed for.

2. **Combine D with extended grid**: With seq_too_long eliminated, padding_ratio becomes the
   dominant miss reason, and D's 3.8% reduction would have higher relative impact.

3. **Investigate P99 improvement**: Run multiple trials at cc=32/64 to confirm whether the
   ~12% p99 reduction is real or noise. If confirmed, D may be worth keeping for latency SLOs.

4. **Profile batch_prefill_cg vs piecewise_cg kernel time**: Measure the per-batch time
   difference. If piecewise_cg is within 5% of batch_prefill_cg, then improving hit rate
   will never help throughput — the fallback is already good enough.

---

## 8. Raw Data

All results stored in:
```
bench_laps_prefill_throughput/results_exp17_2026-03-04/
  vanilla/cc{4,8,32,64}/summary.json
  queue_gate/cc{4,8,32,64}/summary.json
  queue_gate_length_sort/cc{4,8,32,64}/summary.json
  vanilla/vanilla_server.log
  queue_gate/queue_gate_server.log
  queue_gate_length_sort/queue_gate_length_sort_server.log
```
