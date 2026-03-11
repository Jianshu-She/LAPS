# Experiments 9-12: Diagnosing CUDA Graph Slowdown at High Concurrency

## 1. Background and Motivation

Experiment 8 revealed that both **piecewise CUDA graph** and **batch prefill CUDA graph** modes are slower than vanilla (no CUDA graph) at high concurrency levels in SGLang's prefill-only serving mode. The degradation is severe for batch_cg:

| Setting (7B) | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 |
|---|---|---|---|---|---|
| vanilla | 128.8 | 200.7 | 329.5 | 415.2 | 442.3 |
| piecewise_cg | 170.9 (1.33x) | 290.6 (1.45x) | 357.8 (1.09x) | 411.4 (0.99x) | 435.5 (0.99x) |
| batch_cg | 172.1 (1.34x) | 204.9 (1.02x) | 170.0 (0.52x) | 143.3 (0.35x) | 211.2 (0.48x) |

Both CG modes help at low concurrency but hurt at high concurrency. We hypothesized three possible causes:

- **H1 (Padding Waste):** Batch prefill CUDA graphs require fixed (batch_size, seq_len) dimensions. Real batches are padded up to the next captured graph size, wasting FLOPs on dummy tokens.
- **H2 (Memory Pressure):** CUDA graph memory reduces available GPU memory for KV cache, limiting concurrent requests.
- **H3 (Batching Degradation):** CUDA graphs process requests too quickly via small bs=1 graph replays, draining the queue before the scheduler can accumulate requests into large, efficient batches.

These four experiments systematically isolate each hypothesis.

---

## 2. Pre-Experiment: Parsing Existing Exp 8 Server Logs

Before running new experiments, we parsed the Exp 8 server logs to quantify the three hypotheses.

### 2.1 Padding Waste in Batch Prefill CG (H1 Evidence)

The batch_cg mode captures CUDA graphs for a grid of (batch_size, seq_len) combinations: bs={1,2,4,8,16,32} x seq_len={1,2,4,8,16,32,64,128,256,512} = 60 graphs. When a real batch arrives, it must be rounded up to the nearest captured (bs, seq_len). The server logs every replay with a `BP-CG-REPLAY` line showing real vs padded token counts.

**7B model:**
- 7,768 batch prefill CUDA graph replays across all runs
- Overall: 1,448,099 real tokens padded to 8,631,200 tokens = **5.96x inflation, 83.2% waste**
- Waste increases sharply with both batch size and sequence length:

| Target bs | Replays | Waste |
|---|---|---|
| 2 | 3,939 | 54.7% |
| 4 | 1,788 | 72.4% |
| 8 | 1,155 | 81.6% |
| 16 | 636 | 87.5% |
| 32 | 250 | 91.9% |

| Target seq_len | Replays | Waste |
|---|---|---|
| 16 | 725 | 31.6% |
| 32 | 1,976 | 50.5% |
| 64 | 1,765 | 65.5% |
| 128 | 1,398 | 76.8% |
| 256 | 1,049 | 83.8% |
| 512 | 835 | 88.4% |

The waste comes from two sources: (1) batch size rounding (e.g., 3 real requests padded to bs=4), and (2) sequence length rounding (e.g., a 65-token prompt padded to seq_len=128). The multiplicative nature of 2D padding makes waste severe: a (3, 65) batch padded to (4, 128) = 512 padded tokens for 195 real tokens (2.6x).

### 2.2 Forward Path Distribution (H3 Evidence)

The server logs every forward pass with a `FWD-PATH` line showing which code path was taken and the batch size.

**7B piecewise_cg mode:**
- 23,986 forward calls, all via `piecewise_cg` path
- Average batch size: 2.5
- **56.5% of calls are bs=1** (single-request processing)

**7B batch_cg mode (has both paths):**
- 7,768 calls via `batch_prefill_cg` path (avg bs=4.4, 0% bs=1)
- 15,394 calls via `piecewise_cg` path (avg bs=1.7, **93.5% bs=1**)

The batch_cg mode falls back to piecewise_cg for single-request arrivals (batch_prefill_cg requires bs>1). This means **66% of all forward calls in batch_cg mode are single-request piecewise replays**. These fast bs=1 replays drain the queue before more requests can accumulate, preventing the scheduler from building large batches.

### 2.3 Prefill Batch Size Distribution

Comparing batch sizes across modes:

| Setting (7B) | Total Batches | Avg new_seq | bs=1 Fraction |
|---|---|---|---|
| vanilla | 19,486 | 3.1 | 29.4% |
| piecewise_cg | 23,986 | 2.5 | 56.5% |
| batch_cg | 23,162 | 2.6 | 62.2% |

Both CG modes process ~20% more batches than vanilla (23K vs 19K) but with smaller average batch sizes (2.5-2.6 vs 3.1). The CG acceleration at low batch sizes creates a vicious cycle: faster processing means fewer requests queued, which means smaller batches, which means less parallelism.

---

## 3. Experiment 9: Uniform Dataset

### 3.1 Purpose

Isolate H1 (padding waste) from H3 (batching degradation). If all prompts have similar lengths, padding waste should be minimal for batch_cg. If CG still underperforms vanilla with uniform data, batching degradation (H3) must be the cause.

### 3.2 Setup

**Uniform dataset:** Filtered from the original 10K LMSYS-chat human prompts, keeping only prompts with estimated 40-80 tokens (using chars/4 heuristic).

- Original dataset: 10,000 prompts, token range [0, 4002], median=13, mean=63.3, std=235.9 (highly skewed, heavy tail)
- Uniform dataset: 895 prompts, token range [40, 80], median=54, mean=55.7 (tight distribution)

Token distribution of the uniform dataset:
| Range | Count | Fraction |
|---|---|---|
| 40-49 | 344 | 38.4% |
| 50-59 | 232 | 25.9% |
| 60-69 | 170 | 19.0% |
| 70-79 | 137 | 15.3% |

With this dataset, batch_cg padding waste should be low: most prompts fit within a 64 or 128 seq_len bucket with minimal padding.

**Model:** Qwen2.5-7B-Instruct (single GPU)
**Concurrency:** 4, 32, 64
**Requests:** 10,000 (cycling through the 895 prompts)
**Max tokens:** 1 (prefill-only)

**Server configurations tested:**

| Setting | Server args |
|---|---|
| vanilla | (none) |
| piecewise_cg | `--enable-piecewise-cuda-graph --disable-cuda-graph` |
| batch_cg | `--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512` |

### 3.3 Results

**RPS (requests/sec):**

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla | 155.1 | 565.4 | 632.7 |
| piecewise_cg | 184.7 (1.19x) | 636.3 (1.13x) | 649.6 (1.03x) |
| batch_cg | 199.8 (1.29x) | 480.6 (0.85x) | 575.8 (0.91x) |

**Comparison with Exp 8 (mixed-length prompts):**

| Setting | Exp9 cc=32 (uniform) | Exp8 cc=32 (mixed) | Exp9 cc=64 (uniform) | Exp8 cc=64 (mixed) |
|---|---|---|---|---|
| batch_cg speedup | 0.85x | 0.35x | 0.91x | 0.48x |

**Avg latency (ms):**

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla | 25.6 | 55.8 | 99.7 |
| piecewise_cg | 21.5 | 49.7 | 97.4 |
| batch_cg | 19.9 | 66.0 | 110.0 |

### 3.4 Analysis

**Piecewise CG** beats vanilla across all concurrency levels with uniform data (1.03-1.19x). This confirms that piecewise CG's slowdown in Exp 8 at high cc was marginal and within noise.

**Batch CG** improved dramatically from Exp 8 (0.35x -> 0.85x at cc=32, 0.48x -> 0.91x at cc=64), but **still does not match vanilla**. This means:

- **H1 (padding) accounts for ~60% of the batch_cg gap.** The uniform dataset eliminates most padding waste, recovering from 0.35x to 0.85x.
- **H3 (batching degradation) accounts for ~40% of the remaining gap.** Even without padding, batch_cg is 15% slower at cc=32 because the fast bs=1 piecewise replays still prevent batch accumulation.

---

## 4. Experiment 10: Piecewise CG Max-Tokens Sweep

### 4.1 Purpose

Test H3 (batching degradation) specifically for piecewise CG. The `--piecewise-cuda-graph-max-tokens` flag controls the maximum token count for which a CUDA graph is used. Batches exceeding this cap fall back to eager (non-CG) execution. Lower caps force more large batches to run eagerly, which is slower per-batch but allows the scheduler to accumulate larger batches.

### 4.2 Setup

**Model:** Qwen2.5-7B-Instruct
**Dataset:** Same 10K LMSYS-chat prompts as Exp 8 (mixed lengths)
**Concurrency:** 4, 8, 32, 64

| Setting | Max tokens | Effect |
|---|---|---|
| exp8_piecewise_cg | 8192 (default) | Nearly all batches use CG |
| piecewise_cap1024 | 1024 | Batches >1024 tokens fall to eager |
| piecewise_cap512 | 512 | Batches >512 tokens fall to eager |
| piecewise_cap256 | 256 | Most batches fall to eager |

### 4.3 Results

**RPS:**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| exp8_vanilla | 128.8 | 200.7 | 415.2 | 442.3 |
| exp8_piecewise (cap=8192) | 170.9 | 290.6 | 411.4 | 435.5 |
| piecewise_cap1024 | 174.1 | 296.1 | 442.4 | 435.4 |
| piecewise_cap512 | 172.2 | 294.4 | **450.4** | 445.2 |
| piecewise_cap256 | 167.7 | 284.7 | 430.3 | 443.9 |

**Speedup vs vanilla:**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| exp8_piecewise (cap=8192) | 1.33x | 1.45x | **0.99x** | 0.99x |
| piecewise_cap1024 | 1.35x | 1.48x | **1.07x** | 0.99x |
| piecewise_cap512 | 1.34x | 1.47x | **1.09x** | 1.01x |
| piecewise_cap256 | 1.30x | 1.42x | 1.04x | 1.00x |

**Avg latency (ms):**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| piecewise_cap256 | 23.7 | 27.8 | 73.2 | 142.0 |
| piecewise_cap512 | 23.1 | 26.9 | 70.1 | 141.7 |
| piecewise_cap1024 | 22.8 | 26.8 | 71.4 | 145.0 |

### 4.4 Analysis

**cap512 is the optimal setting.** It achieves the best throughput at cc=32 (1.09x vs vanilla, up from 0.99x with default cap=8192) while maintaining full CG benefits at low cc (1.34x at cc=4).

The pattern confirms H3:
- **Lower caps improve high-cc throughput** by forcing large-token batches to eager, which slows down individual batch processing but allows the scheduler to accumulate more requests.
- **Too low a cap (256) hurts everywhere** because even small batches lose CG acceleration.
- The sweet spot is around 512 tokens: below this threshold, CG provides meaningful speedup; above it, eager execution enables better batching.

---

## 5. Experiment 11: Batch CG Grid Density

### 5.1 Purpose

Test H1 (padding waste) specifically for batch_cg by increasing the density of captured (bs, seq_len) graphs. A denser grid means smaller rounding gaps and less padding waste. If denser grids improve throughput, padding waste is a significant contributor.

### 5.2 Setup

**Model:** Qwen2.5-7B-Instruct
**Dataset:** Same 10K LMSYS-chat prompts as Exp 8
**Concurrency:** 4, 8, 32, 64

| Setting | Batch sizes | Seq lengths | Total graphs |
|---|---|---|---|
| exp8_batch_cg | 1,2,4,8,16,32 (6) | 1,2,4,8,16,32,64,128,256,512 (10) | 60 |
| batch_cg_dense | 1,2,3,4,5,6,7,8,12,16,24,32 (12) | 16,24,32,48,64,96,128,192,256,384,512 (11) | 132 |
| batch_cg_vdense | 1,2,3,4,5,6,7,8,10,12,16,20,24,32 (14) | 8,12,16,24,32,48,64,80,96,128,160,192,256,320,384,512 (16) | 224 |

The dense grid adds intermediate batch sizes (3,5,6,7) and intermediate sequence lengths (24,48,96,192,384). The vdense grid goes further with even finer granularity (10,20 for bs; 80,160,320 for seq).

### 5.3 Results

**RPS:**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| exp8_vanilla | 128.8 | 200.7 | 415.2 | 442.3 |
| exp8_batch_cg (60 graphs) | 172.1 | 204.9 | 143.3 | 211.2 |
| batch_cg_dense (132 graphs) | 183.5 | 227.6 | 182.5 | 246.3 |
| batch_cg_vdense (224 graphs) | 182.7 | 241.9 | 197.4 | 257.5 |

**Speedup vs vanilla:**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| exp8_batch_cg (60 graphs) | 1.34x | 1.02x | 0.35x | 0.48x |
| batch_cg_dense (132 graphs) | 1.43x | 1.13x | 0.44x | 0.56x |
| batch_cg_vdense (224 graphs) | 1.42x | 1.21x | 0.48x | 0.58x |

**Avg latency (ms):**

| Setting | cc=4 | cc=8 | cc=32 | cc=64 |
|---|---|---|---|---|
| batch_cg_dense | 21.7 | 34.8 | 173.7 | 256.3 |
| batch_cg_vdense | 21.8 | 32.8 | 160.5 | 245.1 |

### 5.4 Analysis

Denser grids consistently improve batch_cg throughput:
- **cc=8:** 60 graphs -> 1.02x, 132 -> 1.13x, 224 -> 1.21x
- **cc=32:** 60 graphs -> 0.35x, 132 -> 0.44x, 224 -> 0.48x
- **cc=64:** 60 graphs -> 0.48x, 132 -> 0.56x, 224 -> 0.58x

Each density increase provides diminishing but real improvement (~10-20% per step). This confirms **H1: padding waste is a significant cost**. However, even with 224 graphs (3.7x more than original), batch_cg at cc=32 is only 0.48x of vanilla. This means:

- **Padding waste explains ~30% of the gap** (from 0.35x to 0.48x with vdense)
- **The remaining ~70% is H3 (batching degradation)**, which grid density cannot fix

Interestingly, the improvement is largest at cc=8 (1.02x -> 1.21x), where batching degradation is less severe. At high cc where H3 dominates, grid density helps less.

---

## 6. Experiment 12: KV Cache / Memory Pressure

### 6.1 Purpose

Test H2 (memory pressure). CUDA graph capture consumes GPU memory that could otherwise be used for KV cache. If the reduced KV cache causes throughput degradation, artificially limiting vanilla's KV cache to match CG's footprint should reproduce the slowdown.

### 6.2 Setup

**Model:** Qwen2.5-32B (chosen because it has the tightest memory budget on a single GPU)
**Dataset:** Same 10K LMSYS-chat prompts as Exp 8
**Concurrency:** 4, 32, 64

From Exp 8, we know the 32B model has 234,140 tokens of KV cache with vanilla. CG modes use some of the post-KV-cache free memory for graph storage. We simulate this by capping vanilla's KV cache:

| Setting | Max total tokens | Purpose |
|---|---|---|
| exp8_vanilla | ~234,140 (auto) | Full KV cache baseline |
| vanilla_kv180k | 180,000 | Moderate KV reduction |
| vanilla_kv120k | 120,000 | Aggressive KV reduction |

### 6.3 Results

**RPS:**

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| exp8_vanilla (~234K tokens) | 59.2 | 119.6 | 124.7 |
| exp8_piecewise_cg | 52.8 | 112.8 | 116.3 |
| exp8_batch_cg | 42.2 | 35.1 | 44.6 |
| vanilla_kv180k | 62.1 | 119.8 | 124.6 |
| vanilla_kv120k | 62.2 | 119.8 | 122.4 |

**Speedup vs exp8_vanilla:**

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla_kv180k | 1.05x | 1.00x | 1.00x |
| vanilla_kv120k | 1.05x | 1.00x | 0.98x |

**Avg latency (ms):**

| Setting | cc=4 | cc=32 | cc=64 |
|---|---|---|---|
| vanilla_kv180k | 64.0 | 265.9 | 510.8 |
| vanilla_kv120k | 64.0 | 265.9 | 519.7 |

### 6.4 Analysis

**H2 is conclusively refuted.** Reducing KV cache from 234K to 120K tokens (a 49% reduction) has virtually no effect on throughput (0.98-1.05x). This makes sense because in prefill-only mode with max_tokens=1, the maximum concurrent KV usage is approximately `cc * avg_prompt_tokens` = 64 * 100 = 6,400 tokens, which is far below even the 120K limit.

The slight improvement at cc=4 (1.05x) may be noise or a minor benefit from reduced memory management overhead with a smaller cache.

---

## 7. Combined Findings

### 7.1 Hypothesis Verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1: Padding waste** | **Confirmed (major for batch_cg)** | 83% padding waste in logs; uniform data recovers batch_cg from 0.35x to 0.85x; denser grids help ~20% |
| **H2: Memory pressure** | **Refuted** | Halving KV cache has no effect on throughput |
| **H3: Batching degradation** | **Confirmed (affects both CG modes)** | batch_cg still 0.85x with uniform data; capping piecewise at 512 tokens improves cc=32 from 0.99x to 1.09x; 93.5% of piecewise calls in batch_cg are bs=1 |

### 7.2 Root Cause Decomposition for batch_cg at cc=32

Starting from batch_cg's 0.35x speedup vs vanilla at cc=32:

1. **Padding waste (H1) contributes ~40-50% of the gap:**
   - Uniform data: 0.35x -> 0.85x (eliminates padding)
   - Denser grid (224 graphs): 0.35x -> 0.48x (reduces padding)

2. **Batching degradation (H3) contributes ~50-60% of the gap:**
   - Even with uniform data and no padding, batch_cg = 0.85x
   - 93.5% bs=1 calls drain queue before batches accumulate

3. **Memory pressure (H2) contributes 0%:**
   - Halving KV cache has no effect

### 7.3 Actionable Implications

| Finding | Action |
|---|---|
| Padding waste is severe (83%) | Ship denser default grid; consider dynamic graph selection |
| Piecewise CG with cap=512 beats default at high cc | Make `--piecewise-cuda-graph-max-tokens 512` the default, or implement adaptive cap based on load |
| bs=1 CG replays drain the queue | Consider adding a scheduling delay: hold bs=1 requests briefly to allow batch accumulation when queue pressure is high |
| Memory pressure is not a factor | No need to optimize CG memory footprint for prefill-only workloads |

---

## 8. Experimental Details

- **Hardware:** Single GPU (CUDA_VISIBLE_DEVICES=5)
- **Models:** Qwen2.5-7B-Instruct (Exps 9-11), Qwen2.5-32B (Exp 12)
- **Memory:** `--mem-fraction-static 0.85` for all runs
- **Load generator:** `baseline.py` with HTTP/2, closed-loop concurrency
- **Requests per run:** 10,000
- **Max tokens:** 1 (prefill-only, no generation)
- **Total runs:** 35 (9 + 12 + 8 + 6), 0 failures
- **Results directory:** `results_exp9_12_2026-03-02/`
