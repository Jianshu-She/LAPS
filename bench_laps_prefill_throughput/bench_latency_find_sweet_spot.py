"""
Find where Batch Prefill CUDA Graph beats Piecewise on 32B model.

Hypothesis: batch prefill should win when kernel launch overhead is a significant
fraction of total time. This means:
  - Small batch sizes (bs=1) where total compute is low
  - Very short sequences (seq_len=4,8,16) where compute per layer is tiny
  - The ~64 piecewise kernel launches (one per attention layer boundary) become
    a larger fraction of the ~5-10ms total forward pass

We test a grid of (bs, seq_len) pairs focused on the small end, comparing
Piecewise vs Batch Prefill (with matching captured graphs).

Usage:
    CUDA_VISIBLE_DEVICES=2 python bench_latency_find_sweet_spot.py --model Qwen/Qwen2.5-32B
"""

import argparse
import time
import numpy as np
from sglang import Engine


def make_prompts(num_prompts, target_tokens):
    """Create prompts that are approximately target_tokens long."""
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    chars_needed = max(target_tokens * 4, 10)  # at least 10 chars
    prompt = base_text[:chars_needed]
    return [prompt] * num_prompts


def benchmark_latency(engine, prompts, sampling_params, warmup=20, repeat=300):
    """Measure per-batch latency."""
    for _ in range(warmup):
        engine.generate(prompts, sampling_params=sampling_params)

    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    arr = np.array(latencies)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "p50": np.percentile(arr, 50),
        "p10": np.percentile(arr, 10),
        "p90": np.percentile(arr, 90),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=300)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()

    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    # Test cases: focus on small bs and short seq where launch overhead matters
    test_cases = [
        (1, 4),
        (1, 8),
        (1, 16),
        (1, 32),
        (1, 64),
        (1, 128),
        (2, 4),
        (2, 8),
        (2, 16),
        (2, 32),
        (2, 64),
        (4, 4),
        (4, 8),
        (4, 16),
        (4, 32),
        (8, 4),
        (8, 8),
        (8, 16),
    ]

    # Batch prefill graph dimensions to capture — cover all test cases
    bp_batch_sizes = [1, 2, 4, 8]
    bp_seq_lengths = [4, 8, 16, 32, 64, 128]

    # --- Mode 1: Piecewise only ---
    print("=" * 90)
    print(f"  MODE 1: Piecewise CUDA Graph (attention OUTSIDE graph)")
    print(f"  Model: {args.model}")
    print("=" * 90)
    engine1 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=False,
        mem_fraction_static=args.mem_fraction,
    )
    results_pw = {}
    for bs, seq_len in test_cases:
        prompts = make_prompts(bs, seq_len)
        r = benchmark_latency(engine1, prompts, sampling_params, args.warmup, args.repeat)
        results_pw[(bs, seq_len)] = r
        print(f"  bs={bs:>2}, seq_len={seq_len:>4}: "
              f"p50={r['p50']:>7.2f}ms  mean={r['mean']:>7.2f}±{r['std']:.2f}ms  "
              f"min={r['min']:>7.2f}ms")
    engine1.shutdown()
    del engine1
    print()

    # --- Mode 2: Batch Prefill CUDA Graph ---
    print("=" * 90)
    print(f"  MODE 2: Batch Prefill CUDA Graph (attention INSIDE graph)")
    print(f"  Captured: bs={bp_batch_sizes}, seq={bp_seq_lengths}")
    print(f"  Model: {args.model}")
    print("=" * 90)
    engine2 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=max(bp_seq_lengths),
        batch_prefill_batch_sizes=bp_batch_sizes,
        batch_prefill_seq_lengths=bp_seq_lengths,
        mem_fraction_static=args.mem_fraction,
    )
    results_bp = {}
    for bs, seq_len in test_cases:
        prompts = make_prompts(bs, seq_len)
        r = benchmark_latency(engine2, prompts, sampling_params, args.warmup, args.repeat)
        results_bp[(bs, seq_len)] = r
        print(f"  bs={bs:>2}, seq_len={seq_len:>4}: "
              f"p50={r['p50']:>7.2f}ms  mean={r['mean']:>7.2f}±{r['std']:.2f}ms  "
              f"min={r['min']:>7.2f}ms")
    engine2.shutdown()
    del engine2
    print()

    # --- Comparison ---
    print("=" * 90)
    print(f"  COMPARISON: Piecewise vs Batch Prefill — Finding the Sweet Spot")
    print(f"  Model: {args.model}, Warmup: {args.warmup}, Repeat: {args.repeat}")
    print("=" * 90)
    print(f"{'bs':>4}  {'seq_len':>8}  {'total_tok':>10}  {'PW p50':>10}  {'BP p50':>10}  "
          f"{'Speedup':>10}  {'PW mean±std':>16}  {'BP mean±std':>16}")
    print("-" * 90)

    for bs, seq_len in test_cases:
        pw = results_pw[(bs, seq_len)]
        bp = results_bp[(bs, seq_len)]
        total_tok = bs * seq_len
        speedup = pw["p50"] / bp["p50"] if bp["p50"] > 0 else float("inf")
        if speedup > 1.05:
            marker = " <-- BP WINS"
        elif speedup < 0.95:
            marker = " <-- PW WINS"
        else:
            marker = ""
        print(f"{bs:>4}  {seq_len:>8}  {total_tok:>10}  {pw['p50']:>7.2f}ms  {bp['p50']:>7.2f}ms  "
              f"{speedup:>9.3f}x  "
              f"{pw['mean']:>7.2f}±{pw['std']:>5.2f}  {bp['mean']:>7.2f}±{bp['std']:>5.2f}{marker}")

    print("-" * 90)
    print("  Speedup > 1.0 means Batch Prefill is faster")
    print("=" * 90)


if __name__ == "__main__":
    main()
