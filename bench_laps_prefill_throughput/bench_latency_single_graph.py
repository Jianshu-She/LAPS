"""
Focused microbenchmark: only capture bs=8 seq_len=128 graph.
Tests whether capturing fewer graphs improves batch prefill performance.

Usage:
    CUDA_VISIBLE_DEVICES=2 python bench_latency_single_graph.py --model Qwen/Qwen2.5-32B
"""

import argparse
import time
import numpy as np
from sglang import Engine


def make_prompts(num_prompts, target_tokens):
    """Create prompts that are approximately target_tokens long."""
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    chars_needed = target_tokens * 4
    prompt = base_text[:chars_needed]
    return [prompt] * num_prompts


def benchmark_latency(engine, prompts, sampling_params, warmup=20, repeat=200):
    """Measure per-batch latency with more iterations for stability."""
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
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()

    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    # Only test bs=8, seq_len=128
    test_bs = 8
    test_seq_len = 128
    prompts = make_prompts(test_bs, test_seq_len)

    # --- Mode 1: Piecewise only ---
    print("=" * 70)
    print(f"  MODE 1: Piecewise CUDA Graph (attention OUTSIDE graph)")
    print(f"  Model: {args.model}")
    print("=" * 70)
    engine1 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=False,
        mem_fraction_static=args.mem_fraction,
    )
    r1 = benchmark_latency(engine1, prompts, sampling_params, args.warmup, args.repeat)
    print(f"  bs={test_bs}, seq_len={test_seq_len}: "
          f"p50={r1['p50']:.2f}ms  mean={r1['mean']:.2f}±{r1['std']:.2f}ms  "
          f"min={r1['min']:.2f}ms  p10={r1['p10']:.2f}ms  p90={r1['p90']:.2f}ms")
    engine1.shutdown()
    del engine1
    print()

    # --- Mode 2: Batch Prefill - ONLY bs=8, seq_len=128 ---
    print("=" * 70)
    print(f"  MODE 2: Batch Prefill CUDA Graph (ONLY bs=8, seq_len=128)")
    print(f"  Model: {args.model}")
    print("=" * 70)
    engine2 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=128,
        batch_prefill_batch_sizes=[8],
        batch_prefill_seq_lengths=[128],
        mem_fraction_static=args.mem_fraction,
    )
    r2 = benchmark_latency(engine2, prompts, sampling_params, args.warmup, args.repeat)
    print(f"  bs={test_bs}, seq_len={test_seq_len}: "
          f"p50={r2['p50']:.2f}ms  mean={r2['mean']:.2f}±{r2['std']:.2f}ms  "
          f"min={r2['min']:.2f}ms  p10={r2['p10']:.2f}ms  p90={r2['p90']:.2f}ms")
    engine2.shutdown()
    del engine2
    print()

    # --- Mode 3: Batch Prefill - all graphs (for comparison) ---
    print("=" * 70)
    print(f"  MODE 3: Batch Prefill CUDA Graph (ALL graphs: bs=[1,2,4,8] x seq=[16..256])")
    print(f"  Model: {args.model}")
    print("=" * 70)
    engine3 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4, 8],
        batch_prefill_seq_lengths=[16, 32, 64, 128, 256],
        mem_fraction_static=args.mem_fraction,
    )
    r3 = benchmark_latency(engine3, prompts, sampling_params, args.warmup, args.repeat)
    print(f"  bs={test_bs}, seq_len={test_seq_len}: "
          f"p50={r3['p50']:.2f}ms  mean={r3['mean']:.2f}±{r3['std']:.2f}ms  "
          f"min={r3['min']:.2f}ms  p10={r3['p10']:.2f}ms  p90={r3['p90']:.2f}ms")
    engine3.shutdown()
    del engine3
    print()

    # --- Summary ---
    speedup_single = r1['p50'] / r2['p50']
    speedup_all = r1['p50'] / r3['p50']
    print("=" * 70)
    print(f"  SUMMARY: bs={test_bs}, seq_len={test_seq_len}")
    print(f"  Model: {args.model}, Warmup: {args.warmup}, Repeat: {args.repeat}")
    print("=" * 70)
    print(f"  Piecewise:                p50={r1['p50']:>7.2f}ms  mean={r1['mean']:>7.2f}±{r1['std']:.2f}ms")
    print(f"  BatchPF (single graph):   p50={r2['p50']:>7.2f}ms  mean={r2['mean']:>7.2f}±{r2['std']:.2f}ms  speedup={speedup_single:.3f}x")
    print(f"  BatchPF (all graphs):     p50={r3['p50']:>7.2f}ms  mean={r3['mean']:>7.2f}±{r3['std']:.2f}ms  speedup={speedup_all:.3f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
