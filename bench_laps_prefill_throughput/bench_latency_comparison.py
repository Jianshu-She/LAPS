"""
Microbenchmark: Piecewise CUDA graph vs Batch Prefill CUDA graph latency.

Tests raw per-batch prefill latency across (batch_size, seq_len) combinations
to isolate the CUDA graph benefit. Uses the SGLang Engine (offline mode) so
there's no scheduler/router overhead — just pure model execution.

Usage:
    # 0.5B model (quick test)
    python bench_latency_comparison.py --model /mnt/weka/home/jianshu.she/models/qwen2.5-0.5b

    # 32B model (realistic)
    CUDA_VISIBLE_DEVICES=4 python bench_latency_comparison.py --model Qwen/Qwen2.5-32B
"""

import argparse
import time
import numpy as np
from sglang import Engine


def make_prompts(num_prompts, target_tokens):
    """Create prompts that are approximately target_tokens long."""
    # ~4 chars per token for English text
    base_text = "The quick brown fox jumps over the lazy dog. " * 100
    chars_needed = target_tokens * 4
    prompt = base_text[:chars_needed]
    return [prompt] * num_prompts


def benchmark_latency(engine, prompts, sampling_params, warmup=10, repeat=100):
    """Measure per-batch latency."""
    # Warmup
    for _ in range(warmup):
        engine.generate(prompts, sampling_params=sampling_params)

    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        latencies.append(elapsed)

    arr = np.array(latencies)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "p50": np.percentile(arr, 50),
        "p90": np.percentile(arr, 90),
        "p99": np.percentile(arr, 99),
        "min": np.min(arr),
        "max": np.max(arr),
    }


def run_all_tests(engine, label, test_cases, sampling_params, warmup, repeat):
    """Run all test cases and return results dict."""
    results = {}
    for bs, seq_len in test_cases:
        prompts = make_prompts(bs, seq_len)
        stats = benchmark_latency(engine, prompts, sampling_params, warmup, repeat)
        results[(bs, seq_len)] = stats
        print(f"  bs={bs:>2}, seq_len={seq_len:>4}: "
              f"mean={stats['mean']:>7.2f}ms  "
              f"p50={stats['p50']:>7.2f}ms  "
              f"std={stats['std']:>6.2f}ms  "
              f"min={stats['min']:>7.2f}ms")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--mem-fraction", type=float, default=0.85)
    args = parser.parse_args()

    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    # Test cases: (batch_size, approx_seq_len)
    test_cases = [
        # Single request (baseline, both should use piecewise)
        (1, 16),
        (1, 64),
        (1, 128),
        (1, 256),
        # Batch prefill target cases (bs > 1)
        (2, 16),
        (2, 64),
        (2, 128),
        (2, 256),
        (4, 16),
        (4, 64),
        (4, 128),
        (4, 256),
        (8, 16),
        (8, 64),
        (8, 128),
        (8, 256),
    ]

    # --- Mode 1: Piecewise only ---
    print("=" * 80)
    print(f"  MODE 1: Piecewise CUDA Graph (attention OUTSIDE graph)")
    print(f"  Model: {args.model}")
    print("=" * 80)
    engine1 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=False,
        mem_fraction_static=args.mem_fraction,
    )
    results_piecewise = run_all_tests(
        engine1, "Piecewise", test_cases, sampling_params, args.warmup, args.repeat
    )
    engine1.shutdown()
    del engine1
    print()

    # --- Mode 2: Batch Prefill CUDA Graph ---
    print("=" * 80)
    print(f"  MODE 2: Batch Prefill CUDA Graph (attention INSIDE graph)")
    print(f"  Model: {args.model}")
    print("=" * 80)
    engine2 = Engine(
        model_path=args.model,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4, 8],
        batch_prefill_seq_lengths=[16, 32, 64, 128, 256],
        mem_fraction_static=args.mem_fraction,
    )
    results_batch_prefill = run_all_tests(
        engine2, "BatchPrefill", test_cases, sampling_params, args.warmup, args.repeat
    )
    engine2.shutdown()
    del engine2
    print()

    # --- Comparison ---
    print("=" * 90)
    print(f"  COMPARISON: Piecewise vs Batch Prefill CUDA Graph")
    print(f"  Model: {args.model}, Warmup: {args.warmup}, Repeat: {args.repeat}")
    print("=" * 90)
    print(f"{'bs':>4}  {'seq_len':>8}  {'Piecewise p50':>14}  {'BatchPF p50':>14}  {'Speedup':>10}  {'PW mean±std':>16}  {'BP mean±std':>16}")
    print("-" * 90)

    for bs, seq_len in test_cases:
        pw = results_piecewise[(bs, seq_len)]
        bp = results_batch_prefill[(bs, seq_len)]
        speedup = pw["p50"] / bp["p50"] if bp["p50"] > 0 else float("inf")
        marker = " ***" if speedup > 1.05 else (" !!!" if speedup < 0.95 else "")
        print(f"{bs:>4}  {seq_len:>8}  {pw['p50']:>11.2f}ms  {bp['p50']:>11.2f}ms  {speedup:>9.2f}x  "
              f"{pw['mean']:>7.2f}±{pw['std']:>5.2f}  {bp['mean']:>7.2f}±{bp['std']:>5.2f}{marker}")

    print("-" * 90)
    print("  *** = batch prefill faster (>5%), !!! = batch prefill slower (>5%)")
    print("=" * 90)


if __name__ == "__main__":
    main()
