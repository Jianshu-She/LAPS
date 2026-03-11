"""Benchmark batch prefill CUDA graph v2 (attention in graph) vs baseline."""
import time
from sglang import Engine


def benchmark(engine, prompts, sampling_params, warmup=10, repeat=50):
    """Run warmup + timed iterations, return average latency in ms."""
    for _ in range(warmup):
        engine.generate(prompts, sampling_params=sampling_params)

    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies.sort()
    # Drop top/bottom 10% outliers
    trim = max(1, len(latencies) // 10)
    trimmed = latencies[trim:-trim]
    avg = sum(trimmed) / len(trimmed)
    mn = min(latencies)
    mx = max(latencies)
    p50 = latencies[len(latencies) // 2]
    return avg, mn, mx, p50


def main():
    model_path = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    # Different batch sizes and prompt lengths to test
    test_cases = [
        ("batch=4, short prompts", [
            "What is 2+2?",
            "Hello, how are you?",
            "Tell me a joke.",
            "The sky is blue.",
        ]),
        ("batch=4, medium prompts", [
            "Explain the theory of relativity in simple terms for a high school student.",
            "Write a short poem about the ocean and its waves crashing on the shore.",
            "What are the main differences between Python and JavaScript programming?",
            "Describe the process of photosynthesis in plants and why it matters.",
        ]),
        ("batch=2, short prompts", [
            "Hello world!",
            "Good morning!",
        ]),
    ]

    # --- Baseline: piecewise only (no batch prefill graph) ---
    print("=== Launching BASELINE engine (piecewise only) ===")
    baseline = Engine(
        model_path=model_path,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=False,
        mem_fraction_static=0.7,
        cuda_graph_max_bs=32,
    )

    baseline_results = {}
    for name, prompts in test_cases:
        avg, mn, mx, p50 = benchmark(baseline, prompts, sampling_params)
        baseline_results[name] = (avg, mn, mx, p50)
        print(f"  {name}: avg={avg:.2f}ms  p50={p50:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

    baseline.shutdown()
    print()

    # --- Test: batch prefill CUDA graph v2 (attention in graph) ---
    print("=== Launching BATCH PREFILL V2 engine (attention in graph) ===")
    test = Engine(
        model_path=model_path,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4, 8],
        batch_prefill_seq_lengths=[16, 32, 64],
        mem_fraction_static=0.7,
        cuda_graph_max_bs=32,
    )

    test_results = {}
    for name, prompts in test_cases:
        avg, mn, mx, p50 = benchmark(test, prompts, sampling_params)
        test_results[name] = (avg, mn, mx, p50)
        print(f"  {name}: avg={avg:.2f}ms  p50={p50:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms")

    test.shutdown()
    print()

    # --- Summary ---
    print("=" * 80)
    print(f"{'Test Case':<35} {'Base avg':>10} {'V2 avg':>10} {'Base p50':>10} {'V2 p50':>10} {'Speedup':>10}")
    print("=" * 80)
    for name in baseline_results:
        b_avg, _, _, b_p50 = baseline_results[name]
        t_avg, _, _, t_p50 = test_results[name]
        speedup = b_avg / t_avg
        print(f"{name:<35} {b_avg:>8.2f}ms {t_avg:>8.2f}ms {b_p50:>8.2f}ms {t_p50:>8.2f}ms {speedup:>9.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
