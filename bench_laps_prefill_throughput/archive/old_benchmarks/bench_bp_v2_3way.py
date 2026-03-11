"""3-way benchmark: no graph vs piecewise vs attn-in-graph."""
import time
from sglang import Engine


def benchmark(engine, prompts, sampling_params, warmup=10, repeat=50):
    for _ in range(warmup):
        engine.generate(prompts, sampling_params=sampling_params)

    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        engine.generate(prompts, sampling_params=sampling_params)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies.sort()
    trim = max(1, len(latencies) // 10)
    trimmed = latencies[trim:-trim]
    avg = sum(trimmed) / len(trimmed)
    p50 = latencies[len(latencies) // 2]
    mn = min(latencies)
    return avg, p50, mn


def main():
    model_path = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
    sampling_params = {"max_new_tokens": 1, "temperature": 0}

    test_cases = [
        ("batch=4, short", [
            "What is 2+2?",
            "Hello, how are you?",
            "Tell me a joke.",
            "The sky is blue.",
        ]),
        ("batch=4, medium", [
            "Explain the theory of relativity in simple terms for a high school student.",
            "Write a short poem about the ocean and its waves crashing on the shore.",
            "What are the main differences between Python and JavaScript programming?",
            "Describe the process of photosynthesis in plants and why it matters.",
        ]),
        ("batch=2, short", [
            "Hello world!",
            "Good morning!",
        ]),
    ]

    configs = [
        ("No graph", dict(
            enable_piecewise_cuda_graph=False,
            enable_batch_prefill_cuda_graph=False,
            disable_cuda_graph=True,
        )),
        ("Piecewise", dict(
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=False,
        )),
        ("Attn graph", dict(
            enable_piecewise_cuda_graph=True,
            enable_batch_prefill_cuda_graph=True,
            batch_prefill_max_seq_len=256,
            batch_prefill_batch_sizes=[1, 2, 4, 8],
            batch_prefill_seq_lengths=[16, 32, 64],
        )),
    ]

    all_results = {}  # config_name -> {test_name -> (avg, p50, min)}

    for config_name, config_kwargs in configs:
        print(f"\n=== Launching engine: {config_name} ===")
        engine = Engine(
            model_path=model_path,
            mem_fraction_static=0.7,
            cuda_graph_max_bs=32,
            **config_kwargs,
        )

        all_results[config_name] = {}
        for test_name, prompts in test_cases:
            avg, p50, mn = benchmark(engine, prompts, sampling_params)
            all_results[config_name][test_name] = (avg, p50, mn)
            print(f"  {test_name}: avg={avg:.2f}ms  p50={p50:.2f}ms  min={mn:.2f}ms")

        engine.shutdown()

    # Summary
    print("\n")
    print("=" * 95)
    print(f"{'Test Case':<20} {'No graph avg':>14} {'Piecewise avg':>14} {'Attn graph avg':>15} {'PW speedup':>12} {'AG speedup':>12}")
    print("=" * 95)
    for test_name, _ in test_cases:
        ng = all_results["No graph"][test_name][0]
        pw = all_results["Piecewise"][test_name][0]
        ag = all_results["Attn graph"][test_name][0]
        pw_sp = ng / pw
        ag_sp = ng / ag
        print(f"{test_name:<20} {ng:>12.2f}ms {pw:>12.2f}ms {ag:>13.2f}ms {pw_sp:>11.2f}x {ag_sp:>11.2f}x")

    print()
    print(f"{'Test Case':<20} {'No graph p50':>14} {'Piecewise p50':>14} {'Attn graph p50':>15} {'PW speedup':>12} {'AG speedup':>12}")
    print("=" * 95)
    for test_name, _ in test_cases:
        ng = all_results["No graph"][test_name][1]
        pw = all_results["Piecewise"][test_name][1]
        ag = all_results["Attn graph"][test_name][1]
        pw_sp = ng / pw
        ag_sp = ng / ag
        print(f"{test_name:<20} {ng:>12.2f}ms {pw:>12.2f}ms {ag:>13.2f}ms {pw_sp:>11.2f}x {ag_sp:>11.2f}x")
    print("=" * 95)


if __name__ == "__main__":
    main()
