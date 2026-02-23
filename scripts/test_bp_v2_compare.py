"""Compare batch prefill CUDA graph v2 outputs against baseline (no graph)."""
from sglang import Engine


def main():
    model_path = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"

    prompts = [
        "What is 2+2?",
        "Hello, how are you?",
        "Tell me a joke.",
        "The sky is blue.",
        "Once upon a time",
        "The capital of France is",
    ]
    sampling_params = {"max_new_tokens": 32, "temperature": 0}

    # Baseline: no batch prefill CUDA graph
    print("=== Launching baseline engine (piecewise only) ===")
    baseline = Engine(
        model_path=model_path,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=False,
        mem_fraction_static=0.7,
        cuda_graph_max_bs=32,
    )
    baseline_outputs = baseline.generate(prompts, sampling_params=sampling_params)
    baseline.shutdown()

    # Test: with batch prefill CUDA graph v2
    print("=== Launching batch prefill v2 engine ===")
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
    test_outputs = test.generate(prompts, sampling_params=sampling_params)
    test.shutdown()

    # Compare
    print("\n=== Comparison ===")
    all_match = True
    for i, (prompt, base, test_out) in enumerate(
        zip(prompts, baseline_outputs, test_outputs)
    ):
        match = base["text"] == test_out["text"]
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"[{status}] Prompt: {repr(prompt)}")
        print(f"  Baseline: {repr(base['text'][:80])}")
        print(f"  Test:     {repr(test_out['text'][:80])}")
        print()

    if all_match:
        print("ALL OUTPUTS MATCH!")
    else:
        print("SOME OUTPUTS DIFFER!")


if __name__ == "__main__":
    main()
