"""Quick test for batch prefill CUDA graph v2 (attention inside graph)."""
from sglang import Engine


def main():
    model_path = "/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"

    print("=== Test 1: Launch Engine with batch prefill CUDA graph ===")
    engine = Engine(
        model_path=model_path,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4],
        batch_prefill_seq_lengths=[16, 32, 64],
        mem_fraction_static=0.7,
        cuda_graph_max_bs=32,
    )
    print("Engine launched successfully!")

    print()
    print("=== Test 2: Single request ===")
    output = engine.generate(
        ["Hello, world!"],
        sampling_params={"max_new_tokens": 16, "temperature": 0},
    )
    print(f"  Output: {repr(output[0]['text'])}")

    print()
    print("=== Test 3: Batch of 4 same-length prompts ===")
    prompts = [
        "What is 2+2?",
        "Hello, how are you?",
        "Tell me a joke.",
        "The sky is blue.",
    ]
    outputs = engine.generate(
        prompts,
        sampling_params={"max_new_tokens": 16, "temperature": 0},
    )
    for p, o in zip(prompts, outputs):
        print(f"  {repr(p):30s} -> {repr(o['text'])}")

    engine.shutdown()
    print()
    print("=== Done ===")


if __name__ == "__main__":
    main()
