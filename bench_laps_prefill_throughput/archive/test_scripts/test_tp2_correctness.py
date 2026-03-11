"""Correctness test for batch prefill CUDA graph with TP=2.

Tests 4 modes with TP=2 and compares outputs against a TP=2 baseline (no CUDA graph):
  1. Baseline:    TP=2, no CUDA graph
  2. Piecewise:   TP=2, piecewise CUDA graph
  3. BatchPF:     TP=2, piecewise + batch prefill CUDA graph
  4. DualQueue+BatchPF: TP=2, piecewise + batch prefill + LAPS scheduler

Usage:
    python test_tp2_correctness.py [--model Qwen/Qwen2.5-7B]
"""
import argparse
import sys
from sglang import Engine


def run_engine(model_path, tp_size, label, prompts, sampling_params, **engine_kwargs):
    """Launch an engine, generate outputs, and shut down."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  model={model_path}, tp={tp_size}")
    print(f"  kwargs: {engine_kwargs}")
    print(f"{'='*60}")

    engine = Engine(
        model_path=model_path,
        tp_size=tp_size,
        mem_fraction_static=0.7,
        cuda_graph_max_bs=32,
        **engine_kwargs,
    )
    outputs = engine.generate(prompts, sampling_params=sampling_params)
    engine.shutdown()
    return outputs


def compare_outputs(baseline_outputs, test_outputs, baseline_label, test_label, prompts):
    """Compare two sets of outputs and report matches/mismatches."""
    print(f"\n--- {test_label} vs {baseline_label} ---")
    all_match = True
    for i, (prompt, base, test_out) in enumerate(
        zip(prompts, baseline_outputs, test_outputs)
    ):
        match = base["text"] == test_out["text"]
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"  [{status}] Prompt: {repr(prompt[:50])}")
        if not match:
            print(f"    Baseline: {repr(base['text'][:100])}")
            print(f"    Test:     {repr(test_out['text'][:100])}")

    if all_match:
        print(f"  => ALL {len(prompts)} OUTPUTS MATCH!")
    else:
        print(f"  => SOME OUTPUTS DIFFER!")
    return all_match


def main():
    parser = argparse.ArgumentParser(description="TP=2 correctness test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model path (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallelism size")
    args = parser.parse_args()

    model_path = args.model
    tp_size = args.tp

    prompts = [
        "What is 2+2?",
        "Hello, how are you doing today?",
        "Tell me a joke about computers.",
        "The sky is blue because",
        "Once upon a time in a land far far away",
        "The capital of France is",
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about the ocean.",
        "The difference between machine learning and deep learning is",
        "In the year 2050, technology will",
    ]
    sampling_params = {"max_new_tokens": 32, "temperature": 0}

    results = {}

    # Mode 1: Baseline — no CUDA graph
    results["baseline"] = run_engine(
        model_path, tp_size,
        "Mode 1: Baseline (no CUDA graph, TP=2)",
        prompts, sampling_params,
    )

    # Mode 2: Piecewise CUDA graph
    results["piecewise"] = run_engine(
        model_path, tp_size,
        "Mode 2: Piecewise CUDA graph (TP=2)",
        prompts, sampling_params,
        enable_piecewise_cuda_graph=True,
    )

    # Mode 3: Piecewise + Batch Prefill CUDA graph
    results["batch_prefill"] = run_engine(
        model_path, tp_size,
        "Mode 3: Piecewise + Batch Prefill CUDA graph (TP=2)",
        prompts, sampling_params,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4, 8],
        batch_prefill_seq_lengths=[16, 32, 64, 128],
    )

    # Mode 4: Piecewise + Batch Prefill + LAPS Dual-Queue
    results["dualqueue_bp"] = run_engine(
        model_path, tp_size,
        "Mode 4: Piecewise + Batch Prefill + LAPS Dual-Queue (TP=2)",
        prompts, sampling_params,
        enable_piecewise_cuda_graph=True,
        enable_batch_prefill_cuda_graph=True,
        batch_prefill_max_seq_len=256,
        batch_prefill_batch_sizes=[1, 2, 4, 8],
        batch_prefill_seq_lengths=[16, 32, 64, 128],
        enable_laps_scheduler=True,
        laps_length_threshold=256,
    )

    # Compare all modes against baseline
    print("\n" + "=" * 60)
    print("  COMPARISON RESULTS")
    print("=" * 60)

    all_pass = True
    for mode_name, label in [
        ("piecewise", "Piecewise"),
        ("batch_prefill", "BatchPF"),
        ("dualqueue_bp", "DualQueue+BatchPF"),
    ]:
        passed = compare_outputs(
            results["baseline"], results[mode_name],
            "Baseline", label, prompts,
        )
        if not passed:
            all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("  ALL MODES PASS CORRECTNESS CHECK (TP=2)")
    else:
        print("  SOME MODES FAILED CORRECTNESS CHECK")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
