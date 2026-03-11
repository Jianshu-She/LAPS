"""
Analyze padding waste for batch prefill CUDA graph under single-queue vs dual-queue (LAPS).

Simulates how requests get batched and padded to CUDA graph dimensions,
measuring the ratio of real tokens to padded tokens.

Usage:
    python analyze_padding_waste.py --dataset ../data/lmsys_chat_10k.jsonl
    python analyze_padding_waste.py --dataset ../data/lmsys_chat_10k.jsonl --num-prompts 500 --seed 42
"""

import argparse
import json
import random
import math
from collections import defaultdict


def load_dataset(path, num_prompts=None, seed=42):
    """Load prompts and return their token lengths (approximated by char/4 or actual if available)."""
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Use the prompt text length / 4 as a rough token estimate,
            # or use actual token count if available
            if "input_ids" in d:
                length = len(d["input_ids"])
            elif "prompt" in d:
                # Rough approximation: 1 token ~= 4 chars
                length = max(1, len(d["prompt"]) // 4)
            elif "text" in d:
                length = max(1, len(d["text"]) // 4)
            elif "conversations" in d:
                # LMSYS format: take the first human turn
                text = d["conversations"][0].get("value", "")
                length = max(1, len(text) // 4)
            else:
                continue
            prompts.append(length)

    random.seed(seed)
    random.shuffle(prompts)
    if num_prompts:
        prompts = prompts[:num_prompts]
    return prompts


def find_graph_dims(bs, max_seq_len, captured_bs_list, captured_seq_len_list):
    """Find the smallest captured (bs, seq_len) that fits the batch."""
    # Find smallest bs >= actual bs
    target_bs = None
    for b in captured_bs_list:
        if b >= bs:
            target_bs = b
            break
    if target_bs is None:
        target_bs = captured_bs_list[-1]

    # Find smallest seq_len >= max_seq_len
    target_seq_len = None
    for s in captured_seq_len_list:
        if s >= max_seq_len:
            target_seq_len = s
            break
    if target_seq_len is None:
        target_seq_len = captured_seq_len_list[-1]

    return target_bs, target_seq_len


def simulate_batching(lengths, max_batch_size, captured_bs, captured_seq_lens, max_captured_seq_len):
    """Simulate FIFO batching and compute padding waste.

    Returns list of (real_tokens, padded_tokens, batch_size, target_bs, max_len, target_seq_len)
    """
    batches = []
    queue = list(lengths)  # FIFO

    while queue:
        # Take up to max_batch_size from front of queue
        # Only take requests that fit within max captured seq len
        batch_lens = []
        remaining = []
        for l in queue:
            if len(batch_lens) < max_batch_size and l <= max_captured_seq_len:
                batch_lens.append(l)
            elif l > max_captured_seq_len:
                # Too long for CUDA graph, would fall back to non-graph path
                # Still count it but separately
                batch_lens.append(l)  # Include it, it'll exceed graph dims
            else:
                remaining.append(l)

        if not batch_lens:
            break

        # In reality, the scheduler fills greedily up to max_batch_size
        batch_lens = batch_lens[:max_batch_size]
        queue = queue[len(batch_lens):]
        queue = queue  # remaining requests

        real_tokens = sum(batch_lens)
        max_len = max(batch_lens)
        bs = len(batch_lens)

        target_bs, target_seq_len = find_graph_dims(bs, max_len, captured_bs, captured_seq_lens)
        padded_tokens = target_bs * target_seq_len

        batches.append((real_tokens, padded_tokens, bs, target_bs, max_len, target_seq_len))

    return batches


def simulate_dual_queue(lengths, max_batch_size, captured_bs, captured_seq_lens,
                        max_captured_seq_len, threshold=256, max_consecutive_short=4):
    """Simulate LAPS dual-queue batching."""
    short_queue = [l for l in lengths if l <= threshold]
    long_queue = [l for l in lengths if l > threshold]
    batches = []
    consecutive_short = 0

    while short_queue or long_queue:
        force_long = consecutive_short >= max_consecutive_short and len(long_queue) > 0

        batch_lens = None
        source = None

        if force_long:
            batch_lens = long_queue[:max_batch_size]
            long_queue = long_queue[len(batch_lens):]
            consecutive_short = 0
            source = "long"
        elif short_queue:
            batch_lens = short_queue[:max_batch_size]
            short_queue = short_queue[len(batch_lens):]
            consecutive_short += 1
            source = "short"
        elif long_queue:
            batch_lens = long_queue[:max_batch_size]
            long_queue = long_queue[len(batch_lens):]
            consecutive_short = 0
            source = "long"
        else:
            break

        if not batch_lens:
            break

        real_tokens = sum(batch_lens)
        max_len = max(batch_lens)
        bs = len(batch_lens)

        target_bs, target_seq_len = find_graph_dims(bs, max_len, captured_bs, captured_seq_lens)
        padded_tokens = target_bs * target_seq_len

        batches.append((real_tokens, padded_tokens, bs, target_bs, max_len, target_seq_len))

    return batches


def print_stats(label, batches):
    """Print padding waste statistics."""
    total_real = sum(b[0] for b in batches)
    total_padded = sum(b[1] for b in batches)
    waste = 1 - total_real / total_padded if total_padded > 0 else 0
    num_batches = len(batches)

    per_batch_waste = [(1 - b[0] / b[1]) * 100 for b in batches]
    avg_waste = sum(per_batch_waste) / len(per_batch_waste) if per_batch_waste else 0
    p50_waste = sorted(per_batch_waste)[len(per_batch_waste) // 2] if per_batch_waste else 0
    p90_waste = sorted(per_batch_waste)[int(len(per_batch_waste) * 0.9)] if per_batch_waste else 0

    print(f"\n  {label}")
    print(f"  {'─' * 60}")
    print(f"  Total batches:           {num_batches}")
    print(f"  Total real tokens:       {total_real:,}")
    print(f"  Total padded tokens:     {total_padded:,}")
    print(f"  Overall padding waste:   {waste * 100:.1f}%")
    print(f"  Per-batch waste (mean):  {avg_waste:.1f}%")
    print(f"  Per-batch waste (p50):   {p50_waste:.1f}%")
    print(f"  Per-batch waste (p90):   {p90_waste:.1f}%")

    # Show batch size distribution
    bs_counts = defaultdict(int)
    for b in batches:
        bs_counts[f"bs={b[2]}→{b[3]}, seq={b[4]}→{b[5]}"] += 1

    print(f"\n  Top 10 batch configurations (actual→padded):")
    for k, v in sorted(bs_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k}: {v} batches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batch-size", type=int, default=8,
                        help="Max requests per batch (default: 8)")
    parser.add_argument("--threshold", type=int, default=256,
                        help="LAPS short/long threshold (default: 256)")
    parser.add_argument("--max-consecutive-short", type=int, default=4)
    args = parser.parse_args()

    # Typical batch prefill CUDA graph dimensions
    captured_bs = [1, 2, 4, 8]
    captured_seq_lens = [16, 32, 64, 128, 256, 512]
    max_captured_seq_len = max(captured_seq_lens)

    lengths = load_dataset(args.dataset, args.num_prompts, args.seed)

    print("=" * 70)
    print(f"  PADDING WASTE ANALYSIS")
    print(f"  Dataset: {args.dataset}")
    print(f"  Prompts: {len(lengths)}, Seed: {args.seed}")
    print(f"  Max batch size: {args.max_batch_size}")
    print(f"  Captured graphs: bs={captured_bs}, seq_lens={captured_seq_lens}")
    print(f"  LAPS threshold: {args.threshold}")
    print("=" * 70)

    # Dataset stats
    short = [l for l in lengths if l <= args.threshold]
    long = [l for l in lengths if l > args.threshold]
    print(f"\n  Dataset distribution:")
    print(f"    Short (≤{args.threshold}): {len(short)} ({len(short)/len(lengths)*100:.0f}%)")
    print(f"    Long  (>{args.threshold}): {len(long)} ({len(long)/len(lengths)*100:.0f}%)")
    if short:
        print(f"    Short length: mean={sum(short)/len(short):.0f}, "
              f"min={min(short)}, max={max(short)}")
    if long:
        print(f"    Long length:  mean={sum(long)/len(long):.0f}, "
              f"min={min(long)}, max={max(long)}")

    # Simulate single queue (FIFO)
    single_batches = simulate_batching(
        lengths, args.max_batch_size, captured_bs, captured_seq_lens, max_captured_seq_len
    )
    print_stats("SINGLE QUEUE (no LAPS)", single_batches)

    # Simulate dual queue with different N values
    for n in [1, 2, 4, 8, 999]:
        dual_batches = simulate_dual_queue(
            lengths, args.max_batch_size, captured_bs, captured_seq_lens,
            max_captured_seq_len, args.threshold, n
        )
        label = f"DUAL QUEUE (LAPS N={n})" if n < 999 else "DUAL QUEUE (LAPS N=∞, strict short-first)"
        print_stats(label, dual_batches)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
