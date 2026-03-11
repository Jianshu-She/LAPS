"""Sort LMSYS chat dataset by estimated token count (ascending).

Groups similar-length prompts together to improve batch efficiency
and reduce padding waste for CUDA graph batch prefill.

Usage:
    python sort_dataset.py \
        --input /path/to/human_prompts_stream_10000.jsonl \
        --output /path/to/human_prompts_stream_10000_sorted.jsonl
"""

import argparse
import json


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Sort dataset by token count")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    records = []
    with open(args.input) as f:
        for line in f:
            record = json.loads(line)
            est_tokens = estimate_tokens(record["text"])
            records.append((est_tokens, line.strip()))

    # Sort ascending by estimated token count
    records.sort(key=lambda x: x[0])

    token_counts = [t for t, _ in records]
    print(f"Total prompts: {len(records)}")
    print(f"Token range: [{min(token_counts)}, {max(token_counts)}]")
    print(f"Mean: {sum(token_counts)/len(token_counts):.1f}")
    print(f"Median: {token_counts[len(token_counts)//2]}")

    with open(args.output, "w") as f:
        for _, line in records:
            f.write(line + "\n")

    print(f"Sorted dataset written to: {args.output}")


if __name__ == "__main__":
    main()
