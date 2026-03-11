"""Filter LMSYS chat dataset to only prompts with uniform token lengths.

Keeps prompts estimated at 40-80 tokens (using chars/4 heuristic).
Outputs up to 2000 prompts to a new JSONL file.

Usage:
    python create_uniform_dataset.py \
        --input /path/to/lmsys_chat_10k.jsonl \
        --output /path/to/lmsys_chat_uniform.jsonl \
        --min-tokens 40 --max-tokens 80 --max-prompts 2000
"""

import argparse
import json
import random


def estimate_tokens(text: str) -> int:
    """Rough token estimate: chars / 4."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Filter dataset to uniform-length prompts")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--min-tokens", type=int, default=40, help="Min estimated tokens")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max estimated tokens")
    parser.add_argument("--max-prompts", type=int, default=2000, help="Max prompts to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    candidates = []
    total = 0
    with open(args.input) as f:
        for line in f:
            total += 1
            record = json.loads(line)
            text = record["text"]
            est_tokens = estimate_tokens(text)
            if args.min_tokens <= est_tokens <= args.max_tokens:
                candidates.append((est_tokens, line.strip()))

    print(f"Total prompts in input: {total}")
    print(f"Prompts in [{args.min_tokens}, {args.max_tokens}] token range: {len(candidates)}")

    # Shuffle and take up to max_prompts
    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[: args.max_prompts]

    # Report stats
    token_counts = [t for t, _ in selected]
    if token_counts:
        print(f"Selected: {len(selected)} prompts")
        print(f"  Token range: [{min(token_counts)}, {max(token_counts)}]")
        print(f"  Mean tokens: {sum(token_counts)/len(token_counts):.1f}")
    else:
        print("WARNING: No prompts matched the filter criteria!")
        return

    with open(args.output, "w") as f:
        for _, line in selected:
            f.write(line + "\n")

    print(f"Written to: {args.output}")


if __name__ == "__main__":
    main()
