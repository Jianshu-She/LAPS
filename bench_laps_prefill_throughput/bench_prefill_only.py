"""Benchmark prefill-only throughput in PD disaggregation.

Sends requests from a JSONL dataset ({"text": "..."} per line) to the
router's /generate endpoint with max_new_tokens=1, measures TTFT (which
equals prefill latency in PD mode), and reports prefill throughput.

Supports two modes:
  --request-rate R   Poisson-like arrival at R req/s (0 = send all at once)
  --concurrency C    Fixed number of concurrent in-flight requests (semaphore)

Usage:
    python bench_prefill_only.py \
        --dataset /path/to/prompts.jsonl \
        --url http://127.0.0.1:30302 \
        --num-prompts 2000 \
        --max-new-tokens 1 \
        --concurrency 8 \
        --output results.json
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


async def send_request(session, url, text, max_new_tokens, timeout):
    """Send one request and return (input_chars, ttft_s, success)."""
    payload = {
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{url}/generate", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as resp:
            await resp.json()
            ttft = time.monotonic() - t0
            return len(text), ttft, resp.status == 200
    except Exception as e:
        ttft = time.monotonic() - t0
        return len(text), ttft, False


async def run_with_concurrency(session, url, prompts, max_new_tokens, timeout, concurrency):
    """Run requests with a fixed concurrency limit using a semaphore."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def limited_request(text):
        async with sem:
            return await send_request(session, url, text, max_new_tokens, timeout)

    tasks = [asyncio.create_task(limited_request(p)) for p in prompts]
    results = await asyncio.gather(*tasks)
    return results


async def run_with_rate(session, url, prompts, max_new_tokens, timeout, request_rate):
    """Run requests with a target arrival rate."""
    tasks = []
    wall_start = time.monotonic()

    for i, text in enumerate(prompts):
        if request_rate > 0 and i > 0:
            expected_time = i / request_rate
            elapsed = time.monotonic() - wall_start
            if elapsed < expected_time:
                await asyncio.sleep(expected_time - elapsed)

        task = asyncio.create_task(
            send_request(session, url, text, max_new_tokens, timeout)
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results


async def run_benchmark(args):
    # Load dataset
    prompts = []
    with open(args.dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            prompts.append(d["text"])
            if len(prompts) >= args.num_prompts:
                break

    if len(prompts) < args.num_prompts:
        print(
            f"[warn] Dataset has {len(prompts)} lines, requested {args.num_prompts}. "
            f"Using all {len(prompts)}."
        )

    num = len(prompts)
    mode_str = (
        f"concurrency={args.concurrency}"
        if args.concurrency > 0
        else f"rate={args.request_rate if args.request_rate > 0 else 'inf'} req/s"
    )
    print(f"Benchmark: {num} prompts, max_new_tokens={args.max_new_tokens}, {mode_str}")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.monotonic()

        if args.concurrency > 0:
            results = await run_with_concurrency(
                session, args.url, prompts, args.max_new_tokens,
                args.timeout, args.concurrency,
            )
        else:
            results = await run_with_rate(
                session, args.url, prompts, args.max_new_tokens,
                args.timeout, args.request_rate,
            )

        wall_end = time.monotonic()

    wall_duration = wall_end - wall_start

    # Compute metrics
    succeeded = [(chars, ttft) for chars, ttft, ok in results if ok]
    failed = sum(1 for _, _, ok in results if not ok)

    if not succeeded:
        print("[FAIL] All requests failed!")
        return

    ttfts = [ttft for _, ttft in succeeded]
    ttfts.sort()
    input_chars = [c for c, _ in succeeded]
    total_input_tokens_ok = sum(c // 4 for c in input_chars)

    prefill_throughput = total_input_tokens_ok / wall_duration  # tokens/s

    # Latency stats
    n = len(ttfts)
    median_ttft = ttfts[n // 2]
    mean_ttft = sum(ttfts) / n
    p50_ttft = ttfts[int(n * 0.50)]
    p90_ttft = ttfts[int(n * 0.90)]
    p99_ttft = ttfts[int(n * 0.99)]

    print(f"\n{'='*60}")
    print(f"  PREFILL THROUGHPUT RESULTS")
    print(f"{'='*60}")
    print(f"  completed:            {n} / {num}")
    print(f"  failed:               {failed}")
    print(f"  total_duration:       {wall_duration:.2f} s")
    print(f"  total_input_tokens:   {total_input_tokens_ok}")
    print(f"  prefill_throughput:   {prefill_throughput:.1f} tokens/s")
    print(f"  request_throughput:   {n / wall_duration:.1f} req/s")
    print(f"  median_ttft:          {median_ttft*1000:.2f} ms")
    print(f"  mean_ttft:            {mean_ttft*1000:.2f} ms")
    print(f"  p50_ttft:             {p50_ttft*1000:.2f} ms")
    print(f"  p90_ttft:             {p90_ttft*1000:.2f} ms")
    print(f"  p99_ttft:             {p99_ttft*1000:.2f} ms")
    print(f"{'='*60}\n")

    # Save to file
    if args.output:
        out = {
            "completed": n,
            "failed": failed,
            "num_prompts": num,
            "total_duration_s": round(wall_duration, 2),
            "total_input_tokens": total_input_tokens_ok,
            "prefill_throughput_tok_s": round(prefill_throughput, 1),
            "request_throughput_req_s": round(n / wall_duration, 1),
            "median_ttft_ms": round(median_ttft * 1000, 2),
            "mean_ttft_ms": round(mean_ttft * 1000, 2),
            "p50_ttft_ms": round(p50_ttft * 1000, 2),
            "p90_ttft_ms": round(p90_ttft * 1000, 2),
            "p99_ttft_ms": round(p99_ttft * 1000, 2),
            "max_new_tokens": args.max_new_tokens,
            "concurrency": args.concurrency if args.concurrency > 0 else None,
            "request_rate": (args.request_rate if args.request_rate > 0 else "inf") if args.concurrency == 0 else None,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefill throughput")
    parser.add_argument("--dataset", required=True, help="JSONL file with {\"text\": ...} per line")
    parser.add_argument("--url", required=True, help="Router base URL")
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=0,
                        help="Max concurrent in-flight requests (0 = use --request-rate instead)")
    parser.add_argument("--request-rate", type=float, default=0,
                        help="Requests/sec. 0 = send all at once. Ignored if --concurrency > 0.")
    parser.add_argument("--timeout", type=float, default=300, help="Per-request timeout (s)")
    parser.add_argument("--output", type=str, default="", help="Output JSON file path")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
