"""
Benchmark: Batch Prefill CUDA Graph Correctness & Performance

This script:
1. Launches two sglang servers (with/without batch prefill CUDA graph)
2. Verifies output correctness (greedy outputs must match)
3. Measures latency for concurrent short prefill requests (max_new_tokens=1)

Usage:
    python benchmark_batch_prefill.py --model-path /path/to/model

Requirements:
    - At least 2 GPUs available
    - sglang installed with batch prefill CUDA graph support
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import concurrent.futures
import requests
import numpy as np


# ── Server management ──────────────────────────────────────────────

def wait_for_server(port, timeout=300):
    """Wait until server is healthy."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                # Double-check with a test request
                r2 = requests.get(f"http://localhost:{port}/get_model_info", timeout=2)
                if r2.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch_server(model_path, port, gpu_id, enable_batch_prefill, log_file):
    """Launch an sglang server.

    Baseline: no piecewise cuda graph (no prefill CUDA graph at all)
    Batch prefill: piecewise cuda graph + batch prefill enabled
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--log-level", "info",
    ]
    if enable_batch_prefill:
        cmd.append("--enable-piecewise-cuda-graph")
        cmd.append("--enable-batch-prefill-cuda-graph")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    f = open(log_file, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    return proc, f


def kill_server(proc, log_fh):
    """Kill a server process and its children."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    log_fh.close()


# ── Request helpers ────────────────────────────────────────────────

def generate(port, prompt, max_tokens=1):
    """Send a generate request and return (text, latency_ms)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"http://localhost:{port}/generate",
        json={
            "text": prompt,
            "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0},
        },
        timeout=30,
    )
    latency = (time.perf_counter() - t0) * 1000
    data = r.json()
    return data.get("text", ""), latency


def send_batch_concurrent(port, prompts, max_tokens=1):
    """Send prompts concurrently and return list of (text, latency_ms)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(generate, port, p, max_tokens)
            for p in prompts
        ]
        return [f.result() for f in futures]


def measure_batch_latency(port, prompts, max_tokens=1, n_warmup=3, n_repeat=10):
    """
    Measure the wall-clock time to complete a concurrent batch.
    Returns (median_ms, all_times_ms, outputs).
    """
    # Warmup
    for _ in range(n_warmup):
        send_batch_concurrent(port, prompts, max_tokens)

    times = []
    last_outputs = None
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        results = send_batch_concurrent(port, prompts, max_tokens)
        wall_ms = (time.perf_counter() - t0) * 1000
        times.append(wall_ms)
        last_outputs = [r[0] for r in results]

    return float(np.median(times)), times, last_outputs


# ── Test suites ────────────────────────────────────────────────────

# Various batch sizes and prompt lengths for testing
CORRECTNESS_PROMPTS = [
    # Same-length batch
    ["What is 2+2?", "Tell me a joke", "Hello, how are", "The sky is blue"],
    # Different-length batch
    ["Hi", "What is the capital of France?", "A", "The quick brown fox jumps over"],
    # Larger batch
    ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight"],
    # Single long-ish prompt
    ["Explain the theory of relativity in simple terms for a five year old child"],
]

BENCH_PROMPTS_4 = [
    "What is 2+2?",
    "Hello world!",
    "The capital of France is",
    "1, 2, 3, 4, 5,",
]

BENCH_PROMPTS_8 = [
    "What is 2+2?",
    "Hello world!",
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "Once upon a time",
    "The meaning of life",
    "How to cook pasta",
    "Tell me a story",
]

BENCH_PROMPTS_1 = ["What is 2+2?"]


def test_correctness(port_baseline, port_batch):
    """Compare outputs from both servers for correctness."""
    print("\n" + "=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)

    all_pass = True
    for i, prompts in enumerate(CORRECTNESS_PROMPTS):
        print(f"\n  Test group {i+1}: {len(prompts)} prompts")

        results_base = send_batch_concurrent(port_baseline, prompts)
        results_batch = send_batch_concurrent(port_batch, prompts)

        for j, prompt in enumerate(prompts):
            text_base = results_base[j][0]
            text_batch = results_batch[j][0]
            match = text_base == text_batch
            status = "PASS" if match else "FAIL"
            if not match:
                all_pass = False

            print(f"    [{status}] {prompt!r:50s}")
            if not match:
                print(f"           baseline: {text_base!r}")
                print(f"           batch:    {text_batch!r}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


def test_performance(port_baseline, port_batch, n_warmup=5, n_repeat=20):
    """Measure latency comparison."""
    print("\n" + "=" * 70)
    print("PERFORMANCE TEST (max_new_tokens=1, i.e. prefill-dominated)")
    print("=" * 70)

    configs = [
        ("bs=1 (single)", BENCH_PROMPTS_1),
        ("bs=4 (batch)", BENCH_PROMPTS_4),
        ("bs=8 (batch)", BENCH_PROMPTS_8),
    ]

    print(f"\n  {'Config':<20s} {'Baseline (ms)':>15s} {'BatchPrefill (ms)':>18s} {'Speedup':>10s}")
    print("  " + "-" * 65)

    for label, prompts in configs:
        med_base, times_base, _ = measure_batch_latency(
            port_baseline, prompts, max_tokens=1,
            n_warmup=n_warmup, n_repeat=n_repeat,
        )
        med_batch, times_batch, _ = measure_batch_latency(
            port_batch, prompts, max_tokens=1,
            n_warmup=n_warmup, n_repeat=n_repeat,
        )
        speedup = med_base / med_batch if med_batch > 0 else float("inf")
        print(f"  {label:<20s} {med_base:>12.2f} ms {med_batch:>15.2f} ms {speedup:>9.2f}x")
        print(f"  {'':20s} p10={np.percentile(times_base, 10):.1f} p90={np.percentile(times_base, 90):.1f}"
              f"    p10={np.percentile(times_batch, 10):.1f} p90={np.percentile(times_batch, 90):.1f}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark batch prefill CUDA graph")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--gpu-baseline", type=int, default=0, help="GPU for baseline server")
    parser.add_argument("--gpu-batch", type=int, default=1, help="GPU for batch-prefill server")
    parser.add_argument("--port-baseline", type=int, default=30020, help="Port for baseline")
    parser.add_argument("--port-batch", type=int, default=30021, help="Port for batch-prefill")
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--n-repeat", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--skip-correctness", action="store_true", help="Skip correctness test")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance test")
    parser.add_argument("--use-existing", action="store_true",
                        help="Use already-running servers (don't launch/kill)")
    args = parser.parse_args()

    log_base = "/tmp/sglang_bench_baseline.log"
    log_batch = "/tmp/sglang_bench_batch_prefill.log"

    print("=" * 70)
    print("Batch Prefill CUDA Graph Benchmark")
    print("=" * 70)
    print(f"  Model:          {args.model_path}")
    print(f"  Baseline:       GPU {args.gpu_baseline}, port {args.port_baseline}")
    print(f"  Batch prefill:  GPU {args.gpu_batch}, port {args.port_batch}")
    print()

    proc_base = proc_batch = fh_base = fh_batch = None

    if not args.use_existing:
        print("Launching baseline server (no batch prefill)...")
        proc_base, fh_base = launch_server(
            args.model_path, args.port_baseline, args.gpu_baseline,
            enable_batch_prefill=False, log_file=log_base,
        )

        print("Launching batch-prefill server...")
        proc_batch, fh_batch = launch_server(
            args.model_path, args.port_batch, args.gpu_batch,
            enable_batch_prefill=True, log_file=log_batch,
        )

    try:
        print("Waiting for baseline server...", end=" ", flush=True)
        if not wait_for_server(args.port_baseline):
            print("TIMEOUT! Check", log_base)
            return 1
        print("ready.")

        print("Waiting for batch-prefill server...", end=" ", flush=True)
        if not wait_for_server(args.port_batch):
            print("TIMEOUT! Check", log_batch)
            return 1
        print("ready.")

        # Run tests
        if not args.skip_correctness:
            test_correctness(args.port_baseline, args.port_batch)

        if not args.skip_performance:
            test_performance(
                args.port_baseline, args.port_batch,
                n_warmup=args.n_warmup, n_repeat=args.n_repeat,
            )

        # Show batch prefill log entries
        print("\n" + "=" * 70)
        print("BATCH PREFILL LOG ENTRIES (from server log)")
        print("=" * 70)
        try:
            with open(log_batch) as f:
                for line in f:
                    if "[BatchPrefill]" in line:
                        print(" ", line.rstrip())
        except Exception:
            print("  (No log file found — use --use-existing with separate log)")

        print("\nDone! Server logs at:")
        print(f"  Baseline:      {log_base}")
        print(f"  Batch prefill: {log_batch}")
        return 0

    finally:
        if not args.use_existing:
            print("\nShutting down servers...")
            if proc_base:
                kill_server(proc_base, fh_base)
            if proc_batch:
                kill_server(proc_batch, fh_batch)


if __name__ == "__main__":
    sys.exit(main())
