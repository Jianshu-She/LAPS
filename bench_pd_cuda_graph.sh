#!/bin/bash
# Benchmark: PD disaggregation with vs without batch prefill CUDA graph
# Usage: bash bench_pd_cuda_graph.sh
set -euo pipefail

MODEL="/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
PREFILL_PORT=30100
DECODE_PORT=30200
ROUTER_PORT=30000
HOST="127.0.0.1"
IB_DEVICE="mlx5_0,mlx5_1"
BACKEND="mooncake"
RESULTS_DIR="/mnt/weka/home/jianshu.she/graph/sglang/bench_results"
mkdir -p "$RESULTS_DIR"

# Activate conda env
eval "$(conda shell.bash hook)"
conda activate /mnt/weka/home/jianshu.she/miniconda3/envs/graph

cleanup() {
    echo "[cleanup] Killing all servers..."
    pkill -f "sglang.launch_server.*--port ${PREFILL_PORT}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${DECODE_PORT}" 2>/dev/null || true
    pkill -f "sglang_router.launch_router.*--port ${ROUTER_PORT}" 2>/dev/null || true
    sleep 5
}

wait_ready() {
    local url=$1
    local timeout=${2:-300}
    echo "[wait] Waiting for $url (timeout=${timeout}s)..."
    for i in $(seq 1 $timeout); do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "[wait] $url is ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[wait] TIMEOUT waiting for $url"
    return 1
}

run_benchmark() {
    local label=$1
    local prefill_extra_args=$2
    local output_file="${RESULTS_DIR}/${label}.json"

    echo ""
    echo "========================================"
    echo "  Benchmark: ${label}"
    echo "========================================"
    echo ""

    cleanup

    # Launch decode server (GPU 1)
    echo "[launch] Starting decode server on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --port $DECODE_PORT \
        --host $HOST \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --mem-fraction-static 0.85 \
        > "${RESULTS_DIR}/${label}_decode.log" 2>&1 &

    # Launch prefill server (GPU 0)
    echo "[launch] Starting prefill server on GPU 0 with args: ${prefill_extra_args}"
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode prefill \
        --port $PREFILL_PORT \
        --host $HOST \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    # Wait for servers
    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 300
    wait_ready "http://${HOST}:${DECODE_PORT}/health" 300

    # Launch router
    echo "[launch] Starting router..."
    python -m sglang_router.launch_router \
        --pd-disaggregation \
        --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" \
        --decode "http://${HOST}:${DECODE_PORT}" \
        --host $HOST \
        --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    echo "[bench] Warming up..."
    # Warmup with a few requests
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://${HOST}:${ROUTER_PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts 20 \
        --random-input 128 \
        --random-output 16 \
        --request-rate 10 \
        --disable-tqdm \
        > /dev/null 2>&1 || true

    sleep 3

    # Benchmark 1: Short prompts, high concurrency (tests CUDA graph batch prefill benefit)
    echo "[bench] Running short-prompt benchmark (input=64, output=16, 200 prompts, rate=inf)..."
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://${HOST}:${ROUTER_PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts 200 \
        --random-input 64 \
        --random-output 16 \
        --request-rate "inf" \
        --output-file "${RESULTS_DIR}/${label}_short.json" \
        2>&1 | tee "${RESULTS_DIR}/${label}_short.txt"

    sleep 3

    # Benchmark 2: Medium prompts
    echo "[bench] Running medium-prompt benchmark (input=256, output=32, 100 prompts, rate=inf)..."
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://${HOST}:${ROUTER_PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts 100 \
        --random-input 256 \
        --random-output 32 \
        --request-rate "inf" \
        --output-file "${RESULTS_DIR}/${label}_medium.json" \
        2>&1 | tee "${RESULTS_DIR}/${label}_medium.txt"

    sleep 3

    # Benchmark 3: Mixed short prompts with moderate rate
    echo "[bench] Running short-prompt moderate-rate benchmark (input=64, output=16, 200 prompts, rate=50)..."
    python -m sglang.bench_serving \
        --backend sglang \
        --base-url "http://${HOST}:${ROUTER_PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --num-prompts 200 \
        --random-input 64 \
        --random-output 16 \
        --request-rate 50 \
        --output-file "${RESULTS_DIR}/${label}_short_r50.json" \
        2>&1 | tee "${RESULTS_DIR}/${label}_short_r50.txt"

    echo "[done] Benchmark ${label} complete."
}

trap cleanup EXIT

# ── Run 1: Baseline (no batch prefill CUDA graph) ──
run_benchmark "baseline" "--enable-piecewise-cuda-graph"

# ── Run 2: With batch prefill CUDA graph ──
run_benchmark "batch_prefill_cudagraph" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph"

echo ""
echo "========================================"
echo "  All benchmarks complete!"
echo "  Results in: ${RESULTS_DIR}/"
echo "========================================"

# Print summary
echo ""
echo "=== SHORT PROMPT (input=64, output=16, rate=inf) ==="
echo "--- Baseline ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/baseline_short.txt" 2>/dev/null || echo "(no results)"
echo "--- Batch Prefill CUDA Graph ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/batch_prefill_cudagraph_short.txt" 2>/dev/null || echo "(no results)"

echo ""
echo "=== MEDIUM PROMPT (input=256, output=32, rate=inf) ==="
echo "--- Baseline ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/baseline_medium.txt" 2>/dev/null || echo "(no results)"
echo "--- Batch Prefill CUDA Graph ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/batch_prefill_cudagraph_medium.txt" 2>/dev/null || echo "(no results)"

echo ""
echo "=== SHORT PROMPT MODERATE RATE (input=64, output=16, rate=50) ==="
echo "--- Baseline ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/baseline_short_r50.txt" 2>/dev/null || echo "(no results)"
echo "--- Batch Prefill CUDA Graph ---"
grep -E "Throughput|Median TTFT|Mean TTFT|P99 TTFT|Median ITL" "${RESULTS_DIR}/batch_prefill_cudagraph_short_r50.txt" 2>/dev/null || echo "(no results)"
