#!/bin/bash
# Benchmark prefill instance throughput under 3 CUDA graph settings in PD disaggregation.
#
# Settings:
#   1. no_cuda_graph        — baseline, no CUDA graph at all
#   2. piecewise             — --enable-piecewise-cuda-graph only
#   3. batch_prefill         — --enable-piecewise-cuda-graph + --enable-batch-prefill-cuda-graph
#
# Reports only prefill-side throughput (input tokens/s via TTFT).
#
# Usage: bash bench_prefill_throughput.sh
set -euo pipefail

MODEL="/mnt/weka/home/jianshu.she/models/qwen2.5-0.5b"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

NUM_PROMPTS=2000        # subset of the 10k dataset
REQUEST_RATE=100        # requests/s; set to 0 for infinite (all at once)
MAX_NEW_TOKENS=1        # minimize decode; we only care about prefill
PREFILL_GPU=0
DECODE_GPU=1

mkdir -p "$RESULTS_DIR"

# ───────────────────────── helpers ─────────────────────────

cleanup() {
    echo "[cleanup] Killing servers on ports ${PREFILL_PORT}, ${DECODE_PORT}, ${ROUTER_PORT}..."
    pkill -f "sglang.launch_server.*--port ${PREFILL_PORT}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${DECODE_PORT}" 2>/dev/null || true
    pkill -f "sglang_router.launch_router.*--port ${ROUTER_PORT}" 2>/dev/null || true
    sleep 5
}

wait_ready() {
    local url=$1
    local timeout=${2:-300}
    echo -n "[wait] $url ..."
    for i in $(seq 1 $timeout); do
        if curl -s "$url" > /dev/null 2>&1; then
            echo " ready (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo " TIMEOUT"
    return 1
}

run_one() {
    local label=$1
    local prefill_extra_args=$2

    echo ""
    echo "============================================================"
    echo "  Setting: ${label}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    # ── launch decode (GPU $DECODE_GPU) ──
    CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${RESULTS_DIR}/${label}_decode.log" 2>&1 &

    # ── launch prefill (GPU $PREFILL_GPU) ──
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 300
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  300

    # ── launch router ──
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    # ── warmup ──
    echo "[bench] Warming up (20 requests)..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --request-rate 10 \
        > /dev/null 2>&1 || true
    sleep 3

    # ── benchmark: rate-limited ──
    echo "[bench] Running rate=${REQUEST_RATE} req/s, ${NUM_PROMPTS} prompts..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts $NUM_PROMPTS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --request-rate $REQUEST_RATE \
        --output "${RESULTS_DIR}/${label}_rate${REQUEST_RATE}.json" \
        2>&1 | tee "${RESULTS_DIR}/${label}_rate${REQUEST_RATE}.txt"

    sleep 3

    # ── benchmark: infinite rate (max throughput) ──
    echo "[bench] Running rate=inf, ${NUM_PROMPTS} prompts..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts $NUM_PROMPTS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --request-rate 0 \
        --output "${RESULTS_DIR}/${label}_rateinf.json" \
        2>&1 | tee "${RESULTS_DIR}/${label}_rateinf.txt"

    echo "[done] ${label} complete."
}

trap cleanup EXIT

# ───────────────────────── run 3 settings ─────────────────────────

run_one "no_cuda_graph" ""

run_one "piecewise" "--enable-piecewise-cuda-graph"

run_one "batch_prefill" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph"

# ───────────────────────── summary ─────────────────────────

echo ""
echo "============================================================"
echo "  PREFILL THROUGHPUT SUMMARY"
echo "============================================================"

for label in no_cuda_graph piecewise batch_prefill; do
    echo ""
    echo "--- ${label} ---"
    for rate_tag in "rate${REQUEST_RATE}" "rateinf"; do
        f="${RESULTS_DIR}/${label}_${rate_tag}.txt"
        if [ -f "$f" ]; then
            echo "  [${rate_tag}]"
            grep -E "prefill_throughput|median_ttft|mean_ttft|p99_ttft|total_duration|completed" "$f" || true
        fi
    done
done

echo ""
echo "Full results in: ${RESULTS_DIR}/"
