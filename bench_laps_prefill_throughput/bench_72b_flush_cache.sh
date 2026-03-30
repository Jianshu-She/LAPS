#!/bin/bash
# Benchmark prefill throughput on Qwen2.5-72B-Instruct with TP=4 and concurrency sweep.
# After each concurrency level, flush the KV cache on both prefill and decode servers.
#
# 4 settings x 6 concurrency levels = 24 data points (LAPS excluded due to flush_cache bug).
# Prefill server: GPU 0,1,2,3 (TP=4); Decode server: GPU 4,5,6,7 (TP=4).
#
# Usage: bash bench_72b_flush_cache.sh
set -euo pipefail

MODEL="Qwen/Qwen2.5-72B-Instruct"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
BACKEND="${BACKEND:-mooncake}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_72b_tp4_flush"
PYTHON="${PYTHON:-$(which python3)}"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="4 8 16 32 64 128"

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
    local timeout=${2:-600}
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

flush_kv_cache() {
    echo -n "[flush] Flushing KV cache on prefill and decode servers..."
    # Wait a moment for in-flight requests to drain
    sleep 3
    local prefill_status
    local decode_status
    prefill_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://${HOST}:${PREFILL_PORT}/flush_cache")
    decode_status=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://${HOST}:${DECODE_PORT}/flush_cache")
    echo " prefill=${prefill_status}, decode=${decode_status}"
    if [ "$prefill_status" != "200" ] || [ "$decode_status" != "200" ]; then
        echo "[flush] WARNING: Cache flush may have failed. Retrying in 5s..."
        sleep 5
        curl -s -X POST "http://${HOST}:${PREFILL_PORT}/flush_cache" > /dev/null 2>&1 || true
        curl -s -X POST "http://${HOST}:${DECODE_PORT}/flush_cache" > /dev/null 2>&1 || true
        echo "[flush] Retry complete."
    fi
    # Give GPU memory time to settle
    sleep 2
}

launch_servers() {
    local label=$1
    local prefill_extra_args=$2

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Model: ${MODEL}, TP=${TP_SIZE}"
    echo "  Prefill GPUs: ${PREFILL_GPUS}, Decode GPUs: ${DECODE_GPUS}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    # ── launch decode (TP=4) ──
    echo "[launch] Decode server on GPUs ${DECODE_GPUS}..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${RESULTS_DIR}/${label}_decode.log" 2>&1 &

    # ── launch prefill (TP=4) ──
    echo "[launch] Prefill server on GPUs ${PREFILL_GPUS} with: ${prefill_extra_args:-<none>}..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 900
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  900

    # ── launch router ──
    echo "[launch] Router..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    # ── warmup ──
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3

    # Flush cache after warmup so each concurrency level starts clean
    flush_kv_cache
}

run_concurrency_sweep() {
    local label=$1

    for cc in $CONCURRENCY_LEVELS; do
        echo ""
        echo "  --- ${label} | cc=${cc} ---"
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" \
            --url "http://${HOST}:${ROUTER_PORT}" \
            --num-prompts $NUM_PROMPTS \
            --max-new-tokens $MAX_NEW_TOKENS \
            --concurrency $cc \
            --output "${RESULTS_DIR}/${label}_cc${cc}.json" \
            2>&1 | tee "${RESULTS_DIR}/${label}_cc${cc}.txt"

        # Flush KV cache after each concurrency level
        flush_kv_cache
    done
}

trap cleanup EXIT

# ───────────────────────── run 4 settings (no LAPS) ─────────────────────────

LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

launch_servers "vanilla_sglang" ""
run_concurrency_sweep "vanilla_sglang"

launch_servers "prefill_cuda_graph" "--enable-piecewise-cuda-graph"
run_concurrency_sweep "prefill_cuda_graph"

launch_servers "batch_prefill_cuda_graph" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph"
run_concurrency_sweep "batch_prefill_cuda_graph"

launch_servers "prefill_disagg" "$LAPS_ARGS"
run_concurrency_sweep "prefill_disagg"

# NOTE: LAPS setting is excluded here due to flush_cache memory leak bug.
# Run bench_72b_laps_rerun.sh separately for LAPS (uses fresh server restarts).

echo ""
echo "============================================================"
echo "  First 4 settings complete. Run bench_72b_laps_rerun.sh for LAPS."
echo "============================================================"
echo ""
echo "Full results in: ${RESULTS_DIR}/"
