#!/bin/bash
# Benchmark prefill throughput on Qwen2.5-14B with 8 GPUs (TP=4) and concurrency sweep.
#
# 3 settings x 7 concurrency levels = 21 data points.
# Prefill server: GPU 0,1,2,3 (TP=4); Decode server: GPU 4,5,6,7 (TP=4).
#
# Usage: bash bench_14b_8gpu.sh
set -euo pipefail

MODEL="Qwen/Qwen2.5-14B"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_14b_8gpu"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/laps/bin/python"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="1 2 4 8 16 32 64"

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

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3
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
        sleep 2
    done
}

trap cleanup EXIT

# ───────────────────────── run 3 settings ─────────────────────────

LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

launch_servers "vanilla_sglang" ""
run_concurrency_sweep "vanilla_sglang"

launch_servers "disaggregation" "$LAPS_ARGS"
run_concurrency_sweep "disaggregation"

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"
run_concurrency_sweep "laps"

echo ""
echo "============================================================"
echo "  All 3 settings complete."
echo "  Results in: ${RESULTS_DIR}/"
echo "============================================================"
