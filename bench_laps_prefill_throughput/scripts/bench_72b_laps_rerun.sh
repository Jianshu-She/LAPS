#!/bin/bash
# Rerun LAPS setting only for 72B flush-cache benchmark.
# Restarts servers for each cc level to avoid flush_cache memory leak.
set -euo pipefail

MODEL="Qwen/Qwen2.5-72B-Instruct"
DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
BACKEND="${BACKEND:-mooncake}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_72b_tp4_flush"
PYTHON="${PYTHON:-$(which python3)}"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="4 8 16 32 64 128"
LAPS_ARGS="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --enable-laps-scheduler --laps-length-threshold 256"
LABEL="laps"

mkdir -p "$RESULTS_DIR"

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
    echo ""
    echo "============================================================"
    echo "  Launching LAPS servers (fresh restart)"
    echo "  Model: ${MODEL}, TP=${TP_SIZE}"
    echo "  Prefill GPUs: ${PREFILL_GPUS}, Decode GPUs: ${DECODE_GPUS}"
    echo "  Args: ${LAPS_ARGS}"
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
        > "${RESULTS_DIR}/${LABEL}_decode.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $LAPS_ARGS \
        > "${RESULTS_DIR}/${LABEL}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 900
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  900

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${LABEL}_router.log" 2>&1 &

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

trap cleanup EXIT

# Run each cc level with a fresh server restart
for cc in $CONCURRENCY_LEVELS; do
    launch_servers

    echo ""
    echo "  --- ${LABEL} | cc=${cc} ---"
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts $NUM_PROMPTS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency $cc \
        --output "${RESULTS_DIR}/${LABEL}_cc${cc}.json" \
        2>&1 | tee "${RESULTS_DIR}/${LABEL}_cc${cc}.txt"
done

echo ""
echo "LAPS rerun complete. Results in: ${RESULTS_DIR}/"
