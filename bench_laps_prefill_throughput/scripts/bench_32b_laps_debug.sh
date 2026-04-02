#!/bin/bash
# Debug: reproduce 32B LAPS crash at cc=8.
# Runs LAPS the same way as bench_32b_concurrency.sh (no fresh restarts).
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_32b_laps_debug"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/laps/bin/python"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="4 8 16 32 64 128"
LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

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

trap cleanup EXIT

label="laps"
prefill_extra_args="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"

echo "============================================================"
echo "  DEBUG: Reproducing LAPS crash on 32B TP=4"
echo "  Model: ${MODEL}, TP=${TP_SIZE}"
echo "  Prefill GPUs: ${PREFILL_GPUS}, Decode GPUs: ${DECODE_GPUS}"
echo "  Prefill args: ${prefill_extra_args}"
echo "============================================================"

cleanup

# Launch decode server
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

# Launch prefill server with LAPS args
echo "[launch] Prefill server on GPUs ${PREFILL_GPUS} with: ${prefill_extra_args}..."
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

wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
wait_ready "http://${HOST}:${DECODE_PORT}/health"  600

# Launch router
echo "[launch] Router..."
$PYTHON -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
    --decode  "http://${HOST}:${DECODE_PORT}" \
    --host $HOST --port $ROUTER_PORT \
    > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

# Warmup
echo "[warmup] 20 requests..."
$PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
    --dataset "$DATASET" \
    --url "http://${HOST}:${ROUTER_PORT}" \
    --num-prompts 20 \
    --max-new-tokens $MAX_NEW_TOKENS \
    --concurrency 4 \
    > /dev/null 2>&1 || true
sleep 3

# Run all cc levels WITHOUT restarting servers (same as original)
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

    # Check if servers are still alive
    if ! curl -s "http://${HOST}:${DECODE_PORT}/health" > /dev/null 2>&1; then
        echo "[ERROR] Decode server is DOWN after cc=${cc}!"
        echo "[ERROR] Saving logs for debugging..."
        echo "Decode server died after cc=${cc}" > "${RESULTS_DIR}/crash_info.txt"
        # Save tail of logs
        tail -200 "${RESULTS_DIR}/${label}_decode.log" > "${RESULTS_DIR}/decode_crash_tail.log" 2>/dev/null || true
        tail -200 "${RESULTS_DIR}/${label}_prefill.log" > "${RESULTS_DIR}/prefill_crash_tail.log" 2>/dev/null || true
        break
    fi
    if ! curl -s "http://${HOST}:${PREFILL_PORT}/health" > /dev/null 2>&1; then
        echo "[ERROR] Prefill server is DOWN after cc=${cc}!"
        echo "Prefill server died after cc=${cc}" > "${RESULTS_DIR}/crash_info.txt"
        tail -200 "${RESULTS_DIR}/${label}_decode.log" > "${RESULTS_DIR}/decode_crash_tail.log" 2>/dev/null || true
        tail -200 "${RESULTS_DIR}/${label}_prefill.log" > "${RESULTS_DIR}/prefill_crash_tail.log" 2>/dev/null || true
        break
    fi

    sleep 2
done

echo ""
echo "Debug run complete. Logs in: ${RESULTS_DIR}/"
