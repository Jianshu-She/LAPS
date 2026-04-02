#!/bin/bash
# Minimal reproduction of req_to_token_pool memory leak with LAPS scheduler.
# Sends a small number of requests, waits for idle, and checks if the leak triggers.
set -euo pipefail

MODEL="Qwen/Qwen2.5-7B"
DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_debug_leak"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/laps/bin/python"

TP_SIZE=1
PREFILL_GPUS="0"
DECODE_GPUS="1"
LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

mkdir -p "$RESULTS_DIR"

cleanup() {
    echo "[cleanup] Killing servers..."
    pkill -f "sglang.launch_server.*--port ${PREFILL_PORT}" 2>/dev/null || true
    pkill -f "sglang.launch_server.*--port ${DECODE_PORT}" 2>/dev/null || true
    pkill -f "sglang_router.launch_router.*--port ${ROUTER_PORT}" 2>/dev/null || true
    sleep 3
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

trap cleanup EXIT
cleanup

echo "=== Launching LAPS servers (prefill_disagg mode) ==="

CUDA_VISIBLE_DEVICES=$DECODE_GPUS $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp-size $TP_SIZE \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend $BACKEND \
    --disaggregation-ib-device $IB_DEVICE \
    --disaggregation-bootstrap-port 9300 \
    --host $HOST --port $DECODE_PORT \
    --mem-fraction-static 0.85 \
    > "${RESULTS_DIR}/decode.log" 2>&1 &

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
    > "${RESULTS_DIR}/prefill.log" 2>&1 &

wait_ready "http://${HOST}:${PREFILL_PORT}/health" 300
wait_ready "http://${HOST}:${DECODE_PORT}/health"  300

$PYTHON -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
    --decode  "http://${HOST}:${DECODE_PORT}" \
    --host $HOST --port $ROUTER_PORT \
    > "${RESULTS_DIR}/router.log" 2>&1 &

wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

echo "=== Sending just 5 requests with cc=1 ==="
$PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
    --dataset "$DATASET" \
    --url "http://${HOST}:${ROUTER_PORT}" \
    --num-prompts 5 \
    --max-new-tokens 1 \
    --concurrency 1 \
    2>&1 | tee "${RESULTS_DIR}/run1.txt"

echo ""
echo "=== Waiting 15s for idle check to trigger ==="
sleep 15

echo "=== Checking prefill log for leak debug info ==="
grep -i "memory leak\|DEBUG:" "${RESULTS_DIR}/prefill.log" | tail -20 || echo "No leak detected!"

echo ""
echo "=== Sending 5 more requests ==="
$PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
    --dataset "$DATASET" \
    --url "http://${HOST}:${ROUTER_PORT}" \
    --num-prompts 5 \
    --max-new-tokens 1 \
    --concurrency 1 \
    2>&1 | tee "${RESULTS_DIR}/run2.txt"

echo ""
echo "=== Waiting 15s again ==="
sleep 15

echo "=== Final check ==="
grep -i "memory leak\|DEBUG:" "${RESULTS_DIR}/prefill.log" | tail -20 || echo "No leak detected!"

echo ""
echo "Done. Full prefill log: ${RESULTS_DIR}/prefill.log"
