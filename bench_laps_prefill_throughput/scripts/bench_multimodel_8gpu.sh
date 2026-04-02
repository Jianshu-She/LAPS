#!/bin/bash
# Benchmark prefill throughput on multiple Qwen2.5 models with 8 GPUs (TP=4).
# Models: 7B, 14B, 32B, 72B
#
# Per model: 5 settings x 7 concurrency levels = 35 data points.
# Total: 4 models x 35 = 140 data points.
# Prefill server: GPU 0,1,2,3 (TP=4); Decode server: GPU 4,5,6,7 (TP=4).
#
# Usage: bash bench_multimodel_8gpu.sh
set -euo pipefail

DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
BACKEND="${BACKEND:-mooncake}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-$(which python3)}"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="1 2 4 8 16 32 64"

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
    local model=$1
    local label=$2
    local prefill_extra_args=$3
    local results_dir=$4
    local health_timeout=${5:-600}

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Model: ${model}, TP=${TP_SIZE}"
    echo "  Prefill GPUs: ${PREFILL_GPUS}, Decode GPUs: ${DECODE_GPUS}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    CUDA_VISIBLE_DEVICES=$DECODE_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --tp-size $TP_SIZE \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${results_dir}/${label}_decode.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --tp-size $TP_SIZE \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${results_dir}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" $health_timeout
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  $health_timeout

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${results_dir}/${label}_router.log" 2>&1 &

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
    local results_dir=$2

    for cc in $CONCURRENCY_LEVELS; do
        echo ""
        echo "  --- ${label} | cc=${cc} ---"
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" \
            --url "http://${HOST}:${ROUTER_PORT}" \
            --num-prompts $NUM_PROMPTS \
            --max-new-tokens $MAX_NEW_TOKENS \
            --concurrency $cc \
            --output "${results_dir}/${label}_cc${cc}.json" \
            2>&1 | tee "${results_dir}/${label}_cc${cc}.txt"
        sleep 2
    done
}

run_model() {
    local model=$1
    local model_short=$2
    local health_timeout=${3:-600}

    local results_dir="${SCRIPT_DIR}/results_${model_short}_tp4"
    mkdir -p "$results_dir"

    local LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

    echo ""
    echo "############################################################"
    echo "  MODEL: ${model} (${model_short})"
    echo "  Results: ${results_dir}"
    echo "############################################################"

    launch_servers "$model" "vanilla_sglang" "" "$results_dir" "$health_timeout"
    run_concurrency_sweep "vanilla_sglang" "$results_dir"

    launch_servers "$model" "prefill_cuda_graph" "--enable-piecewise-cuda-graph" "$results_dir" "$health_timeout"
    run_concurrency_sweep "prefill_cuda_graph" "$results_dir"

    launch_servers "$model" "batch_prefill_cuda_graph" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph" "$results_dir" "$health_timeout"
    run_concurrency_sweep "batch_prefill_cuda_graph" "$results_dir"

    launch_servers "$model" "prefill_disagg" "$LAPS_ARGS" "$results_dir" "$health_timeout"
    run_concurrency_sweep "prefill_disagg" "$results_dir"

    launch_servers "$model" "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS" "$results_dir" "$health_timeout"
    run_concurrency_sweep "laps" "$results_dir"

    echo ""
    echo "  ${model_short} complete. Results in: ${results_dir}/"
}

trap cleanup EXIT

# ───────────────────────── run all models ─────────────────────────

run_model "Qwen/Qwen2.5-7B"            "7b"   300
run_model "Qwen/Qwen2.5-14B"           "14b"  600
run_model "Qwen/Qwen2.5-32B"           "32b"  600
run_model "Qwen/Qwen2.5-72B-Instruct"  "72b"  900

echo ""
echo "============================================================"
echo "  ALL MODELS COMPLETE"
echo "============================================================"
