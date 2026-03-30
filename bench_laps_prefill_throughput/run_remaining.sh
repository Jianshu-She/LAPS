#!/bin/bash
# Resume benchmarks from where we left off.
# 7B: done (21/21)
# 14B: vanilla_sglang cc=1,2,4 done, need cc=8,16,32,64 + disaggregation + laps
# 32B: not started
# 72B: not started
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$(which python3)}"

# ── Common config ──
DATASET="$(cd "${SCRIPT_DIR}/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
BACKEND="${BACKEND:-mooncake}"
NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

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
    local tp_size=$3
    local prefill_gpus=$4
    local decode_gpus=$5
    local model=$6
    local results_dir=$7

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Model: ${model}, TP=${tp_size}"
    echo "  Prefill GPUs: ${prefill_gpus}, Decode GPUs: ${decode_gpus}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    CUDA_VISIBLE_DEVICES=$decode_gpus $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --tp-size $tp_size \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${results_dir}/${label}_decode.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=$prefill_gpus $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --tp-size $tp_size \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${results_dir}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 900
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  900

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

run_cc_list() {
    local label=$1
    local results_dir=$2
    shift 2
    local cc_list="$@"

    for cc in $cc_list; do
        # Skip if already done
        if [ -f "${results_dir}/${label}_cc${cc}.json" ]; then
            echo "  [skip] ${label} | cc=${cc} (already exists)"
            continue
        fi
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

trap cleanup EXIT

CC_ALL="1 2 4 8 16 32 64"

# ════════════════════════════════════════════
#  14B  (resume: vanilla_sglang cc=8+ remaining)
# ════════════════════════════════════════════
echo ""
echo "########################################"
echo "  Resuming: 14B 8GPU  ($(date))"
echo "########################################"
RESULTS_14B="${SCRIPT_DIR}/results_14b_8gpu_0312"
mkdir -p "$RESULTS_14B"

launch_servers "vanilla_sglang" "" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-14B" "$RESULTS_14B"
run_cc_list "vanilla_sglang" "$RESULTS_14B" $CC_ALL

launch_servers "disaggregation" "$LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-14B" "$RESULTS_14B"
run_cc_list "disaggregation" "$RESULTS_14B" $CC_ALL

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-14B" "$RESULTS_14B"
run_cc_list "laps" "$RESULTS_14B" $CC_ALL

$PYTHON "${SCRIPT_DIR}/gen_summary.py" "$RESULTS_14B"
echo "  14B complete at $(date)"

# ════════════════════════════════════════════
#  32B
# ════════════════════════════════════════════
echo ""
echo "########################################"
echo "  Running: 32B 8GPU  ($(date))"
echo "########################################"
RESULTS_32B="${SCRIPT_DIR}/results_32b_8gpu_0312"
mkdir -p "$RESULTS_32B"

launch_servers "vanilla_sglang" "" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-32B-Instruct" "$RESULTS_32B"
run_cc_list "vanilla_sglang" "$RESULTS_32B" $CC_ALL

launch_servers "disaggregation" "$LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-32B-Instruct" "$RESULTS_32B"
run_cc_list "disaggregation" "$RESULTS_32B" $CC_ALL

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-32B-Instruct" "$RESULTS_32B"
run_cc_list "laps" "$RESULTS_32B" $CC_ALL

$PYTHON "${SCRIPT_DIR}/gen_summary.py" "$RESULTS_32B"
echo "  32B complete at $(date)"

# ════════════════════════════════════════════
#  72B
# ════════════════════════════════════════════
echo ""
echo "########################################"
echo "  Running: 72B 8GPU  ($(date))"
echo "########################################"
RESULTS_72B="${SCRIPT_DIR}/results_72b_8gpu_0312"
mkdir -p "$RESULTS_72B"

launch_servers "vanilla_sglang" "" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-72B-Instruct" "$RESULTS_72B"
run_cc_list "vanilla_sglang" "$RESULTS_72B" $CC_ALL

launch_servers "disaggregation" "$LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-72B-Instruct" "$RESULTS_72B"
run_cc_list "disaggregation" "$RESULTS_72B" $CC_ALL

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS" 4 "0,1,2,3" "4,5,6,7" "Qwen/Qwen2.5-72B-Instruct" "$RESULTS_72B"
run_cc_list "laps" "$RESULTS_72B" $CC_ALL

$PYTHON "${SCRIPT_DIR}/gen_summary.py" "$RESULTS_72B"
echo "  72B complete at $(date)"

echo ""
echo "========================================"
echo "  All remaining benchmarks complete!"
echo "  $(date)"
echo "========================================"
