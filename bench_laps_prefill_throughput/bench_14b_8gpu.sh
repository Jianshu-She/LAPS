#!/bin/bash
# Benchmark prefill throughput on Qwen2.5-7B with 8 GPUs (TP=4) and concurrency sweep.
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
IB_DEVICE="${IB_DEVICE:-mlx5_0}"
BACKEND="${BACKEND:-mooncake}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results_14b_8gpu_${TIMESTAMP}"
RAW_DIR="${RESULTS_DIR}/raw"
PYTHON="${PYTHON:-$(which python3)}"

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
TP_SIZE=4
PREFILL_GPUS="0,1,2,3"
DECODE_GPUS="4,5,6,7"

CONCURRENCY_LEVELS="1 2 4 8 16 32 64"
SETTINGS="vanilla_sglang disaggregation laps"

mkdir -p "$RAW_DIR"

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
        > "${RAW_DIR}/${label}_decode.log" 2>&1 &

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
        > "${RAW_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 900
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  900

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RAW_DIR}/${label}_router.log" 2>&1 &

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

check_server_alive() {
    curl -s --max-time 5 "http://${HOST}:${PREFILL_PORT}/health" > /dev/null 2>&1 && \
    curl -s --max-time 5 "http://${HOST}:${DECODE_PORT}/health" > /dev/null 2>&1
}

MAX_RETRIES=3

run_concurrency_sweep() {
    local label=$1
    local prefill_extra_args=$2

    for cc in $CONCURRENCY_LEVELS; do
        local attempt=0
        local success=false

        while [ $attempt -lt $MAX_RETRIES ] && [ "$success" = "false" ]; do
            # Check if servers are alive before running
            if ! check_server_alive; then
                echo ""
                echo "  [CRASH DETECTED] Server down before ${label} cc=${cc} (attempt $((attempt+1))/${MAX_RETRIES})"
                echo "  [RESTART] Relaunching servers for ${label}..."
                launch_servers "$label" "$prefill_extra_args"
            fi

            echo ""
            echo "  --- ${label} | cc=${cc} (attempt $((attempt+1))) ---"
            $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
                --dataset "$DATASET" \
                --url "http://${HOST}:${ROUTER_PORT}" \
                --num-prompts $NUM_PROMPTS \
                --max-new-tokens $MAX_NEW_TOKENS \
                --concurrency $cc \
                --output "${RAW_DIR}/${label}_cc${cc}.json" \
                2>&1 | tee "${RAW_DIR}/${label}_cc${cc}.txt"

            # Check if benchmark produced a valid result (JSON file exists and has data)
            if [ -f "${RAW_DIR}/${label}_cc${cc}.json" ] && check_server_alive; then
                success=true
            else
                echo "  [WARN] Benchmark may have failed for ${label} cc=${cc}"
                attempt=$((attempt + 1))
                # Remove partial result
                rm -f "${RAW_DIR}/${label}_cc${cc}.json"
            fi
            sleep 2
        done

        if [ "$success" = "false" ]; then
            echo "  [FAIL] ${label} cc=${cc} failed after ${MAX_RETRIES} retries, skipping."
        fi
    done
}

trap cleanup EXIT

# ───────────────────────── run 3 settings ─────────────────────────

LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

launch_servers "vanilla_sglang" ""
run_concurrency_sweep "vanilla_sglang" ""

launch_servers "disaggregation" "$LAPS_ARGS"
run_concurrency_sweep "disaggregation" "$LAPS_ARGS"

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"
run_concurrency_sweep "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"

# ───────────────────────── generate summary ─────────────────────────

$PYTHON -c "
import json, os

raw = '${RAW_DIR}'
settings = '${SETTINGS}'.split()
ccs = [int(c) for c in '${CONCURRENCY_LEVELS}'.split()]

def load(setting, cc):
    p = os.path.join(raw, f'{setting}_cc{cc}.json')
    try:
        with open(p) as f:
            return json.load(f)
    except:
        return None

lines = []
lines.append('=' * 80)
lines.append('  BENCHMARK SUMMARY: ${MODEL} (TP=${TP_SIZE})')
lines.append('  Prefill GPUs: ${PREFILL_GPUS}  |  Decode GPUs: ${DECODE_GPUS}')
lines.append('  Prompts: ${NUM_PROMPTS}  |  Max New Tokens: ${MAX_NEW_TOKENS}')
lines.append('=' * 80)

# RPS table
lines.append('')
lines.append('--- Request Throughput (req/s) ---')
hdr = f'{\"Setting\":<20s}' + ''.join(f'  cc={cc:<5d}' for cc in ccs)
lines.append(hdr)
lines.append('-' * len(hdr))
for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load(s, cc)
        row += f'  {d[\"request_throughput_req_s\"]:>8.1f}' if d else f'  {\"N/A\":>8s}'
    lines.append(row)

# Prefill throughput table
lines.append('')
lines.append('--- Prefill Throughput (tok/s) ---')
lines.append(hdr)
lines.append('-' * len(hdr))
for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load(s, cc)
        row += f'  {d[\"prefill_throughput_tok_s\"]:>8.1f}' if d else f'  {\"N/A\":>8s}'
    lines.append(row)

# Latency table
lines.append('')
lines.append('--- Mean TTFT Latency (ms) ---')
lines.append(hdr)
lines.append('-' * len(hdr))
for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load(s, cc)
        row += f'  {d[\"mean_ttft_ms\"]:>8.1f}' if d else f'  {\"N/A\":>8s}'
    lines.append(row)

# Speedup vs vanilla
lines.append('')
lines.append('--- Speedup vs vanilla_sglang (RPS) ---')
lines.append(hdr)
lines.append('-' * len(hdr))
for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load(s, cc)
        v = load('vanilla_sglang', cc)
        if d and v and v['request_throughput_req_s'] > 0:
            ratio = d['request_throughput_req_s'] / v['request_throughput_req_s']
            row += f'  {ratio:>7.3f}x'
        else:
            row += f'  {\"N/A\":>8s}'
    lines.append(row)

lines.append('')
lines.append('Raw results: raw/')
lines.append('')

text = '\n'.join(lines)
print(text)

with open(os.path.join('${RESULTS_DIR}', 'summary.txt'), 'w') as f:
    f.write(text)
"

# ───────────────────────── generate plots ─────────────────────────

echo "[plot] Generating plots..."
$PYTHON "${SCRIPT_DIR}/plot_results.py" "${RESULTS_DIR}" \
    --settings "${SETTINGS// /,}" \
    --cc "${CONCURRENCY_LEVELS// /,}" \
    2>&1 || echo "[plot] Warning: plotting failed (matplotlib may not be installed)"

echo ""
echo "============================================================"
echo "  All 3 settings complete."
echo "  Results in: ${RESULTS_DIR}/"
echo "  Plots:      ${RESULTS_DIR}/plots/"
echo "  Summary:    ${RESULTS_DIR}/summary.txt"
echo "============================================================"
