#!/bin/bash
# Benchmark prefill throughput on 4x H100 GPUs with nixl backend (no InfiniBand required).
#
# Supported models:
#   7b  → Qwen2.5-7B   (TP=1, GPU 0 prefill, GPU 1 decode)
#   14b → Qwen2.5-14B  (TP=2, GPU 0,1 prefill, GPU 2,3 decode)
#
# 3 settings x 7 concurrency levels = 21 data points per model.
#
# Usage:
#   bash bench_h100_4gpu.sh 7b          # run 7B only
#   bash bench_h100_4gpu.sh 14b         # run 14B only
#
# Environment variable overrides:
#   NUM_PROMPTS=1000 bash bench_h100_4gpu.sh 7b   # fewer prompts (faster)
#   BACKEND=mooncake bash bench_h100_4gpu.sh 7b    # override backend
set -uo pipefail

# ───────────────────────── model selection ─────────────────────────

MODEL_SIZE="${1:-7b}"
case "$MODEL_SIZE" in
    7b|7B)
        MODEL="Qwen/Qwen2.5-7B"
        MODEL_SHORT="7b"
        TP_SIZE=2
        PREFILL_GPUS="0,1"
        DECODE_GPUS="2,3"
        ;;
    14b|14B)
        MODEL="Qwen/Qwen2.5-14B"
        MODEL_SHORT="14b"
        TP_SIZE=2
        PREFILL_GPUS="0,1"
        DECODE_GPUS="2,3"
        ;;
    *)
        echo "Usage: bash bench_h100_4gpu.sh [7b|14b]"
        exit 1
        ;;
esac

# ───────────────────────── configuration ─────────────────────────

DATASET="$(cd "$(dirname "$0")/../.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
BACKEND="${BACKEND:-nixl}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results_${MODEL_SHORT}_h100_4gpu_${TIMESTAMP}"
RAW_DIR="${RESULTS_DIR}/raw"
PYTHON="${PYTHON:-$(which python3)}"

NUM_PROMPTS="${NUM_PROMPTS:-10000}"
MAX_NEW_TOKENS=1

CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32 64}"
SETTINGS="vanilla_sglang disaggregation laps"

mkdir -p "$RAW_DIR"

echo "============================================================"
echo "  LAPS H100 Benchmark (4 GPU, nixl)"
echo "  Model:     ${MODEL} (${MODEL_SHORT})"
echo "  TP:        ${TP_SIZE}"
echo "  Backend:   ${BACKEND}"
echo "  Prompts:   ${NUM_PROMPTS}"
echo "  GPUs:      ${PREFILL_GPUS} (prefill), ${DECODE_GPUS} (decode)"
echo "  Results:   ${RESULTS_DIR}"
echo "============================================================"

# ───────────────────────── helpers ─────────────────────────

cleanup() {
    pkill -9 -f "sglang.launch_server.*--port ${PREFILL_PORT}" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server.*--port ${DECODE_PORT}" 2>/dev/null || true
    pkill -9 -f "sglang_router.launch_router.*--port ${ROUTER_PORT}" 2>/dev/null || true
    sleep 5
    for _i in $(seq 1 30); do
        if ! curl -s "http://${HOST}:${DECODE_PORT}/health" >/dev/null 2>&1 && \
           ! curl -s "http://${HOST}:${PREFILL_PORT}/health" >/dev/null 2>&1 && \
           ! curl -s "http://${HOST}:${ROUTER_PORT}/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
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

get_prefill_args() {
    local setting=$1
    case "$setting" in
        vanilla_sglang)
            echo ""
            ;;
        disaggregation)
            echo "--enable-laps-scheduler --laps-length-threshold 256"
            ;;
        laps)
            echo "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --enable-laps-scheduler --laps-length-threshold 256"
            ;;
    esac
}

launch_servers() {
    local label=$1
    local prefill_extra_args=$2

    echo ""
    echo "------------------------------------------------------------"
    echo "  ${label} | ${MODEL_SHORT} | launching servers..."
    echo "------------------------------------------------------------"

    cleanup

    # Decode server first (nixl needs decode ready before prefill)
    CUDA_VISIBLE_DEVICES=$DECODE_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${RAW_DIR}/${label}_decode.log" 2>&1 &

    wait_ready "http://${HOST}:${DECODE_PORT}/health" 300
    sleep 5  # extra wait for nixl bootstrap

    # Prefill server (after decode is ready)
    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${RAW_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 300

    # Router
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RAW_DIR}/${label}_router.log" 2>&1 &

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
}

trap cleanup EXIT

# ───────────────────────── run 3 settings ─────────────────────────

for setting in $SETTINGS; do
    EXTRA_ARGS=$(get_prefill_args "$setting")
    launch_servers "${setting}" "${EXTRA_ARGS}"

    for cc in $CONCURRENCY_LEVELS; do
        echo ""
        echo "  --- ${setting} | cc=${cc} ---"
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" \
            --url "http://${HOST}:${ROUTER_PORT}" \
            --num-prompts $NUM_PROMPTS \
            --max-new-tokens $MAX_NEW_TOKENS \
            --concurrency $cc \
            --output "${RAW_DIR}/${setting}_cc${cc}.json" \
            2>&1 | tee "${RAW_DIR}/${setting}_cc${cc}.txt" || true
        sleep 2
    done

    cleanup
done

# ───────────────────────── generate summary ─────────────────────────

$PYTHON -c "
import json, os

raw = '${RAW_DIR}'
settings = '${SETTINGS}'.split()
ccs = [int(c) for c in '${CONCURRENCY_LEVELS}'.split()]
setting_labels = {
    'vanilla_sglang': 'Vanilla SGLang',
    'disaggregation': 'Dual-Queue',
    'laps': 'LAPS',
}

def load(setting, cc):
    p = os.path.join(raw, f'{setting}_cc{cc}.json')
    try:
        with open(p) as f:
            return json.load(f)
    except:
        return None

lines = []
lines.append('=' * 80)
lines.append('  BENCHMARK SUMMARY: ${MODEL} (TP=${TP_SIZE}) on H100')
lines.append('  Prefill GPUs: ${PREFILL_GPUS}  |  Decode GPUs: ${DECODE_GPUS}')
lines.append('  Prompts: ${NUM_PROMPTS}  |  Backend: ${BACKEND}')
lines.append('=' * 80)

hdr = f'{\"Setting\":<20s}' + ''.join(f'  cc={cc:<5d}' for cc in ccs)
sep = '-' * len(hdr)

for metric, title in [
    ('request_throughput_req_s', 'Request Throughput (req/s)'),
    ('mean_ttft_ms', 'Mean TTFT (ms)'),
    ('p90_ttft_ms', 'P90 TTFT (ms)'),
]:
    lines.append('')
    lines.append(f'--- {title} ---')
    lines.append(hdr)
    lines.append(sep)
    for s in settings:
        label = setting_labels.get(s, s)
        row = f'{label:<20s}'
        for cc in ccs:
            d = load(s, cc)
            row += f'  {d[metric]:>8.1f}' if d and metric in d else f'  {\"N/A\":>8s}'
        lines.append(row)

# Speedup
lines.append('')
lines.append('--- Speedup vs Vanilla SGLang (req/s) ---')
lines.append(hdr)
lines.append(sep)
for s in settings:
    label = setting_labels.get(s, s)
    row = f'{label:<20s}'
    for cc in ccs:
        d = load(s, cc)
        v = load('vanilla_sglang', cc)
        if d and v and v['request_throughput_req_s'] > 0:
            ratio = d['request_throughput_req_s'] / v['request_throughput_req_s']
            row += f'  {ratio:>7.2f}x'
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
echo "  Benchmark complete: ${MODEL} (${MODEL_SHORT})"
echo "  Results:  ${RESULTS_DIR}/"
echo "  Summary:  ${RESULTS_DIR}/summary.txt"
echo "  Plots:    ${RESULTS_DIR}/plots/"
echo "============================================================"
