#!/bin/bash
# Single-GPU benchmark: vanilla SGLang vs LAPS (no PD disaggregation)
#
# Usage:
#   bash bench_1gpu.sh            # run all 3 models (0.5b, 3b, 7b)
#   bash bench_1gpu.sh 7b         # run only 7B
#   bash bench_1gpu.sh 0.5b 3b    # run 0.5B and 3B
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PORT=30300
HOST="127.0.0.1"
PYTHON="${PYTHON:-$(which python3)}"
NUM_PROMPTS="${NUM_PROMPTS:-10000}"
CONCURRENCY_LEVELS="${CONCURRENCY_LEVELS:-1 2 4 8 16 32 64}"
GPU="${GPU:-0}"

export SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=false

if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="0.5b 3b 7b"
fi

# ───────────────────────── helpers ─────────────────────────

cleanup() {
    pkill -9 -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
    sleep 5
}

wait_ready() {
    local url=$1 timeout=${2:-300}
    echo -n "[wait] $url ..."
    for i in $(seq 1 $timeout); do
        curl -s "$url" > /dev/null 2>&1 && echo " ready (${i}s)" && return 0
        sleep 1
    done
    echo " TIMEOUT"; return 1
}

run_setting() {
    local label=$1
    shift
    local extra_args="$*"

    echo ""
    echo "------------------------------------------------------------"
    echo "  ${label} | launching server..."
    echo "------------------------------------------------------------"

    cleanup

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $PORT \
        --mem-fraction-static 0.85 \
        $extra_args \
        > "${RAW_DIR}/${label}_server.log" 2>&1 &

    wait_ready "http://${HOST}:${PORT}/health" 300

    echo "[warmup]..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" --url "http://${HOST}:${PORT}" \
        --num-prompts 20 --max-new-tokens 1 --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3

    for cc in $CONCURRENCY_LEVELS; do
        echo "  ${label} cc=${cc}..."
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" --url "http://${HOST}:${PORT}" \
            --num-prompts $NUM_PROMPTS --max-new-tokens 1 --concurrency $cc \
            --output "${RAW_DIR}/${label}_cc${cc}.json" \
            2>&1 | tee "${RAW_DIR}/${label}_cc${cc}.txt" || true
        sleep 2
    done

    cleanup
}

generate_summary() {
    local results_dir=$1
    local raw_dir="${results_dir}/raw"
    local settings="vanilla_sglang laps"

    $PYTHON -c "
import json, os
raw = '${raw_dir}'
settings = '${settings}'.split()
ccs = [int(c) for c in '${CONCURRENCY_LEVELS}'.split()]
labels = {'vanilla_sglang': 'Vanilla SGLang', 'laps': 'LAPS'}
def load(s, cc):
    try:
        with open(os.path.join(raw, f'{s}_cc{cc}.json')) as f: return json.load(f)
    except: return None
lines = ['='*70, '  Single GPU: ${MODEL} (no disaggregation)', '  Prompts: ${NUM_PROMPTS}', '='*70]
hdr = f'{\"Setting\":<20s}' + ''.join(f'  cc={cc:<5d}' for cc in ccs)
sep = '-' * len(hdr)
for metric, title in [('request_throughput_req_s','Request Throughput (req/s)'),('mean_ttft_ms','Mean TTFT (ms)'),('p90_ttft_ms','P90 TTFT (ms)')]:
    lines += ['', f'--- {title} ---', hdr, sep]
    for s in settings:
        row = f'{labels.get(s,s):<20s}'
        for cc in ccs:
            d = load(s, cc)
            row += f'  {d[metric]:>8.1f}' if d and metric in d else f'  {\"N/A\":>8s}'
        lines.append(row)
lines += ['', '--- Speedup (LAPS / Vanilla) ---', hdr, sep]
row = f'{\"LAPS\":<20s}'
for cc in ccs:
    l, v = load('laps', cc), load('vanilla_sglang', cc)
    if l and v and v['request_throughput_req_s'] > 0:
        row += f'  {l[\"request_throughput_req_s\"]/v[\"request_throughput_req_s\"]:>7.2f}x'
    else: row += f'  {\"N/A\":>8s}'
lines.append(row)
text = '\n'.join(lines)
print(text)
with open('${results_dir}/summary.txt','w') as f: f.write(text)
"
}

trap cleanup EXIT

# ───────────────────────── run models ─────────────────────────

PASSED=0
FAILED=0

for model_size in $MODELS; do
    case "$model_size" in
        0.5b|0.5B) MODEL="Qwen/Qwen2.5-0.5B"; MODEL_SHORT="0.5b" ;;
        3b|3B)     MODEL="Qwen/Qwen2.5-3B";   MODEL_SHORT="3b" ;;
        7b|7B)     MODEL="Qwen/Qwen2.5-7B";   MODEL_SHORT="7b" ;;
        *)
            echo "[ERROR] Unknown model: ${model_size} (valid: 0.5b, 3b, 7b)"
            FAILED=$((FAILED + 1))
            continue
            ;;
    esac

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULTS_DIR="${SCRIPT_DIR}/results_${MODEL_SHORT}_1gpu_${TIMESTAMP}"
    RAW_DIR="${RESULTS_DIR}/raw"
    mkdir -p "$RAW_DIR"

    echo ""
    echo "============================================================"
    echo "  Single GPU Benchmark: ${MODEL} (${MODEL_SHORT})"
    echo "  GPU:     ${GPU}"
    echo "  Prompts: ${NUM_PROMPTS}"
    echo "  Results: ${RESULTS_DIR}"
    echo "  Time:    $(date)"
    echo "============================================================"

    run_setting "vanilla_sglang"
    run_setting "laps" --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --enable-laps-scheduler --laps-length-threshold 256

    generate_summary "${RESULTS_DIR}"

    echo ""
    echo "  ${MODEL_SHORT}: Done → ${RESULTS_DIR}/summary.txt"
    PASSED=$((PASSED + 1))
done

echo ""
echo "============================================================"
echo "  All done. Passed: ${PASSED}, Failed: ${FAILED}"
echo "  Time: $(date)"
echo "============================================================"
