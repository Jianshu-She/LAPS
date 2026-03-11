#!/bin/bash
# Experiment 7: Piecewise CG vs Vanilla using baseline.py framework
#
# Goal: Reproduce Exp 5/6 finding (piecewise CG <= vanilla) using the
#       baseline.py load tester which hits /v1/chat/completions with HTTP/2,
#       and Qwen2.5-7B-Instruct (smaller model where kernel launch overhead
#       should be proportionally larger — potentially more favorable for CG).
#
# 2 server configs:
#   1) vanilla       — no CUDA graph at all
#   2) piecewise_cg  — piecewise CUDA graph only
#
# Client: baseline.py  (HTTP/2, /v1/chat/completions)
# Dataset: 10K human prompts (variable lengths)
# Concurrency: 2, 4, 8, 16, 32, 64, 128
# Max tokens: 1 (prefill-only)
#
# Usage: bash bench_exp7_piecewise_baseline.sh [GPU_ID]
#   GPU_ID defaults to 5
#
set -euo pipefail

GPU=${1:-5}
MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp7_baseline_${TODAY}"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1
CONCURRENCIES=(2 4 8 16 32 64 128)

# ── Server argument presets ──

declare -A SERVER_ARGS
SERVER_ARGS[vanilla]=""
SERVER_ARGS[piecewise_cg]="--enable-piecewise-cuda-graph --disable-cuda-graph"

SETTINGS=(vanilla piecewise_cg)

# ───────────────────────── helpers ─────────────────────────

cleanup() {
    echo "[cleanup] Killing server on port ${PORT}..."
    pkill -f "sglang.launch_server.*--port ${PORT}" 2>/dev/null || true
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

flush_cache() {
    curl -s -X POST "http://${HOST}:${PORT}/flush_cache" > /dev/null 2>&1 || true
    sleep 2
}

launch_server() {
    local label=$1
    local extra_args=$2
    local log_dir=$3

    echo ""
    echo "============================================================"
    echo "  Launching server: ${label}"
    echo "  GPU: ${GPU}, Model: ${MODEL}"
    echo "  Args: ${extra_args:-<none>}"
    echo "============================================================"

    cleanup

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $PORT \
        --mem-fraction-static 0.85 \
        $extra_args \
        > "${log_dir}/${label}_server.log" 2>&1 &

    wait_ready "http://${HOST}:${PORT}/health" 600
}

trap cleanup EXIT

# ── Validate dataset ──
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    exit 1
fi
if [ ! -f "$BASELINE_PY" ]; then
    echo "ERROR: baseline.py not found: $BASELINE_PY"
    exit 1
fi

echo "Experiment 7: Piecewise CG vs Vanilla (baseline.py framework)"
echo "  Model:    ${MODEL}"
echo "  GPU:      ${GPU}"
echo "  Port:     ${PORT}"
echo "  Dataset:  ${DATASET}"
echo "  Requests: ${NUM_REQUESTS}"
echo "  CC:       ${CONCURRENCIES[*]}"
echo "  Results:  ${BASE_DIR}/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# RUN ALL SETTINGS
# ═══════════════════════════════════════════════════════════════════════

for setting in "${SETTINGS[@]}"; do
    SETTING_DIR="${BASE_DIR}/${setting}"
    mkdir -p "$SETTING_DIR"

    launch_server "$setting" "${SERVER_ARGS[$setting]}" "$SETTING_DIR"

    # Warmup
    echo "[warmup] 20 requests with cc=4..."
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$MODEL" \
        --concurrency 4 \
        --num-requests 20 \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$DATASET" \
        --output-dir "${SETTING_DIR}/warmup" \
        --http2 \
        > /dev/null 2>&1 || true
    sleep 3

    # Sweep concurrency levels
    for cc in "${CONCURRENCIES[@]}"; do
        RUN_DIR="${SETTING_DIR}/cc${cc}"
        mkdir -p "$RUN_DIR"

        flush_cache

        echo "  --- ${setting} | cc=${cc} | N=${NUM_REQUESTS} ---"
        $PYTHON "$BASELINE_PY" \
            --base-url "http://${HOST}:${PORT}" \
            --model "$MODEL" \
            --concurrency $cc \
            --num-requests $NUM_REQUESTS \
            --max-tokens $MAX_TOKENS \
            --prompt-file "$DATASET" \
            --output-dir "$RUN_DIR" \
            --http2 \
            --desc "${setting}_cc${cc}" \
            2>&1 | tee "${RUN_DIR}/run.log"
        sleep 3
    done
done

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY — Experiment 7: Piecewise CG vs Vanilla (baseline.py)"
echo "################################################################"

$PYTHON -c "
import json, os, sys

base_dir = '${BASE_DIR}'
settings = ['vanilla', 'piecewise_cg']
ccs = [2, 4, 8, 16, 32, 64, 128]

def load_summary(setting, cc):
    f = os.path.join(base_dir, setting, f'cc{cc}', 'summary.json')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)

# ── RPS table ──
print()
print('=== Requests Per Second (RPS) ===')
print()
header = f'{\"Setting\":<20s}'
for cc in ccs:
    header += f'  cc={cc:<5d}'
print(header)
print('-' * len(header))

for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load_summary(s, cc)
        if d:
            row += f'  {d[\"rps\"]:>8.1f}'
        else:
            row += f'  {\"N/A\":>8s}'
    print(row)

# ── Prefill throughput table ──
print()
print('=== Prefill Throughput (tok/s) ===')
print()
header = f'{\"Setting\":<20s}'
for cc in ccs:
    header += f'  cc={cc:<5d}'
print(header)
print('-' * len(header))

for s in settings:
    row = f'{s:<20s}'
    for cc in ccs:
        d = load_summary(s, cc)
        if d and d['wall_time_sec'] > 0:
            tput = d['tokens']['total_prompt_tokens'] / d['wall_time_sec']
            row += f'  {tput:>8.0f}'
        else:
            row += f'  {\"N/A\":>8s}'
    print(row)

# ── Latency table ──
print()
print('=== Latency avg / p50 / p90 / p99 (ms) ===')
print()
for s in settings:
    print(f'  {s}:')
    header = f'    {\"cc\":<6s}  {\"avg\":>8s}  {\"p50\":>8s}  {\"p90\":>8s}  {\"p99\":>8s}'
    print(header)
    print('    ' + '-' * (len(header) - 4))
    for cc in ccs:
        d = load_summary(s, cc)
        if d:
            lat = d['latency_ms']
            print(f'    {cc:<6d}  {lat[\"avg\"]:>8.1f}  {lat[\"p50\"]:>8.1f}  {lat[\"p90\"]:>8.1f}  {lat[\"p99\"]:>8.1f}')
        else:
            print(f'    {cc:<6d}  {\"N/A\":>8s}  {\"N/A\":>8s}  {\"N/A\":>8s}  {\"N/A\":>8s}')
    print()

# ── Speedup ratio (piecewise / vanilla) ──
print()
print('=== Speedup: piecewise_cg / vanilla ===')
print()
header = f'{\"Metric\":<20s}'
for cc in ccs:
    header += f'  cc={cc:<5d}'
print(header)
print('-' * len(header))

# RPS speedup
row_rps = f'{\"RPS ratio\":<20s}'
row_tput = f'{\"Throughput ratio\":<20s}'
for cc in ccs:
    d_v = load_summary('vanilla', cc)
    d_p = load_summary('piecewise_cg', cc)
    if d_v and d_p and d_v['rps'] > 0:
        row_rps += f'  {d_p[\"rps\"]/d_v[\"rps\"]:>7.3f}x'
    else:
        row_rps += f'  {\"N/A\":>8s}'
    if d_v and d_p and d_v['wall_time_sec'] > 0 and d_p['wall_time_sec'] > 0:
        tput_v = d_v['tokens']['total_prompt_tokens'] / d_v['wall_time_sec']
        tput_p = d_p['tokens']['total_prompt_tokens'] / d_p['wall_time_sec']
        if tput_v > 0:
            row_tput += f'  {tput_p/tput_v:>7.3f}x'
        else:
            row_tput += f'  {\"N/A\":>8s}'
    else:
        row_tput += f'  {\"N/A\":>8s}'
print(row_rps)
print(row_tput)

# ── Fail count check ──
print()
print('=== Fail Count Check ===')
all_ok = True
for s in settings:
    for cc in ccs:
        d = load_summary(s, cc)
        if d and d.get('fail', 0) > 0:
            print(f'  WARNING: {s}/cc{cc} has {d[\"fail\"]} failures!')
            all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 7 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
