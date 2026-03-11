#!/bin/bash
# Experiment 14b: 7B tp=4
#
# Test if tensor parallelism helps batch_cg_v2 for the 7B model,
# analogous to how tp=4 helped 32B in exp13.
#
# Configurations:
#   vanilla_tp4, batch_cg_v2_tp4, piecewise_cap512_tp4
#
# Concurrency levels: 4, 8, 32, 64
# Total: 3 settings x 4 cc = 12 runs, 3 server starts
#
# Usage: bash bench_exp14_7b_tp4.sh [GPU_IDS]
#   GPU_IDS defaults to "4,5,6,7"
#
set -euo pipefail

GPU=${1:-"4,5,6,7"}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp14b_${TODAY}"
EXP13_DIR="${SCRIPT_DIR}/results_exp13_2026-03-02"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1
MODEL="Qwen/Qwen2.5-7B-Instruct"

BATCH_CG_V2_ARGS="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128"

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
    local setting_dir=$1
    local log_name=$2
    shift 2
    local extra_args="$*"

    cleanup

    echo "  [launch] GPU=$GPU model=$MODEL args=${extra_args:-<none>}"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --host $HOST --port $PORT \
        --mem-fraction-static 0.85 \
        $extra_args \
        > "${setting_dir}/${log_name}_server.log" 2>&1 &

    wait_ready "http://${HOST}:${PORT}/health" 600
}

warmup() {
    local setting_dir=$1
    echo "  [warmup] 20 requests with cc=4..."
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$MODEL" \
        --concurrency 4 \
        --num-requests 20 \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$DATASET" \
        --output-dir "${setting_dir}/warmup" \
        --http2 \
        > /dev/null 2>&1 || true
    sleep 3
}

run_cc() {
    local setting_dir=$1
    local cc=$2
    local desc=$3

    local run_dir="${setting_dir}/cc${cc}"
    mkdir -p "$run_dir"

    flush_cache

    echo "  --- ${desc} | cc=${cc} | N=${NUM_REQUESTS} ---"
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$MODEL" \
        --concurrency $cc \
        --num-requests $NUM_REQUESTS \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$DATASET" \
        --output-dir "$run_dir" \
        --http2 \
        --desc "${desc}_cc${cc}" \
        2>&1 | tee "${run_dir}/run.log"
    sleep 3
}

run_sweep() {
    local setting_dir=$1
    local desc=$2
    shift 2
    local ccs=("$@")

    for cc in "${ccs[@]}"; do
        run_cc "$setting_dir" "$cc" "$desc"
    done
}

trap cleanup EXIT

# ── Validate ──
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"; exit 1
fi
if [ ! -f "$BASELINE_PY" ]; then
    echo "ERROR: baseline.py not found: $BASELINE_PY"; exit 1
fi

echo "Experiment 14b: 7B tp=4"
echo "  GPU:       ${GPU}"
echo "  Port:      ${PORT}"
echo "  Requests:  ${NUM_REQUESTS}"
echo "  Exp13 dir: ${EXP13_DIR}"
echo "  Results:   ${BASE_DIR}/"
echo ""

CCS=(4 8 32 64)

# ═══════════════════════════════════════════════════════════════════════
#  1. vanilla_tp4
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  vanilla_tp4"
echo "################################################################"

SDIR="${BASE_DIR}/vanilla_tp4"
mkdir -p "$SDIR"
launch_server "$SDIR" "vanilla_tp4" --tp 4
warmup "$SDIR"
run_sweep "$SDIR" "exp14b_vanilla_tp4" "${CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
#  2. batch_cg_v2_tp4
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  batch_cg_v2_tp4"
echo "################################################################"

SDIR="${BASE_DIR}/batch_cg_v2_tp4"
mkdir -p "$SDIR"
launch_server "$SDIR" "batch_cg_v2_tp4" --tp 4 $BATCH_CG_V2_ARGS
warmup "$SDIR"
run_sweep "$SDIR" "exp14b_batch_cg_v2_tp4" "${CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
#  3. piecewise_cap512_tp4
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  piecewise_cap512_tp4"
echo "################################################################"

SDIR="${BASE_DIR}/piecewise_cap512_tp4"
mkdir -p "$SDIR"
launch_server "$SDIR" "piecewise_cap512_tp4" --tp 4 \
    --enable-piecewise-cuda-graph --disable-cuda-graph --piecewise-cuda-graph-max-tokens 512
warmup "$SDIR"
run_sweep "$SDIR" "exp14b_piecewise_cap512_tp4" "${CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY"
echo "################################################################"

$PYTHON -c "
import json, os

base = '${BASE_DIR}'
exp13 = '${EXP13_DIR}'

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def load_14b(setting, cc):
    return load(os.path.join(base, setting, f'cc{cc}', 'summary.json'))

def load_exp13(setting, cc):
    return load(os.path.join(exp13, '7b', setting, f'cc{cc}', 'summary.json'))

def rps(d):
    return d['rps'] if d else None

def latency(d):
    return d['latency_ms']['avg'] if d and 'latency_ms' in d else None

def fmt(v):
    return f'{v:>8.1f}' if v else f'{\"N/A\":>8s}'

def fmt_speedup(v, base_v):
    if v and base_v and base_v > 0:
        return f'{v/base_v:>8.3f}x'
    return f'{\"N/A\":>8s}'

ccs = [4, 8, 32, 64]

# ═══════════════════════════════════════════════════════════════════
# 7B tp=4 Results
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  7B MODEL: tp=4 RESULTS')
print('=' * 100)

settings_tp4 = ['vanilla_tp4', 'batch_cg_v2_tp4', 'piecewise_cap512_tp4']

# RPS table
print()
print('  --- RPS (requests/sec) ---')
hdr = f'{\"Setting\":<24s}'
for cc in ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

# Exp13 tp=1 baselines
for s in ['vanilla', 'batch_cg_v2', 'piecewise_cap512']:
    row = f'{\"exp13_\" + s + \"_tp1\":<24s}'
    for cc in ccs:
        d = load_exp13(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

print(f'  {\"-\" * len(hdr)}')

# tp=4 results
for s in settings_tp4:
    row = f'{s:<24s}'
    for cc in ccs:
        d = load_14b(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup: tp4 batch_cg_v2 vs tp4 vanilla
print()
print('  --- Speedup vs tp=4 vanilla ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_tp4:
    row = f'{s:<24s}'
    for cc in ccs:
        d_v = load_14b('vanilla_tp4', cc)
        d_s = load_14b(s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# Speedup: tp=4 batch_cg_v2 vs tp=1 batch_cg_v2
print()
print('  --- tp=4 vs tp=1 (same setting) ---')
mapping = {
    'vanilla_tp4': 'vanilla',
    'batch_cg_v2_tp4': 'batch_cg_v2',
    'piecewise_cap512_tp4': 'piecewise_cap512',
}
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s_tp4, s_tp1 in mapping.items():
    row = f'{s_tp4 + \" vs tp1\":<24s}'
    for cc in ccs:
        d_tp1 = load_exp13(s_tp1, cc)
        d_tp4 = load_14b(s_tp4, cc)
        row += f'  {fmt_speedup(rps(d_tp4), rps(d_tp1))}'
    print(f'  {row}')

# Latency table
print()
print('  --- Latency (avg ms) ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_tp4:
    row = f'{s:<24s}'
    for cc in ccs:
        d = load_14b(s, cc)
        row += f'  {fmt(latency(d))}'
    print(f'  {row}')

# Fail check
print()
print('=== Fail Count Check ===')
all_ok = True
for s in settings_tp4:
    for cc in ccs:
        d = load_14b(s, cc)
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
echo "  EXPERIMENT 14b COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
