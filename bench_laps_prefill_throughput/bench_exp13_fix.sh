#!/bin/bash
# Experiment 13: Make Batch Prefill CG Beat Vanilla
#
# Code changes applied:
#   1. Allow bs=1 in batch_prefill_cg (removed bs<=1 guard)
#   2. Skip piecewise fallback when batch_prefill is enabled (eager fallback)
#
# Configurations:
#   7B (tp=1):
#     vanilla, batch_cg_v2, batch_cg_v2_sorted, piecewise_cap512
#   32B (tp=4):
#     vanilla_32b, batch_cg_v2_32b, batch_cg_v2_sorted_32b
#
# Total: 25 runs, 7 server starts
#
# Usage: bash bench_exp13_fix.sh [GPU_ID_7B] [GPU_IDS_32B]
#   GPU_ID_7B defaults to 5 (single GPU for 7B)
#   GPU_IDS_32B defaults to "4,5,6,7" (4 GPUs for 32B tp=4)
#
set -euo pipefail

GPU_7B=${1:-5}
GPU_32B=${2:-"4,5,6,7"}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp13_${TODAY}"
EXP8_DIR="${SCRIPT_DIR}/results_exp8_2026-03-02"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1

MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_32B="Qwen/Qwen2.5-32B"

# Create sorted dataset if not exists
SORTED_DATASET="${BASE_DIR}/human_prompts_stream_10000_sorted.jsonl"
if [ ! -f "$SORTED_DATASET" ]; then
    echo "[prep] Creating sorted dataset..."
    $PYTHON "${SCRIPT_DIR}/sort_dataset.py" \
        --input "$DATASET" \
        --output "$SORTED_DATASET"
fi

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

# launch_server GPU_STR MODEL SETTING_DIR LOG_NAME EXTRA_ARGS...
launch_server() {
    local gpu_str=$1
    local model=$2
    local setting_dir=$3
    local log_name=$4
    shift 4
    local extra_args="$*"

    cleanup

    echo "  [launch] GPU=$gpu_str model=$model args=${extra_args:-<none>}"
    CUDA_VISIBLE_DEVICES=$gpu_str $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --host $HOST --port $PORT \
        --mem-fraction-static 0.85 \
        $extra_args \
        > "${setting_dir}/${log_name}_server.log" 2>&1 &

    wait_ready "http://${HOST}:${PORT}/health" 600
}

# warmup SETTING_DIR MODEL [PROMPT_FILE]
warmup() {
    local setting_dir=$1
    local model=$2
    local prompt_file=${3:-$DATASET}
    echo "  [warmup] 20 requests with cc=4..."
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$model" \
        --concurrency 4 \
        --num-requests 20 \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$prompt_file" \
        --output-dir "${setting_dir}/warmup" \
        --http2 \
        > /dev/null 2>&1 || true
    sleep 3
}

# run_cc SETTING_DIR MODEL CC DESC [PROMPT_FILE]
run_cc() {
    local setting_dir=$1
    local model=$2
    local cc=$3
    local desc=$4
    local prompt_file=${5:-$DATASET}

    local run_dir="${setting_dir}/cc${cc}"
    mkdir -p "$run_dir"

    flush_cache

    echo "  --- ${desc} | cc=${cc} | N=${NUM_REQUESTS} ---"
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$model" \
        --concurrency $cc \
        --num-requests $NUM_REQUESTS \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$prompt_file" \
        --output-dir "$run_dir" \
        --http2 \
        --desc "${desc}_cc${cc}" \
        2>&1 | tee "${run_dir}/run.log"
    sleep 3
}

# run_sweep SETTING_DIR MODEL DESC PROMPT_FILE CC1 CC2 ...
run_sweep() {
    local setting_dir=$1
    local model=$2
    local desc=$3
    local prompt_file=$4
    shift 4
    local ccs=("$@")

    for cc in "${ccs[@]}"; do
        run_cc "$setting_dir" "$model" "$cc" "$desc" "$prompt_file"
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

echo "Experiment 13: Make Batch Prefill CG Beat Vanilla"
echo "  GPU (7B):    ${GPU_7B}"
echo "  GPU (32B):   ${GPU_32B}"
echo "  Port:        ${PORT}"
echo "  Dataset:     ${DATASET}"
echo "  Sorted:      ${SORTED_DATASET}"
echo "  Requests:    ${NUM_REQUESTS}"
echo "  Exp8 dir:    ${EXP8_DIR}"
echo "  Results:     ${BASE_DIR}/"
echo ""

# Batch CG v2 common args (small graphs, bs=1 allowed)
BATCH_CG_V2_ARGS="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128"

# ═══════════════════════════════════════════════════════════════════════
#  PART 1: 7B Model (tp=1, single GPU)
# ═══════════════════════════════════════════════════════════════════════

CCS_7B=(4 8 32 64)

echo ""
echo "################################################################"
echo "  7B: vanilla (baseline)"
echo "################################################################"

SDIR="${BASE_DIR}/7b/vanilla"
mkdir -p "$SDIR"
launch_server "$GPU_7B" "$MODEL_7B" "$SDIR" "vanilla" ""
warmup "$SDIR" "$MODEL_7B"
run_sweep "$SDIR" "$MODEL_7B" "exp13_7b_vanilla" "$DATASET" "${CCS_7B[@]}"

echo ""
echo "################################################################"
echo "  7B: batch_cg_v2 (bs=1 + eager fallback, unsorted)"
echo "################################################################"

SDIR="${BASE_DIR}/7b/batch_cg_v2"
mkdir -p "$SDIR"
launch_server "$GPU_7B" "$MODEL_7B" "$SDIR" "batch_cg_v2" $BATCH_CG_V2_ARGS
warmup "$SDIR" "$MODEL_7B"
run_sweep "$SDIR" "$MODEL_7B" "exp13_7b_batch_cg_v2" "$DATASET" "${CCS_7B[@]}"

echo ""
echo "################################################################"
echo "  7B: batch_cg_v2_sorted (bs=1 + eager fallback, sorted dataset)"
echo "################################################################"

# Reuse same server (already has batch_cg_v2 config)
SDIR="${BASE_DIR}/7b/batch_cg_v2_sorted"
mkdir -p "$SDIR"
warmup "$SDIR" "$MODEL_7B" "$SORTED_DATASET"
run_sweep "$SDIR" "$MODEL_7B" "exp13_7b_batch_cg_v2_sorted" "$SORTED_DATASET" "${CCS_7B[@]}"

echo ""
echo "################################################################"
echo "  7B: piecewise_cap512 (comparison)"
echo "################################################################"

SDIR="${BASE_DIR}/7b/piecewise_cap512"
mkdir -p "$SDIR"
launch_server "$GPU_7B" "$MODEL_7B" "$SDIR" "piecewise_cap512" \
    --enable-piecewise-cuda-graph --disable-cuda-graph --piecewise-cuda-graph-max-tokens 512
warmup "$SDIR" "$MODEL_7B"
run_sweep "$SDIR" "$MODEL_7B" "exp13_7b_piecewise_cap512" "$DATASET" "${CCS_7B[@]}"

# ═══════════════════════════════════════════════════════════════════════
#  PART 2: 32B Model (tp=4, multi-GPU)
# ═══════════════════════════════════════════════════════════════════════

CCS_32B=(4 32 64)

echo ""
echo "################################################################"
echo "  32B: vanilla (baseline, tp=4)"
echo "################################################################"

SDIR="${BASE_DIR}/32b/vanilla"
mkdir -p "$SDIR"
launch_server "$GPU_32B" "$MODEL_32B" "$SDIR" "vanilla" --tp 4
warmup "$SDIR" "$MODEL_32B"
run_sweep "$SDIR" "$MODEL_32B" "exp13_32b_vanilla" "$DATASET" "${CCS_32B[@]}"

echo ""
echo "################################################################"
echo "  32B: batch_cg_v2 (bs=1 + eager fallback, tp=4, unsorted)"
echo "################################################################"

SDIR="${BASE_DIR}/32b/batch_cg_v2"
mkdir -p "$SDIR"
launch_server "$GPU_32B" "$MODEL_32B" "$SDIR" "batch_cg_v2" --tp 4 $BATCH_CG_V2_ARGS
warmup "$SDIR" "$MODEL_32B"
run_sweep "$SDIR" "$MODEL_32B" "exp13_32b_batch_cg_v2" "$DATASET" "${CCS_32B[@]}"

echo ""
echo "################################################################"
echo "  32B: batch_cg_v2_sorted (bs=1 + eager fallback, tp=4, sorted)"
echo "################################################################"

# Reuse same server (already has batch_cg_v2 config)
SDIR="${BASE_DIR}/32b/batch_cg_v2_sorted"
mkdir -p "$SDIR"
warmup "$SDIR" "$MODEL_32B" "$SORTED_DATASET"
run_sweep "$SDIR" "$MODEL_32B" "exp13_32b_batch_cg_v2_sorted" "$SORTED_DATASET" "${CCS_32B[@]}"

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
exp8 = '${EXP8_DIR}'

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def load_exp13(model_label, setting, cc):
    return load(os.path.join(base, model_label, setting, f'cc{cc}', 'summary.json'))

def load_exp8(model_label, setting, cc):
    return load(os.path.join(exp8, f'{model_label}_{setting}', f'cc{cc}', 'summary.json'))

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

# ═══════════════════════════════════════════════════════════════════
# 7B Results
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  7B MODEL RESULTS')
print('=' * 100)

settings_7b = ['vanilla', 'batch_cg_v2', 'batch_cg_v2_sorted', 'piecewise_cap512']
ccs_7b = [4, 8, 32, 64]

# RPS table
print()
print('  --- RPS (requests/sec) ---')
hdr = f'{\"Setting\":<24s}'
for cc in ccs_7b:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

# Exp8 vanilla baseline
row = f'{\"exp8_vanilla\":<24s}'
for cc in ccs_7b:
    d = load_exp8('7b', 'vanilla', cc)
    row += f'  {fmt(rps(d))}'
print(f'  {row}')

print(f'  {\"-\" * len(hdr)}')

for s in settings_7b:
    row = f'{s:<24s}'
    for cc in ccs_7b:
        d = load_exp13('7b', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs exp13 vanilla
print()
print('  --- Speedup vs Exp13 vanilla ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_7b:
    row = f'{s:<24s}'
    for cc in ccs_7b:
        d_v = load_exp13('7b', 'vanilla', cc)
        d_s = load_exp13('7b', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# Speedup vs exp8 vanilla
print()
print('  --- Speedup vs Exp8 vanilla ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_7b:
    row = f'{s:<24s}'
    for cc in ccs_7b:
        d_v = load_exp8('7b', 'vanilla', cc)
        d_s = load_exp13('7b', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# Latency table
print()
print('  --- Latency (avg ms) ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_7b:
    row = f'{s:<24s}'
    for cc in ccs_7b:
        d = load_exp13('7b', s, cc)
        row += f'  {fmt(latency(d))}'
    print(f'  {row}')

# ═══════════════════════════════════════════════════════════════════
# 32B Results
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  32B MODEL RESULTS (tp=4)')
print('=' * 100)

settings_32b = ['vanilla', 'batch_cg_v2', 'batch_cg_v2_sorted']
ccs_32b = [4, 32, 64]

# RPS table
print()
print('  --- RPS (requests/sec) ---')
hdr = f'{\"Setting\":<24s}'
for cc in ccs_32b:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

# Exp8 vanilla baseline (tp=1)
row = f'{\"exp8_vanilla_tp1\":<24s}'
for cc in ccs_32b:
    d = load_exp8('32b', 'vanilla', cc)
    row += f'  {fmt(rps(d))}'
print(f'  {row}')

print(f'  {\"-\" * len(hdr)}')

for s in settings_32b:
    row = f'{s:<24s}'
    for cc in ccs_32b:
        d = load_exp13('32b', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs exp13 vanilla (tp=4)
print()
print('  --- Speedup vs Exp13 vanilla (tp=4) ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_32b:
    row = f'{s:<24s}'
    for cc in ccs_32b:
        d_v = load_exp13('32b', 'vanilla', cc)
        d_s = load_exp13('32b', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# Latency table
print()
print('  --- Latency (avg ms) ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')
for s in settings_32b:
    row = f'{s:<24s}'
    for cc in ccs_32b:
        d = load_exp13('32b', s, cc)
        row += f'  {fmt(latency(d))}'
    print(f'  {row}')

# ═══════════════════════════════════════════════════════════════════
# Fail check
# ═══════════════════════════════════════════════════════════════════
print()
print('=== Fail Count Check ===')
all_ok = True
for mlabel, settings, ccs in [
    ('7b', settings_7b, ccs_7b),
    ('32b', settings_32b, ccs_32b),
]:
    for s in settings:
        for cc in ccs:
            d = load_exp13(mlabel, s, cc)
            if d and d.get('fail', 0) > 0:
                print(f'  WARNING: {mlabel}/{s}/cc{cc} has {d[\"fail\"]} failures!')
                all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 13 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
