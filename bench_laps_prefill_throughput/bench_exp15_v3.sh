#!/bin/bash
# Experiment 15: Fix Batch Prefill CG Degradation
#
# Three cumulative changes, benchmarked per phase:
#   Phase A (Change 1): Padding ratio threshold gate (--batch-prefill-max-padding-ratio 2.0)
#   Phase B (Changes 1+2): + Enable piecewise CG alongside batch prefill (hybrid mode)
#   Phase C (Changes 1+2+3): + Larger/finer CG grids (bs=16, seq=48/96/192)
#
# Baselines reused from exp13 (no re-run):
#   vanilla:          135 / 204 / 411 / 450 RPS  (cc=4/8/32/64)
#   batch_cg_v2:      159 / 237 / 392 / 441 RPS
#   piecewise_cap512: 170 / 293 / 450 / 439 RPS
#
# Usage: bash bench_exp15_v3.sh <PHASE> [GPU_ID]
#   PHASE: A, B, C, or all
#   GPU_ID defaults to 5
#
set -euo pipefail

PHASE="${1:?Usage: bash bench_exp15_v3.sh <A|B|C|all> [GPU_ID]}"
PHASE=$(echo "$PHASE" | tr '[:lower:]' '[:upper:]')

GPU=${2:-5}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp15_${TODAY}"
EXP13_DIR="${SCRIPT_DIR}/results_exp13_2026-03-02"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1
MODEL="Qwen/Qwen2.5-7B-Instruct"
CCS=(4 8 32 64)

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

    for cc in "${CCS[@]}"; do
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

echo "Experiment 15: Fix Batch Prefill CG Degradation"
echo "  Phase:       ${PHASE}"
echo "  GPU:         ${GPU}"
echo "  Port:        ${PORT}"
echo "  Dataset:     ${DATASET}"
echo "  Requests:    ${NUM_REQUESTS}"
echo "  Exp13 dir:   ${EXP13_DIR}"
echo "  Results:     ${BASE_DIR}/"
echo ""

# Common batch_cg_v2 args (original grids from exp13)
BATCH_CG_COMMON="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph"
BATCH_CG_GRIDS_V2="--batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128"
BATCH_CG_GRIDS_V3="--batch-prefill-batch-sizes 1 2 4 8 16 --batch-prefill-seq-lengths 8 16 32 48 64 96 128 --batch-prefill-max-seq-len 128"

# ═══════════════════════════════════════════════════════════════════════
#  Phase A: Change 1 only (padding ratio threshold)
#  Code: batch_prefill_max_padding_ratio implemented, piecewise still
#         disabled (skip_piecewise guard present), original grids
# ═══════════════════════════════════════════════════════════════════════

run_phase_a() {
    echo ""
    echo "################################################################"
    echo "  Phase A: v3_pad2 (padding ratio threshold = 2.0)"
    echo "################################################################"

    SDIR="${BASE_DIR}/v3_pad2"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "v3_pad2" \
        $BATCH_CG_COMMON $BATCH_CG_GRIDS_V2 \
        --batch-prefill-max-padding-ratio 2.0
    warmup "$SDIR"
    run_sweep "$SDIR" "exp15_v3_pad2"
}

# ═══════════════════════════════════════════════════════════════════════
#  Phase B: Changes 1+2 (padding threshold + piecewise hybrid mode)
#  Code: skip_piecewise guard removed, same grids as Phase A
# ═══════════════════════════════════════════════════════════════════════

run_phase_b() {
    echo ""
    echo "################################################################"
    echo "  Phase B: v3_hybrid (+ piecewise CG enabled alongside batch prefill)"
    echo "################################################################"

    SDIR="${BASE_DIR}/v3_hybrid"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "v3_hybrid" \
        $BATCH_CG_COMMON $BATCH_CG_GRIDS_V2 \
        --batch-prefill-max-padding-ratio 2.0
    warmup "$SDIR"
    run_sweep "$SDIR" "exp15_v3_hybrid"
}

# ═══════════════════════════════════════════════════════════════════════
#  Phase C: Changes 1+2+3 (padding threshold + hybrid + larger grids)
#  Code: all changes applied, expanded grids
# ═══════════════════════════════════════════════════════════════════════

run_phase_c() {
    echo ""
    echo "################################################################"
    echo "  Phase C: v3_full (+ larger/finer CG grids)"
    echo "################################################################"

    SDIR="${BASE_DIR}/v3_full"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "v3_full" \
        $BATCH_CG_COMMON $BATCH_CG_GRIDS_V3 \
        --batch-prefill-max-padding-ratio 2.0
    warmup "$SDIR"
    run_sweep "$SDIR" "exp15_v3_full"
}

# ═══════════════════════════════════════════════════════════════════════
#  Run requested phase(s)
# ═══════════════════════════════════════════════════════════════════════

case "$PHASE" in
    A) run_phase_a ;;
    B) run_phase_b ;;
    C) run_phase_c ;;
    ALL)
        run_phase_a
        run_phase_b
        run_phase_c
        ;;
    *) echo "ERROR: Unknown phase '$PHASE'. Use A, B, C, or all."; exit 1 ;;
esac

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

def load_exp15(setting, cc):
    return load(os.path.join(base, setting, f'cc{cc}', 'summary.json'))

def load_exp13(setting, cc):
    return load(os.path.join(exp13, '7b', setting, f'cc{cc}', 'summary.json'))

def rps(d):
    return d['rps'] if d else None

def fmt(v):
    return f'{v:>8.1f}' if v else f'{\"N/A\":>8s}'

def fmt_speedup(v, base_v):
    if v and base_v and base_v > 0:
        return f'{v/base_v:>8.3f}x'
    return f'{\"N/A\":>8s}'

ccs = [4, 8, 32, 64]

# Build header
hdr = f'{\"Setting\":<24s}'
for cc in ccs:
    hdr += f'  cc={cc:<5d}'

# ─── RPS Table ───
print()
print('=' * 80)
print('  RPS (requests/sec)')
print('=' * 80)
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

# Baselines from exp13
for s in ['vanilla', 'batch_cg_v2', 'piecewise_cap512']:
    row = f'{\"exp13_\" + s:<24s}'
    for cc in ccs:
        d = load_exp13(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

print(f'  {\"-\" * len(hdr)}')

# Exp15 results
for s in ['v3_pad2', 'v3_hybrid', 'v3_full']:
    row = f'{s:<24s}'
    for cc in ccs:
        d = load_exp15(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# ─── Speedup vs exp13 vanilla ───
print()
print('  --- Speedup vs exp13 vanilla ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

for s_label, loader in [
    ('exp13_batch_cg_v2', lambda cc: load_exp13('batch_cg_v2', cc)),
    ('exp13_piecewise_512', lambda cc: load_exp13('piecewise_cap512', cc)),
    ('v3_pad2', lambda cc: load_exp15('v3_pad2', cc)),
    ('v3_hybrid', lambda cc: load_exp15('v3_hybrid', cc)),
    ('v3_full', lambda cc: load_exp15('v3_full', cc)),
]:
    row = f'{s_label:<24s}'
    for cc in ccs:
        d_v = load_exp13('vanilla', cc)
        d_s = loader(cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# ─── Speedup vs exp13 batch_cg_v2 ───
print()
print('  --- Speedup vs exp13 batch_cg_v2 ---')
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

for s_label, loader in [
    ('exp13_piecewise_512', lambda cc: load_exp13('piecewise_cap512', cc)),
    ('v3_pad2', lambda cc: load_exp15('v3_pad2', cc)),
    ('v3_hybrid', lambda cc: load_exp15('v3_hybrid', cc)),
    ('v3_full', lambda cc: load_exp15('v3_full', cc)),
]:
    row = f'{s_label:<24s}'
    for cc in ccs:
        d_v = load_exp13('batch_cg_v2', cc)
        d_s = loader(cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# ─── Fail Check ───
print()
print('=== Fail Count Check ===')
all_ok = True
for s in ['v3_pad2', 'v3_hybrid', 'v3_full']:
    for cc in ccs:
        d = load_exp15(s, cc)
        if d and d.get('fail', 0) > 0:
            print(f'  WARNING: {s}/cc{cc} has {d[\"fail\"]} failures!')
            all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary_phase_${PHASE}.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 15 PHASE ${PHASE} COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
