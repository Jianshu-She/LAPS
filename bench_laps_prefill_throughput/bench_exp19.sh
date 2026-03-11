#!/bin/bash
# Experiment 19: Optimized CG Grid for Batch Prefill
#
# Compare current CG grid vs optimized grids based on dataset analysis.
#
#   current:  bs=[1,2,4,8] × seq=[8,16,32,64,128], ratio≤2.0              (20 graphs)
#   grid_L:   bs=[1,2,3,4,6,8,12,16] × seq=[4,8,12,16,20,24,32,48,64,96,128,192,256,384,512], ratio≤3.0  (120 graphs)
#   grid_G:   bs=[1,2,4,8] × seq=[4,8,12,16,20,24,32,48,64,96,128], ratio≤2.0        (44 graphs, fine seq only)
#
# Model: Qwen/Qwen2.5-7B-Instruct, TP=1
# Dataset: lmsys-chat-1m (10000 prompts)
# Concurrency: 4, 8, 32, 64
#
# Usage: bash bench_exp19.sh <vanilla|current|G|L|all> [GPU_ID]
#
set -euo pipefail

PHASE="${1:?Usage: bash bench_exp19.sh <vanilla|current|G|L|all> [GPU_ID]}"
PHASE=$(echo "$PHASE" | tr '[:lower:]' '[:upper:]')

GPU=${2:-5}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp19_${TODAY}"
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

echo "Experiment 19: Optimized CG Grid for Batch Prefill"
echo "  Phase:       ${PHASE}"
echo "  GPU:         ${GPU}"
echo "  Port:        ${PORT}"
echo "  Model:       ${MODEL} (TP=1)"
echo "  Dataset:     ${DATASET}"
echo "  Requests:    ${NUM_REQUESTS}"
echo "  Results:     ${BASE_DIR}/"
echo ""

# Common CG args (without grid-specific settings)
CG_COMMON="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph"

# ═══════════════════════════════════════════════════════════════════════
#  Vanilla: No CG optimizations
# ═══════════════════════════════════════════════════════════════════════

run_vanilla() {
    echo ""
    echo "################################################################"
    echo "  Vanilla (no CG optimizations)"
    echo "################################################################"

    SDIR="${BASE_DIR}/vanilla"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "vanilla"
    warmup "$SDIR"
    run_sweep "$SDIR" "vanilla"
}

# ═══════════════════════════════════════════════════════════════════════
#  Current grid: bs=[1,2,4,8] × seq=[8,16,32,64,128], ratio≤2.0
#  20 graphs
# ═══════════════════════════════════════════════════════════════════════

run_current() {
    echo ""
    echo "################################################################"
    echo "  Current grid: 20 graphs (power-of-2, max_seq=128)"
    echo "################################################################"

    SDIR="${BASE_DIR}/current"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "current" \
        $CG_COMMON \
        --batch-prefill-batch-sizes 1 2 4 8 \
        --batch-prefill-seq-lengths 8 16 32 64 128 \
        --batch-prefill-max-seq-len 128 \
        --batch-prefill-max-padding-ratio 2.0 \
        --piecewise-extend-max-bs 9999 \
        --piecewise-max-waiting-reqs 8
    warmup "$SDIR"
    run_sweep "$SDIR" "current"
}

# ═══════════════════════════════════════════════════════════════════════
#  Grid G: Fine seq grid, same batch sizes
#  bs=[1,2,4,8] × seq=[4,8,12,16,20,24,32,48,64,96,128], ratio≤2.0
#  44 graphs — tests the impact of finer seq granularity alone
# ═══════════════════════════════════════════════════════════════════════

run_grid_g() {
    echo ""
    echo "################################################################"
    echo "  Grid G: 44 graphs (fine seq, bs≤8, max_seq=128)"
    echo "################################################################"

    SDIR="${BASE_DIR}/grid_g"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "grid_g" \
        $CG_COMMON \
        --batch-prefill-batch-sizes 1 2 4 8 \
        --batch-prefill-seq-lengths 4 8 12 16 20 24 32 48 64 96 128 \
        --batch-prefill-max-seq-len 128 \
        --batch-prefill-max-padding-ratio 2.0 \
        --piecewise-extend-max-bs 9999 \
        --piecewise-max-waiting-reqs 8
    warmup "$SDIR"
    run_sweep "$SDIR" "grid_g"
}

# ═══════════════════════════════════════════════════════════════════════
#  Grid L: Dataset-optimized
#  bs=[1,2,3,4,6,8,12,16] × seq=[4,8,12,16,20,24,32,48,64,96,128,192,256,384,512]
#  ratio≤3.0, 120 graphs
# ═══════════════════════════════════════════════════════════════════════

run_grid_l() {
    echo ""
    echo "################################################################"
    echo "  Grid L: 120 graphs (dataset-optimized, max_seq=512, ratio≤3.0)"
    echo "################################################################"

    SDIR="${BASE_DIR}/grid_l"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "grid_l" \
        $CG_COMMON \
        --batch-prefill-batch-sizes 1 2 3 4 6 8 12 16 \
        --batch-prefill-seq-lengths 4 8 12 16 20 24 32 48 64 96 128 192 256 384 512 \
        --batch-prefill-max-seq-len 512 \
        --batch-prefill-max-padding-ratio 3.0 \
        --piecewise-extend-max-bs 9999 \
        --piecewise-max-waiting-reqs 8
    warmup "$SDIR"
    run_sweep "$SDIR" "grid_l"
}

# ═══════════════════════════════════════════════════════════════════════
#  Run requested phase(s)
# ═══════════════════════════════════════════════════════════════════════

case "$PHASE" in
    VANILLA) run_vanilla ;;
    CURRENT) run_current ;;
    G) run_grid_g ;;
    L) run_grid_l ;;
    ALL)
        run_vanilla
        run_current
        run_grid_g
        run_grid_l
        ;;
    *) echo "ERROR: Unknown phase '$PHASE'. Use vanilla, current, G, L, or all."; exit 1 ;;
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

def load(setting, cc):
    try:
        with open(os.path.join(base, setting, f'cc{cc}', 'summary.json')) as f:
            return json.load(f)
    except:
        return None

def rps(d):
    return d['rps'] if d else None

def lat(d, key):
    return d['latency_ms'][key] if d else None

def fmt(v):
    return f'{v:>8.1f}' if v else f'{\"N/A\":>8s}'

def fmt_pct(v, base_v):
    if v and base_v and base_v > 0:
        delta = (v / base_v - 1) * 100
        sign = '+' if delta >= 0 else ''
        return f'{sign}{delta:>6.1f}%'
    return f'{\"N/A\":>7s}'

ccs = [4, 8, 32, 64]
settings = ['vanilla', 'current', 'grid_g', 'grid_l']
labels = {'vanilla': 'vanilla', 'current': 'current (20 graphs)', 'grid_g': 'G: fine_seq (44)', 'grid_l': 'L: optimized (120)'}

# ─── RPS Table ───
print()
print('=' * 100)
print('  Throughput (req/s)')
print('=' * 100)
hdr = f'{\"Setting\":<28s}'
for cc in ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

for s in settings:
    row = f'{labels[s]:<28s}'
    for cc in ccs:
        d = load(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# ─── Latency Tables ───
for lat_key, lat_label in [('avg', 'Avg Latency'), ('p50', 'P50 Latency'), ('p99', 'P99 Latency')]:
    print()
    print('=' * 100)
    print(f'  {lat_label} (ms)')
    print('=' * 100)
    print(f'  {hdr}')
    print(f'  {\"-\" * len(hdr)}')

    for s in settings:
        row = f'{labels[s]:<28s}'
        for cc in ccs:
            d = load(s, cc)
            row += f'  {fmt(lat(d, lat_key))}'
        print(f'  {row}')

# ─── Delta vs Vanilla ───
print()
print('=' * 100)
print('  RPS Delta vs Vanilla (%)')
print('=' * 100)
hdr2 = f'{\"Setting\":<28s}'
for cc in ccs:
    hdr2 += f'  cc={cc:<5d}'
print(f'  {hdr2}')
print(f'  {\"-\" * len(hdr2)}')

for s in ['current', 'grid_g', 'grid_l']:
    row = f'{labels[s]:<28s}'
    for cc in ccs:
        d_v = load('vanilla', cc)
        d_s = load(s, cc)
        row += f'  {fmt_pct(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# ─── Delta vs Current ───
print()
print('=' * 100)
print('  RPS Delta vs Current Grid (%)')
print('=' * 100)
print(f'  {hdr2}')
print(f'  {\"-\" * len(hdr2)}')

for s in ['grid_g', 'grid_l']:
    row = f'{labels[s]:<28s}'
    for cc in ccs:
        d_c = load('current', cc)
        d_s = load(s, cc)
        row += f'  {fmt_pct(rps(d_s), rps(d_c))}'
    print(f'  {row}')

# ─── Fail Check ───
print()
print('=== Fail Count Check ===')
all_ok = True
for s in settings:
    for cc in ccs:
        d = load(s, cc)
        if d and d.get('fail', 0) > 0:
            print(f'  WARNING: {labels[s]}/cc{cc} has {d[\"fail\"]} failures!')
            all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary_${PHASE}.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 19 — PHASE ${PHASE} COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
