#!/bin/bash
# Experiment 16 Final: Vanilla vs Approach B vs Approach C
#
# Fair head-to-head comparison on same hardware, same session.
#   vanilla:     No CG optimizations (baseline)
#   queue_gate:  Approach B (batch_prefill CG + piecewise CG + queue-depth gate)
#   combined:    Approach C (batch_prefill CG + piecewise CG + bs gate + adaptive padding)
#
# Model: Qwen/Qwen2.5-7B-Instruct, TP=1
# Dataset: lmsys-chat-1m (10000 prompts)
# Concurrency: 4, 8, 32, 64
#
# Usage: bash bench_exp16_final.sh <vanilla|B|C|all> [GPU_ID]
#
set -euo pipefail

PHASE="${1:?Usage: bash bench_exp16_final.sh <vanilla|B|C|all> [GPU_ID]}"
PHASE=$(echo "$PHASE" | tr '[:lower:]' '[:upper:]')

GPU=${2:-5}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp16_final_${TODAY}"
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

echo "Experiment 16 Final: Vanilla vs Approach B vs Approach C"
echo "  Phase:       ${PHASE}"
echo "  GPU:         ${GPU}"
echo "  Port:        ${PORT}"
echo "  Model:       ${MODEL} (TP=1)"
echo "  Dataset:     ${DATASET}"
echo "  Requests:    ${NUM_REQUESTS}"
echo "  Results:     ${BASE_DIR}/"
echo ""

# Common CG args
CG_COMMON="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph"
CG_GRIDS="--batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128 --batch-prefill-max-padding-ratio 2.0"

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
#  Approach B: Queue-depth gate
#  piecewise CG skipped when num_waiting_reqs > 8
#  batch-size gate disabled (9999)
# ═══════════════════════════════════════════════════════════════════════

run_approach_b() {
    echo ""
    echo "################################################################"
    echo "  Approach B: queue_gate (--piecewise-max-waiting-reqs 8)"
    echo "################################################################"

    SDIR="${BASE_DIR}/queue_gate"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "queue_gate" \
        $CG_COMMON $CG_GRIDS \
        --piecewise-extend-max-bs 9999 \
        --piecewise-max-waiting-reqs 8
    warmup "$SDIR"
    run_sweep "$SDIR" "queue_gate"
}

# ═══════════════════════════════════════════════════════════════════════
#  Approach C: Batch-size gate + adaptive padding
#  piecewise CG skipped when batch_size > 4
#  relaxed padding ratio 4.0 for large batches
#  queue gate disabled (9999)
# ═══════════════════════════════════════════════════════════════════════

run_approach_c() {
    echo ""
    echo "################################################################"
    echo "  Approach C: combined (bs gate + adaptive padding 4.0)"
    echo "################################################################"

    SDIR="${BASE_DIR}/combined"
    mkdir -p "$SDIR"
    launch_server "$SDIR" "combined" \
        $CG_COMMON $CG_GRIDS \
        --piecewise-extend-max-bs 4 \
        --piecewise-max-waiting-reqs 9999 \
        --batch-prefill-high-cc-padding-ratio 4.0
    warmup "$SDIR"
    run_sweep "$SDIR" "combined"
}

# ═══════════════════════════════════════════════════════════════════════
#  Run requested phase(s)
# ═══════════════════════════════════════════════════════════════════════

case "$PHASE" in
    VANILLA) run_vanilla ;;
    B) run_approach_b ;;
    C) run_approach_c ;;
    ALL)
        run_vanilla
        run_approach_b
        run_approach_c
        ;;
    *) echo "ERROR: Unknown phase '$PHASE'. Use vanilla, B, C, or all."; exit 1 ;;
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
settings = ['vanilla', 'queue_gate', 'combined']
labels = {'vanilla': 'vanilla', 'queue_gate': 'B: queue_gate', 'combined': 'C: combined'}

# ─── RPS Table ───
print()
print('=' * 90)
print('  Throughput (req/s)')
print('=' * 90)
hdr = f'{\"Setting\":<20s}'
for cc in ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f'  {\"-\" * len(hdr)}')

for s in settings:
    row = f'{labels[s]:<20s}'
    for cc in ccs:
        d = load(s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# ─── Latency Table ───
for lat_key, lat_label in [('avg', 'Avg Latency'), ('p50', 'P50 Latency'), ('p90', 'P90 Latency'), ('p99', 'P99 Latency')]:
    print()
    print('=' * 90)
    print(f'  {lat_label} (ms)')
    print('=' * 90)
    print(f'  {hdr}')
    print(f'  {\"-\" * len(hdr)}')

    for s in settings:
        row = f'{labels[s]:<20s}'
        for cc in ccs:
            d = load(s, cc)
            row += f'  {fmt(lat(d, lat_key))}'
        print(f'  {row}')

# ─── Delta vs Vanilla ───
print()
print('=' * 90)
print('  RPS Delta vs Vanilla (%)')
print('=' * 90)
hdr2 = f'{\"Setting\":<20s}'
for cc in ccs:
    hdr2 += f'  cc={cc:<5d}'
print(f'  {hdr2}')
print(f'  {\"-\" * len(hdr2)}')

for s in ['queue_gate', 'combined']:
    row = f'{labels[s]:<20s}'
    for cc in ccs:
        d_v = load('vanilla', cc)
        d_s = load(s, cc)
        row += f'  {fmt_pct(rps(d_s), rps(d_v))}'
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
echo "  EXPERIMENT 16 FINAL — PHASE ${PHASE} COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
