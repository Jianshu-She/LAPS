#!/bin/bash
# Experiment 5: Single-GPU No-Disagg Comparison
#
# Goal: Remove PD disagg overhead entirely. Single sglang server on one GPU,
#       max_new_tokens=1. Pure prefill compute comparison.
#
# 4 server configs:
#   1) vanilla          — no CUDA graph at all
#   2) piecewise_cg     — piecewise CUDA graph only
#   3) batch_prefill_cg — BP-CG with standard grid (6×10=60 graphs)
#   4) batch_prefill_cg_fine — BP-CG with finer grid (8×18=144 graphs)
#
# Datasets:
#   - Uniform (895 prompts, 40-80 tokens)  × cc=1,2,4,8,16,32
#   - Mixed 10K (variable lengths)          × cc=8,32
#
# Usage: bash bench_debug_nodisagg.sh [GPU_ID]
#   GPU_ID defaults to 4
#
set -euo pipefail

GPU=${1:-4}
MODEL="Qwen/Qwen2.5-32B"
PORT=30300
HOST="127.0.0.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

LMSYS_10K="/mnt/weka/home/jianshu.she/mlsys_ae/LAPS/data/lmsys_chat_10k.jsonl"
UNIFORM_DATASET="${SCRIPT_DIR}/results_debug_2026-03-02/exp2_uniform/lmsys_chat_uniform.jsonl"

TODAY=$(date +%Y-%m-%d)
BASE_DIR="${SCRIPT_DIR}/results_nodisagg_${TODAY}"
mkdir -p "$BASE_DIR"

MAX_NEW_TOKENS=1

# ── Server argument presets ──

declare -A SERVER_ARGS
SERVER_ARGS[vanilla]=""
SERVER_ARGS[piecewise_cg]="--enable-piecewise-cuda-graph --disable-cuda-graph"
SERVER_ARGS[batch_prefill_cg]="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512"
SERVER_ARGS[batch_prefill_cg_fine]="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 3 4 5 6 7 8 --batch-prefill-seq-lengths 1 2 3 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 --batch-prefill-max-seq-len 512"
SERVER_ARGS[batch_prefill_cg_small]="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 --batch-prefill-seq-lengths 4 8 16 --batch-prefill-max-seq-len 16"

SETTINGS=(vanilla piecewise_cg batch_prefill_cg batch_prefill_cg_fine batch_prefill_cg_small)

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
    echo "  Launching single server: ${label}"
    echo "  GPU: ${GPU}"
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

run_bench() {
    local label=$1
    local dataset=$2
    local num_prompts=$3
    local cc=$4
    local run_dir=$5

    flush_cache

    echo "  --- ${label} | cc=${cc} | N=${num_prompts} ---"
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$dataset" \
        --url "http://${HOST}:${PORT}" \
        --num-prompts $num_prompts \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency $cc \
        --seed 42 \
        --output "${run_dir}/${label}_cc${cc}.json" \
        2>&1 | tee "${run_dir}/${label}_cc${cc}.txt"
    sleep 1
}

trap cleanup EXIT

# ── Validate datasets ──
if [ ! -f "$UNIFORM_DATASET" ]; then
    echo "ERROR: Uniform dataset not found: $UNIFORM_DATASET"
    echo "Run create_uniform_dataset.py first or adjust path."
    exit 1
fi
if [ ! -f "$LMSYS_10K" ]; then
    echo "ERROR: 10K dataset not found: $LMSYS_10K"
    exit 1
fi

UNIFORM_COUNT=$(wc -l < "$UNIFORM_DATASET")
echo "Uniform dataset: ${UNIFORM_COUNT} prompts"
echo "10K dataset: ${LMSYS_10K}"
echo "GPU: ${GPU}"
echo "Results: ${BASE_DIR}/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# RUN ALL SETTINGS
# ═══════════════════════════════════════════════════════════════════════

for setting in "${SETTINGS[@]}"; do
    SETTING_DIR="${BASE_DIR}/${setting}"
    mkdir -p "$SETTING_DIR"

    launch_server "$setting" "${SERVER_ARGS[$setting]}" "$SETTING_DIR"

    # Warmup
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$LMSYS_10K" \
        --url "http://${HOST}:${PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3

    # Uniform dataset: cc=1,2,4,8,16,32
    echo ""
    echo "  ── Uniform dataset (${UNIFORM_COUNT} prompts) ──"
    for cc in 1 2 4 8 16 32; do
        run_bench "${setting}_uniform" "$UNIFORM_DATASET" $UNIFORM_COUNT $cc "$SETTING_DIR"
    done

    # 10K dataset: cc=8,32
    echo ""
    echo "  ── 10K dataset ──"
    for cc in 8 32; do
        run_bench "${setting}_10k" "$LMSYS_10K" 10000 $cc "$SETTING_DIR"
    done
done

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY — Experiment 5: No-Disagg Comparison"
echo "################################################################"

$PYTHON -c "
import json, os, sys

base_dir = '${BASE_DIR}'
settings = ['vanilla', 'piecewise_cg', 'batch_prefill_cg', 'batch_prefill_cg_fine', 'batch_prefill_cg_small']

# ── Uniform dataset ──
print()
print('=== Uniform Dataset (40-80 tokens) ===')
print()
header = f'{\"Setting\":<25s}'
ccs_uniform = [1, 2, 4, 8, 16, 32]
for cc in ccs_uniform:
    header += f'  cc={cc:<4d}'
print(header)
print('-' * len(header))

for s in settings:
    row = f'{s:<25s}'
    for cc in ccs_uniform:
        f = os.path.join(base_dir, s, f'{s}_uniform_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            row += f'  {tput:>7.0f}'
        else:
            row += f'  {\"N/A\":>7s}'
    print(row)

# ── 10K dataset ──
print()
print('=== 10K Dataset (variable lengths) ===')
print()
header = f'{\"Setting\":<25s}'
ccs_10k = [8, 32]
for cc in ccs_10k:
    header += f'  cc={cc:<4d}'
print(header)
print('-' * len(header))

for s in settings:
    row = f'{s:<25s}'
    for cc in ccs_10k:
        f = os.path.join(base_dir, s, f'{s}_10k_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            row += f'  {tput:>7.0f}'
        else:
            row += f'  {\"N/A\":>7s}'
    print(row)

# ── Speedup vs piecewise ──
print()
print('=== Speedup vs piecewise_cg (uniform) ===')
print()
header = f'{\"Setting\":<25s}'
for cc in ccs_uniform:
    header += f'  cc={cc:<4d}'
print(header)
print('-' * len(header))

# Load piecewise baseline
pw_tput = {}
for cc in ccs_uniform:
    f = os.path.join(base_dir, 'piecewise_cg', f'piecewise_cg_uniform_cc{cc}.json')
    if os.path.exists(f):
        with open(f) as fh:
            pw_tput[cc] = json.load(fh).get('prefill_throughput_tok_s', 0)

for s in settings:
    row = f'{s:<25s}'
    for cc in ccs_uniform:
        f = os.path.join(base_dir, s, f'{s}_uniform_cc{cc}.json')
        if os.path.exists(f) and cc in pw_tput and pw_tput[cc] > 0:
            with open(f) as fh:
                tput = json.load(fh).get('prefill_throughput_tok_s', 0)
            speedup = tput / pw_tput[cc]
            row += f'  {speedup:>6.2f}x'
        else:
            row += f'  {\"N/A\":>7s}'
    print(row)

# ── Padding analysis from logs ──
import re
print()
print('=== BP-CG Padding Analysis (from server logs) ===')
print()

replay_pat = re.compile(
    r'BP-CG-REPLAY actual_bs=(\d+) target_bs=(\d+) '
    r'max_extend_len=(\d+) target_seq_len=(\d+) '
    r'real_tokens=(\d+) padded_tokens=(\d+) '
    r'ratio=([\d.]+)'
)
path_pat = re.compile(r'FWD-PATH path=(\S+) bs=(\d+)')

for s in ['batch_prefill_cg', 'batch_prefill_cg_fine', 'batch_prefill_cg_small']:
    log_file = os.path.join(base_dir, s, f'{s}_server.log')
    if not os.path.exists(log_file):
        print(f'  {s}: log not found')
        continue

    replays = []
    path_counts = {}
    with open(log_file) as f:
        for line in f:
            m = replay_pat.search(line)
            if m:
                replays.append({
                    'real_tokens': int(m.group(5)),
                    'padded_tokens': int(m.group(6)),
                    'ratio': float(m.group(7)),
                })
            m = path_pat.search(line)
            if m:
                path = m.group(1)
                path_counts[path] = path_counts.get(path, 0) + 1

    if replays:
        ratios = [r['ratio'] for r in replays]
        real_total = sum(r['real_tokens'] for r in replays)
        padded_total = sum(r['padded_tokens'] for r in replays)
        ratios_sorted = sorted(ratios)
        print(f'  {s}:')
        print(f'    BP-CG replays:   {len(replays)}')
        print(f'    Mean ratio:      {sum(ratios)/len(ratios):.2f}x')
        print(f'    Median ratio:    {ratios_sorted[len(ratios)//2]:.2f}x')
        print(f'    Overall ratio:   {padded_total/max(real_total,1):.2f}x')
        print(f'    Total real:      {real_total:,}')
        print(f'    Total padded:    {padded_total:,}')
    else:
        print(f'  {s}: no BP-CG-REPLAY events')

    if path_counts:
        total_fwd = sum(path_counts.values())
        print(f'    Fwd paths ({total_fwd} total):')
        for path, cnt in sorted(path_counts.items(), key=lambda x: -x[1]):
            print(f'      {path:<20s} {cnt:>6d}  ({100*cnt/total_fwd:.1f}%)')
    print()

print()
print('Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 5 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
