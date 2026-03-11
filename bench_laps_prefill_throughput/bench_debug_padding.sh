#!/bin/bash
# Debug experiments to test batch prefill CUDA graph hypotheses
#
# Experiment 1: Instrumented run (10K prompts, batch_prefill_cg, cc=8,32)
#               → logs padding ratio per forward pass
# Experiment 2: Uniform-length dataset (2K prompts, batch_prefill_cg + piecewise, cc=8,32)
#               → removes variable-length padding as a variable
# Experiment 3: 500 prompts with cache flush between cc levels
#               → tests if radix cache explains 500-prompt speed
# Experiment 4: 10K prompts with cache flush between cc levels
#               → cross-check: does flush matter for 10K?
#
# Usage: bash bench_debug_padding.sh
#
# Assumes GPUs 4,5 are available. Change PREFILL_GPU/DECODE_GPU if needed.
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
LMSYS_10K="/mnt/weka/home/jianshu.she/mlsys_ae/LAPS/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
PREFILL_GPU=4
DECODE_GPU=5
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

TODAY=$(date +%Y-%m-%d)
BASE_RESULTS_DIR="${SCRIPT_DIR}/results_debug_${TODAY}"
mkdir -p "$BASE_RESULTS_DIR"

MAX_NEW_TOKENS=1

# Common server args
PREFILL_CG_OPTS="--disable-cuda-graph"
BATCH_PREFILL_OPTS="--batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512"
BATCH_PREFILL_CG_ARGS="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $PREFILL_CG_OPTS $BATCH_PREFILL_OPTS"
PIECEWISE_CG_ARGS="--enable-piecewise-cuda-graph $PREFILL_CG_OPTS"

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

flush_cache() {
    echo "[flush] Flushing KV cache on all servers..."
    curl -s -X POST "http://${HOST}:${PREFILL_PORT}/flush_cache" > /dev/null 2>&1 || true
    curl -s -X POST "http://${HOST}:${DECODE_PORT}/flush_cache" > /dev/null 2>&1 || true
    sleep 2
}

launch_servers() {
    local label=$1
    local prefill_extra_args=$2
    local log_dir=$3

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Prefill GPU: ${PREFILL_GPU}, Decode GPU: ${DECODE_GPU}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    # ── launch decode ──
    echo "[launch] Decode server on GPU ${DECODE_GPU}..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${log_dir}/${label}_decode.log" 2>&1 &

    # ── launch prefill ──
    echo "[launch] Prefill server on GPU ${PREFILL_GPU} with: ${prefill_extra_args:-<none>}..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${log_dir}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  600

    # ── launch router ──
    echo "[launch] Router..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${log_dir}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    # ── warmup ──
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$LMSYS_10K" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3
}

run_bench() {
    local label=$1
    local dataset=$2
    local num_prompts=$3
    local cc=$4
    local run_dir=$5
    local seed=${6:-42}
    local do_flush=${7:-false}

    if [ "$do_flush" = "true" ]; then
        flush_cache
    fi

    echo "  --- ${label} | cc=${cc} | N=${num_prompts} | flush=${do_flush} ---"
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$dataset" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts $num_prompts \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency $cc \
        --seed $seed \
        --output "${run_dir}/${label}_cc${cc}.json" \
        2>&1 | tee "${run_dir}/${label}_cc${cc}.txt"
    sleep 2
}

trap cleanup EXIT

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Instrumented run — 10K prompts, batch_prefill_cg
#   cc=8 and cc=32 only. Padding ratio logged in prefill server log.
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 1: Instrumented 10K (batch_prefill_cg, cc=8,32)"
echo "################################################################"

EXP1_DIR="${BASE_RESULTS_DIR}/exp1_instrumented_10k"
mkdir -p "$EXP1_DIR"

launch_servers "batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP1_DIR"

for cc in 8 32; do
    run_bench "batch_prefill_cg" "$LMSYS_10K" 10000 $cc "$EXP1_DIR" 42 false
done

# Copy the prefill log (contains BP-CG-REPLAY lines)
echo "[exp1] Prefill log saved to: ${EXP1_DIR}/batch_prefill_cg_prefill.log"

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Uniform-length dataset — 2K prompts, cc=8,32
#   batch_prefill_cg vs piecewise_cg
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 2: Uniform-length dataset (40-80 tokens)"
echo "################################################################"

EXP2_DIR="${BASE_RESULTS_DIR}/exp2_uniform"
UNIFORM_DATASET="${EXP2_DIR}/lmsys_chat_uniform.jsonl"
mkdir -p "$EXP2_DIR"

# Create uniform dataset
echo "[exp2] Creating uniform dataset..."
$PYTHON "${SCRIPT_DIR}/create_uniform_dataset.py" \
    --input "$LMSYS_10K" \
    --output "$UNIFORM_DATASET" \
    --min-tokens 40 --max-tokens 80 --max-prompts 2000

UNIFORM_COUNT=$(wc -l < "$UNIFORM_DATASET")
echo "[exp2] Uniform dataset: ${UNIFORM_COUNT} prompts"

# 2a: batch_prefill_cg (servers already running from exp1)
echo ""
echo "  --- Exp 2a: batch_prefill_cg with uniform dataset ---"
for cc in 8 32; do
    run_bench "uniform_batch_prefill_cg" "$UNIFORM_DATASET" $UNIFORM_COUNT $cc "$EXP2_DIR" 42 true
done

# 2b: piecewise_cg — need to restart servers
echo ""
echo "  --- Exp 2b: piecewise_cg with uniform dataset ---"
launch_servers "piecewise_cg" "$PIECEWISE_CG_ARGS" "$EXP2_DIR"

for cc in 8 32; do
    run_bench "uniform_piecewise_cg" "$UNIFORM_DATASET" $UNIFORM_COUNT $cc "$EXP2_DIR" 42 true
done

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 500 prompts with cache flush — all cc levels
#   batch_prefill_cg vs vanilla_sglang
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 3: 500 prompts with flush (test cache hypothesis)"
echo "################################################################"

EXP3_DIR="${BASE_RESULTS_DIR}/exp3_500_flush"
mkdir -p "$EXP3_DIR"

# 3a: batch_prefill_cg
echo ""
echo "  --- Exp 3a: batch_prefill_cg, 500 prompts, flush ---"
launch_servers "batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP3_DIR"

for cc in 1 2 4 8 16 32 64 128; do
    run_bench "flush_batch_prefill_cg" "$LMSYS_10K" 500 $cc "$EXP3_DIR" 42 true
done

# 3b: vanilla_sglang (no CG at all)
echo ""
echo "  --- Exp 3b: vanilla_sglang, 500 prompts, flush ---"
launch_servers "vanilla_sglang" "" "$EXP3_DIR"

for cc in 1 2 4 8 16 32 64 128; do
    run_bench "flush_vanilla_sglang" "$LMSYS_10K" 500 $cc "$EXP3_DIR" 42 true
done

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: 10K prompts with cache flush — cc=8,128
#   batch_prefill_cg vs vanilla_sglang
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 4: 10K prompts with flush (cross-check)"
echo "################################################################"

EXP4_DIR="${BASE_RESULTS_DIR}/exp4_10k_flush"
mkdir -p "$EXP4_DIR"

# 4a: vanilla_sglang (servers already running from exp3b)
echo ""
echo "  --- Exp 4a: vanilla_sglang, 10K prompts, flush ---"
for cc in 8 128; do
    run_bench "flush_vanilla_sglang" "$LMSYS_10K" 10000 $cc "$EXP4_DIR" 42 true
done

# 4b: batch_prefill_cg
echo ""
echo "  --- Exp 4b: batch_prefill_cg, 10K prompts, flush ---"
launch_servers "batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP4_DIR"

for cc in 8 128; do
    run_bench "flush_batch_prefill_cg" "$LMSYS_10K" 10000 $cc "$EXP4_DIR" 42 true
done

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  ANALYSIS"
echo "################################################################"

# Experiment 1 analysis: parse BP-CG-REPLAY lines from prefill log
echo ""
echo "=== Experiment 1: Padding ratio analysis ==="
$PYTHON -c "
import re, sys

log_file = '${EXP1_DIR}/batch_prefill_cg_prefill.log'
replay_pat = re.compile(
    r'BP-CG-REPLAY actual_bs=(\d+) target_bs=(\d+) '
    r'max_extend_len=(\d+) target_seq_len=(\d+) '
    r'real_tokens=(\d+) padded_tokens=(\d+) '
    r'ratio=([\d.]+)'
)
path_pat = re.compile(r'FWD-PATH path=(\S+) bs=(\d+)')

replays = []
path_counts = {}

try:
    with open(log_file) as f:
        for line in f:
            m = replay_pat.search(line)
            if m:
                replays.append({
                    'actual_bs': int(m.group(1)),
                    'target_bs': int(m.group(2)),
                    'max_extend_len': int(m.group(3)),
                    'target_seq_len': int(m.group(4)),
                    'real_tokens': int(m.group(5)),
                    'padded_tokens': int(m.group(6)),
                    'ratio': float(m.group(7)),
                })
            m = path_pat.search(line)
            if m:
                path = m.group(1)
                path_counts[path] = path_counts.get(path, 0) + 1
except FileNotFoundError:
    print(f'WARNING: Log file not found: {log_file}')
    sys.exit(0)

if replays:
    ratios = [r['ratio'] for r in replays]
    real_total = sum(r['real_tokens'] for r in replays)
    padded_total = sum(r['padded_tokens'] for r in replays)
    ratios_sorted = sorted(ratios)

    print(f'  BP-CG-REPLAY events: {len(replays)}')
    print(f'  Mean padding ratio:  {sum(ratios)/len(ratios):.2f}x')
    print(f'  Median ratio:        {ratios_sorted[len(ratios)//2]:.2f}x')
    print(f'  P99 ratio:           {ratios_sorted[int(len(ratios)*0.99)]:.2f}x')
    print(f'  Max ratio:           {max(ratios):.2f}x')
    print(f'  Total real tokens:   {real_total:,}')
    print(f'  Total padded tokens: {padded_total:,}')
    print(f'  Overall ratio:       {padded_total/max(real_total,1):.2f}x')
else:
    print('  No BP-CG-REPLAY events found in log.')

print()
if path_counts:
    total_fwd = sum(path_counts.values())
    print(f'  Forward path distribution ({total_fwd} total):')
    for path, cnt in sorted(path_counts.items(), key=lambda x: -x[1]):
        print(f'    {path:<20s} {cnt:>6d}  ({100*cnt/total_fwd:.1f}%)')
else:
    print('  No FWD-PATH events found in log.')
" 2>&1 | tee "${EXP1_DIR}/padding_analysis.txt"

# Experiment 2 analysis
echo ""
echo "=== Experiment 2: Uniform dataset comparison ==="
$PYTHON -c "
import json, os

exp_dir = '${EXP2_DIR}'
for cc in [8, 32]:
    print(f'  cc={cc}:')
    for label in ['uniform_batch_prefill_cg', 'uniform_piecewise_cg']:
        f = os.path.join(exp_dir, f'{label}_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            print(f'    {label:<35s} {tput:>8.0f} tok/s')
        else:
            print(f'    {label:<35s} N/A')
" 2>&1 | tee "${EXP2_DIR}/comparison.txt"

# Experiment 3 analysis
echo ""
echo "=== Experiment 3: 500 prompts flush vs no-flush ==="
$PYTHON -c "
import json, os

exp_dir = '${EXP3_DIR}'
ccs = [1, 2, 4, 8, 16, 32, 64, 128]
labels = ['flush_batch_prefill_cg', 'flush_vanilla_sglang']

for label in labels:
    print(f'  {label}:')
    for cc in ccs:
        f = os.path.join(exp_dir, f'{label}_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            print(f'    cc={cc:<4d} → {tput:>8.0f} tok/s')
        else:
            print(f'    cc={cc:<4d} → N/A')
    print()
" 2>&1 | tee "${EXP3_DIR}/comparison.txt"

# Experiment 4 analysis
echo ""
echo "=== Experiment 4: 10K flush cross-check ==="
$PYTHON -c "
import json, os

exp_dir = '${EXP4_DIR}'
for cc in [8, 128]:
    print(f'  cc={cc}:')
    for label in ['flush_vanilla_sglang', 'flush_batch_prefill_cg']:
        f = os.path.join(exp_dir, f'{label}_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            print(f'    {label:<35s} {tput:>8.0f} tok/s')
        else:
            print(f'    {label:<35s} N/A')
" 2>&1 | tee "${EXP4_DIR}/comparison.txt"

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: ${BASE_RESULTS_DIR}/"
echo "  Exp 1 (padding ratio):    ${EXP1_DIR}/"
echo "  Exp 2 (uniform dataset):  ${EXP2_DIR}/"
echo "  Exp 3 (500 flush):        ${EXP3_DIR}/"
echo "  Exp 4 (10K flush):        ${EXP4_DIR}/"
echo "================================================================"
