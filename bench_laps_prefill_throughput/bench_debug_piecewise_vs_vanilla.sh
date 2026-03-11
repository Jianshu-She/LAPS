#!/bin/bash
# Experiment 6: Piecewise CG vs Vanilla — detailed comparison
#
# Test 1: Very short prompts (8, 16 tokens) with low cc (1,2,3,4)
#         → Does piecewise help with tiny prefills?
# Test 2: 1000 prompts from 10K dataset, flush before each run
#         → Fair comparison without radix cache inflation
#
# Single GPU, no disagg, no router.
#
# Usage: bash bench_debug_piecewise_vs_vanilla.sh [GPU_ID]
set -euo pipefail

GPU=${1:-4}
MODEL="Qwen/Qwen2.5-32B"
PORT=30300
HOST="127.0.0.1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

LMSYS_10K="/mnt/weka/home/jianshu.she/mlsys_ae/LAPS/data/lmsys_chat_10k.jsonl"

TODAY=$(date +%Y-%m-%d)
BASE_DIR="${SCRIPT_DIR}/results_piecewise_vs_vanilla_${TODAY}"
mkdir -p "$BASE_DIR"

MAX_NEW_TOKENS=1

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
    echo "  GPU: ${GPU}, Args: ${extra_args:-<none>}"
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
    local do_flush=${6:-true}

    if [ "$do_flush" = "true" ]; then
        flush_cache
    fi

    echo "  --- ${label} | cc=${cc} | N=${num_prompts} | flush=${do_flush} ---"
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

# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Very short prompts
# Create datasets with ~8 tokens and ~16 tokens
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  TEST 1: Very short prompts (8 and 16 tokens)"
echo "################################################################"

TEST1_DIR="${BASE_DIR}/test1_short_prompts"
mkdir -p "$TEST1_DIR"

# Create short-prompt datasets
echo "[test1] Creating 8-token dataset (6-10 tokens)..."
$PYTHON "${SCRIPT_DIR}/create_uniform_dataset.py" \
    --input "$LMSYS_10K" \
    --output "${TEST1_DIR}/short_8tok.jsonl" \
    --min-tokens 6 --max-tokens 10 --max-prompts 2000

echo "[test1] Creating 16-token dataset (12-20 tokens)..."
$PYTHON "${SCRIPT_DIR}/create_uniform_dataset.py" \
    --input "$LMSYS_10K" \
    --output "${TEST1_DIR}/short_16tok.jsonl" \
    --min-tokens 12 --max-tokens 20 --max-prompts 2000

SHORT_8_COUNT=$(wc -l < "${TEST1_DIR}/short_8tok.jsonl")
SHORT_16_COUNT=$(wc -l < "${TEST1_DIR}/short_16tok.jsonl")
echo "[test1] 8-token dataset: ${SHORT_8_COUNT} prompts"
echo "[test1] 16-token dataset: ${SHORT_16_COUNT} prompts"

# Settings to test
declare -A TEST1_ARGS
TEST1_ARGS[vanilla]=""
TEST1_ARGS[piecewise_cg]="--enable-piecewise-cuda-graph --disable-cuda-graph"

for setting in vanilla piecewise_cg; do
    SETTING_DIR="${TEST1_DIR}/${setting}"
    mkdir -p "$SETTING_DIR"

    launch_server "$setting" "${TEST1_ARGS[$setting]}" "$SETTING_DIR"

    # Warmup
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$LMSYS_10K" \
        --url "http://${HOST}:${PORT}" \
        --num-prompts 20 --max-new-tokens $MAX_NEW_TOKENS --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3

    # 8-token prompts, cc=1,2,3,4
    echo ""
    echo "  ── 8-token prompts (${SHORT_8_COUNT} prompts) ──"
    for cc in 1 2 3 4; do
        run_bench "${setting}_8tok" "${TEST1_DIR}/short_8tok.jsonl" $SHORT_8_COUNT $cc "$SETTING_DIR"
    done

    # 16-token prompts, cc=1,2,3,4
    echo ""
    echo "  ── 16-token prompts (${SHORT_16_COUNT} prompts) ──"
    for cc in 1 2 3 4; do
        run_bench "${setting}_16tok" "${TEST1_DIR}/short_16tok.jsonl" $SHORT_16_COUNT $cc "$SETTING_DIR"
    done
done

# ═══════════════════════════════════════════════════════════════════════
# TEST 2: 1000 prompts from 10K, flush before each run
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  TEST 2: 1000 prompts, flush before each run"
echo "################################################################"

TEST2_DIR="${BASE_DIR}/test2_1000_flush"
mkdir -p "$TEST2_DIR"

declare -A TEST2_ARGS
TEST2_ARGS[vanilla]=""
TEST2_ARGS[piecewise_cg]="--enable-piecewise-cuda-graph --disable-cuda-graph"

for setting in vanilla piecewise_cg; do
    SETTING_DIR="${TEST2_DIR}/${setting}"
    mkdir -p "$SETTING_DIR"

    launch_server "$setting" "${TEST2_ARGS[$setting]}" "$SETTING_DIR"

    # Warmup
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$LMSYS_10K" \
        --url "http://${HOST}:${PORT}" \
        --num-prompts 20 --max-new-tokens $MAX_NEW_TOKENS --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3

    echo ""
    echo "  ── 1000 prompts, flush before each cc ──"
    for cc in 1 2 4 8 16 32 64 128; do
        run_bench "${setting}_1k" "$LMSYS_10K" 1000 $cc "$SETTING_DIR" true
    done
done

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY"
echo "################################################################"

$PYTHON -c "
import json, os

base_dir = '${BASE_DIR}'

# ── Test 1: Short prompts ──
print()
print('=== Test 1: Very Short Prompts — Piecewise vs Vanilla ===')
print()

for tok_label in ['8tok', '16tok']:
    print(f'  --- {tok_label} ---')
    header = f'{\"Setting\":<20s}'
    ccs = [1, 2, 3, 4]
    for cc in ccs:
        header += f'  cc={cc:<3d}'
    print(f'  {header}')
    print(f'  ' + '-' * len(header))

    pw_tput = {}
    van_tput = {}

    for setting in ['vanilla', 'piecewise_cg']:
        row_tput = f'{setting:<20s}'
        row_ttft = f'{\"\":<20s}'
        for cc in ccs:
            f = os.path.join(base_dir, 'test1_short_prompts', setting, f'{setting}_{tok_label}_cc{cc}.json')
            if os.path.exists(f):
                with open(f) as fh:
                    d = json.load(fh)
                tput = d.get('prefill_throughput_tok_s', 0)
                ttft = d.get('mean_ttft_ms', 0)
                row_tput += f'  {tput:>6.0f}'
                if setting == 'piecewise_cg':
                    pw_tput[cc] = tput
                else:
                    van_tput[cc] = tput
            else:
                row_tput += f'  {\"N/A\":>6s}'
        print(f'  {row_tput}  tok/s')

    # Speedup row
    row = f'{\"pw/vanilla\":<20s}'
    for cc in ccs:
        if cc in pw_tput and cc in van_tput and van_tput[cc] > 0:
            ratio = pw_tput[cc] / van_tput[cc]
            row += f'  {ratio:>5.2f}x'
        else:
            row += f'  {\"N/A\":>6s}'
    print(f'  {row}')

    # Mean TTFT row
    print()
    header2 = f'{\"Setting\":<20s}'
    for cc in ccs:
        header2 += f'  cc={cc:<3d}'
    print(f'  {header2}  (mean TTFT ms)')
    print(f'  ' + '-' * len(header2))
    for setting in ['vanilla', 'piecewise_cg']:
        row = f'{setting:<20s}'
        for cc in ccs:
            f = os.path.join(base_dir, 'test1_short_prompts', setting, f'{setting}_{tok_label}_cc{cc}.json')
            if os.path.exists(f):
                with open(f) as fh:
                    d = json.load(fh)
                ttft = d.get('mean_ttft_ms', 0)
                row += f'  {ttft:>6.1f}'
            else:
                row += f'  {\"N/A\":>6s}'
        print(f'  {row}')
    print()

# ── Test 2: 1000 prompts with flush ──
print()
print('=== Test 2: 1000 Prompts with Flush — Piecewise vs Vanilla ===')
print()

ccs = [1, 2, 4, 8, 16, 32, 64, 128]
header = f'{\"Setting\":<20s}'
for cc in ccs:
    header += f'  cc={cc:<4d}'
print(f'{header}  (tok/s)')
print('-' * len(header))

pw_tput = {}
van_tput = {}

for setting in ['vanilla', 'piecewise_cg']:
    row = f'{setting:<20s}'
    for cc in ccs:
        f = os.path.join(base_dir, 'test2_1000_flush', setting, f'{setting}_1k_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            tput = d.get('prefill_throughput_tok_s', 0)
            row += f'  {tput:>7.0f}'
            if setting == 'piecewise_cg':
                pw_tput[cc] = tput
            else:
                van_tput[cc] = tput
        else:
            row += f'  {\"N/A\":>7s}'
    print(row)

# Speedup
row = f'{\"pw/vanilla\":<20s}'
for cc in ccs:
    if cc in pw_tput and cc in van_tput and van_tput[cc] > 0:
        ratio = pw_tput[cc] / van_tput[cc]
        row += f'  {ratio:>6.2f}x'
    else:
        row += f'  {\"N/A\":>7s}'
print(row)

# Mean TTFT
print()
header = f'{\"Setting\":<20s}'
for cc in ccs:
    header += f'  cc={cc:<4d}'
print(f'{header}  (mean TTFT ms)')
print('-' * len(header))

for setting in ['vanilla', 'piecewise_cg']:
    row = f'{setting:<20s}'
    for cc in ccs:
        f = os.path.join(base_dir, 'test2_1000_flush', setting, f'{setting}_1k_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                d = json.load(fh)
            ttft = d.get('mean_ttft_ms', 0)
            row += f'  {ttft:>7.1f}'
        else:
            row += f'  {\"N/A\":>7s}'
    print(row)

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 6 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
