#!/bin/bash
# Experiments 9-12: Diagnose CG Slowdown at High Concurrency
#
# Exp 8 showed piecewise CG & batch prefill CG are slower than vanilla
# at high concurrency. These experiments test three hypotheses:
#   H1: Padding waste — CG pads prefill batches, wasting FLOPs
#   H2: Memory pressure — CG memory reduces available GPU memory
#   H3: Batching degradation — CG drains queue fast (bs=1 replays),
#       preventing scheduler from accumulating large batches
#
# Pre-exp:  Parse existing Exp 8 server logs (no new runs)
# Exp 9:   Uniform dataset (isolate padding from batching)     — 9 runs
# Exp 10:  Piecewise CG max-tokens sweep (control CG coverage) — 12 runs
# Exp 11:  Batch CG grid density (isolate batch_cg padding)    — 8 runs
# Exp 12:  KV cache / memory pressure (rule out H2)            — 6 runs
# Total: 35 runs + log analysis
#
# Usage: bash bench_exp9_12_diagnose.sh [GPU_ID]
#   GPU_ID defaults to 5
#
set -euo pipefail

GPU=${1:-5}
PORT=30003
HOST="127.0.0.1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"
BASELINE_PY="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/baseline.py"
DATASET="/mnt/weka/home/jianshu.she/routing_system/sglang/prefill_interfere/scheduling/lmsys-chat/dataset/human_prompts_stream_10000.jsonl"

TODAY=$(date +%Y-%m-%d)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/results_exp9_12_${TODAY}"
EXP8_DIR="${SCRIPT_DIR}/results_exp8_2026-03-02"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1

MODEL_7B="Qwen/Qwen2.5-7B-Instruct"
MODEL_32B="Qwen/Qwen2.5-32B"

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

# launch_server MODEL SETTING_DIR LOG_NAME EXTRA_ARGS...
launch_server() {
    local model=$1
    local setting_dir=$2
    local log_name=$3
    shift 3
    local extra_args="$*"

    cleanup

    echo "  [launch] model=$model args=${extra_args:-<none>}"
    CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
        --model-path "$model" \
        --host $HOST --port $PORT \
        --mem-fraction-static 0.85 \
        $extra_args \
        > "${setting_dir}/${log_name}_server.log" 2>&1 &

    wait_ready "http://${HOST}:${PORT}/health" 600
}

# warmup SETTING_DIR MODEL
warmup() {
    local setting_dir=$1
    local model=$2
    echo "  [warmup] 20 requests with cc=4..."
    $PYTHON "$BASELINE_PY" \
        --base-url "http://${HOST}:${PORT}" \
        --model "$model" \
        --concurrency 4 \
        --num-requests 20 \
        --max-tokens $MAX_TOKENS \
        --prompt-file "$DATASET" \
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

# run_sweep SETTING_DIR MODEL DESC CC_ARRAY [PROMPT_FILE]
# CC_ARRAY is passed as remaining args after prompt_file
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

echo "Experiments 9-12: Diagnose CG Slowdown at High Concurrency"
echo "  GPU:      ${GPU}"
echo "  Port:     ${PORT}"
echo "  Dataset:  ${DATASET}"
echo "  Requests: ${NUM_REQUESTS}"
echo "  Exp8 dir: ${EXP8_DIR}"
echo "  Results:  ${BASE_DIR}/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PRE-EXPERIMENT: Parse Existing Exp 8 Server Logs
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  PRE-EXPERIMENT: Parse Exp 8 Server Logs"
echo "################################################################"

$PYTHON -c "
import re, os, json, sys
from collections import defaultdict
from datetime import datetime

exp8_dir = '${EXP8_DIR}'
output_dir = '${BASE_DIR}/pre_experiment'
os.makedirs(output_dir, exist_ok=True)

def parse_timestamp(line):
    m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
    if m:
        return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
    return None

def get_run_windows(mlabel, setting):
    '''Get (start_ts, end_ts) for each cc run from milestones/summary.'''
    windows = {}
    setting_dir = os.path.join(exp8_dir, mlabel, setting)
    if not os.path.isdir(setting_dir):
        return windows

    for d in os.listdir(setting_dir):
        if not d.startswith('cc'):
            continue
        cc = int(d[2:])
        summary_f = os.path.join(setting_dir, d, 'summary.json')
        if os.path.exists(summary_f):
            with open(summary_f) as f:
                s = json.load(f)
            windows[cc] = s.get('wall_time_sec', 0)
    return windows

# ── 1) BP-CG-REPLAY padding analysis (batch_cg logs) ──
print('=' * 80)
print('  1) BP-CG-REPLAY Padding Analysis (batch_cg)')
print('=' * 80)

for mlabel in ['7b', '32b']:
    log_path = os.path.join(exp8_dir, mlabel, 'batch_cg', 'batch_cg_server.log')
    if not os.path.exists(log_path):
        print(f'  [{mlabel}] No batch_cg server log found')
        continue

    # Parse all BP-CG-REPLAY lines
    replays = []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'BP-CG-REPLAY actual_bs=(\d+) target_bs=(\d+) '
                r'max_extend_len=(\d+) target_seq_len=(\d+) '
                r'real_tokens=(\d+) padded_tokens=(\d+) ratio=([\d.]+)',
                line
            )
            if m:
                ts = parse_timestamp(line)
                replays.append({
                    'ts': ts,
                    'actual_bs': int(m.group(1)),
                    'target_bs': int(m.group(2)),
                    'max_extend_len': int(m.group(3)),
                    'target_seq_len': int(m.group(4)),
                    'real_tokens': int(m.group(5)),
                    'padded_tokens': int(m.group(6)),
                    'ratio': float(m.group(7)),
                })

    if not replays:
        print(f'  [{mlabel}] No BP-CG-REPLAY lines found')
        continue

    print(f'  [{mlabel}] Total BP-CG-REPLAY entries: {len(replays)}')
    total_real = sum(r['real_tokens'] for r in replays)
    total_padded = sum(r['padded_tokens'] for r in replays)
    avg_ratio = total_padded / total_real if total_real > 0 else 0
    print(f'  [{mlabel}] Overall: real={total_real:,} padded={total_padded:,} '
          f'avg_ratio={avg_ratio:.2f}x waste={1-total_real/total_padded:.1%}')

    # Breakdown by target_bs
    by_bs = defaultdict(lambda: {'real': 0, 'padded': 0, 'count': 0})
    for r in replays:
        k = r['target_bs']
        by_bs[k]['real'] += r['real_tokens']
        by_bs[k]['padded'] += r['padded_tokens']
        by_bs[k]['count'] += 1

    print(f'  [{mlabel}] By target_bs:')
    print(f\"    {'bs':>4s}  {'count':>6s}  {'real':>10s}  {'padded':>10s}  {'ratio':>6s}  {'waste':>6s}\")
    for bs in sorted(by_bs.keys()):
        d = by_bs[bs]
        ratio = d['padded'] / d['real'] if d['real'] > 0 else 0
        waste = 1 - d['real'] / d['padded'] if d['padded'] > 0 else 0
        print(f\"    {bs:>4d}  {d['count']:>6d}  {d['real']:>10,d}  {d['padded']:>10,d}  {ratio:>6.2f}  {waste:>5.1%}\")

    # Breakdown by target_seq_len
    by_seq = defaultdict(lambda: {'real': 0, 'padded': 0, 'count': 0})
    for r in replays:
        k = r['target_seq_len']
        by_seq[k]['real'] += r['real_tokens']
        by_seq[k]['padded'] += r['padded_tokens']
        by_seq[k]['count'] += 1

    print(f'  [{mlabel}] By target_seq_len:')
    print(f\"    {'seq':>6s}  {'count':>6s}  {'real':>10s}  {'padded':>10s}  {'ratio':>6s}  {'waste':>6s}\")
    for seq in sorted(by_seq.keys()):
        d = by_seq[seq]
        ratio = d['padded'] / d['real'] if d['real'] > 0 else 0
        waste = 1 - d['real'] / d['padded'] if d['padded'] > 0 else 0
        print(f\"    {seq:>6d}  {d['count']:>6d}  {d['real']:>10,d}  {d['padded']:>10,d}  {ratio:>6.2f}  {waste:>5.1%}\")
    print()

# ── 2) Forward path distribution ──
print('=' * 80)
print('  2) Forward Path Distribution (FWD-PATH)')
print('=' * 80)

for mlabel in ['7b', '32b']:
    for setting in ['piecewise_cg', 'batch_cg']:
        log_path = os.path.join(exp8_dir, mlabel, setting, f'{setting}_server.log')
        if not os.path.exists(log_path):
            continue

        fwd_paths = []
        with open(log_path) as f:
            for line in f:
                m = re.search(r'FWD-PATH path=(\S+) bs=(\d+)', line)
                if m:
                    ts = parse_timestamp(line)
                    fwd_paths.append({
                        'ts': ts,
                        'path': m.group(1),
                        'bs': int(m.group(2)),
                    })

        if not fwd_paths:
            continue

        # Aggregate
        path_counts = defaultdict(lambda: {'count': 0, 'total_bs': 0, 'bs_list': []})
        for fp in fwd_paths:
            p = fp['path']
            path_counts[p]['count'] += 1
            path_counts[p]['total_bs'] += fp['bs']
            path_counts[p]['bs_list'].append(fp['bs'])

        print(f'  [{mlabel}/{setting}] Total FWD-PATH entries: {len(fwd_paths)}')
        for path_name in sorted(path_counts.keys()):
            d = path_counts[path_name]
            avg_bs = d['total_bs'] / d['count'] if d['count'] > 0 else 0
            bs_list = d['bs_list']
            max_bs = max(bs_list) if bs_list else 0
            # Count bs=1
            bs1_count = sum(1 for b in bs_list if b == 1)
            print(f'    path={path_name}: count={d[\"count\"]:,} avg_bs={avg_bs:.1f} '
                  f'max_bs={max_bs} bs=1_frac={bs1_count/len(bs_list):.1%}')
        print()

# ── 3) Prefill batch size distribution ──
print('=' * 80)
print('  3) Prefill Batch Size Distribution')
print('=' * 80)

for mlabel in ['7b', '32b']:
    for setting in ['vanilla', 'piecewise_cg', 'batch_cg']:
        log_path = os.path.join(exp8_dir, mlabel, setting, f'{setting}_server.log')
        if not os.path.exists(log_path):
            continue

        prefill_batches = []
        with open(log_path) as f:
            for line in f:
                m = re.search(r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)', line)
                if m:
                    ts = parse_timestamp(line)
                    prefill_batches.append({
                        'ts': ts,
                        'new_seq': int(m.group(1)),
                        'new_token': int(m.group(2)),
                    })

        if not prefill_batches:
            continue

        seqs = [p['new_seq'] for p in prefill_batches]
        tokens = [p['new_token'] for p in prefill_batches]
        avg_seq = sum(seqs) / len(seqs) if seqs else 0
        avg_tok = sum(tokens) / len(tokens) if tokens else 0
        max_seq = max(seqs) if seqs else 0
        max_tok = max(tokens) if tokens else 0

        print(f'  [{mlabel}/{setting}] Prefill batches: {len(prefill_batches):,}')
        print(f'    avg_new_seq={avg_seq:.1f} max_new_seq={max_seq} '
              f'avg_new_token={avg_tok:.1f} max_new_token={max_tok}')

        # Distribution of new_seq counts
        seq_dist = defaultdict(int)
        for s in seqs:
            seq_dist[s] += 1
        top_seqs = sorted(seq_dist.items(), key=lambda x: -x[1])[:10]
        print(f'    top new_seq counts: ' +
              ', '.join(f'{s}:{c}' for s, c in top_seqs))
        print()

# Save analysis
print(f'Pre-experiment analysis complete. Output dir: {output_dir}')
" 2>&1 | tee "${BASE_DIR}/pre_experiment_analysis.txt"

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 9: Uniform Dataset (isolate padding from batching)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 9: Uniform Dataset"
echo "  Tests H1 (padding) vs H3 (batching)"
echo "################################################################"

# Create uniform dataset
UNIFORM_DATASET="${BASE_DIR}/uniform_dataset.jsonl"
echo "[exp9] Creating uniform dataset (40-80 token prompts)..."
$PYTHON "${SCRIPT_DIR}/create_uniform_dataset.py" \
    --input "$DATASET" \
    --output "$UNIFORM_DATASET" \
    --min-tokens 40 --max-tokens 80 --max-prompts 10000

EXP9_CCS=(4 32 64)

# ── Exp 9: vanilla ──
EXP9_DIR="${BASE_DIR}/exp9/vanilla"
mkdir -p "$EXP9_DIR"

echo ""
echo "============================================================"
echo "  Exp 9: vanilla (uniform dataset)"
echo "============================================================"
launch_server "$MODEL_7B" "$EXP9_DIR" "vanilla"
warmup "$EXP9_DIR" "$MODEL_7B"
run_sweep "$EXP9_DIR" "$MODEL_7B" "exp9_vanilla" "$UNIFORM_DATASET" "${EXP9_CCS[@]}"

# ── Exp 9: piecewise_cg ──
EXP9_DIR="${BASE_DIR}/exp9/piecewise_cg"
mkdir -p "$EXP9_DIR"

echo ""
echo "============================================================"
echo "  Exp 9: piecewise_cg (uniform dataset)"
echo "============================================================"
launch_server "$MODEL_7B" "$EXP9_DIR" "piecewise_cg" \
    --enable-piecewise-cuda-graph --disable-cuda-graph
warmup "$EXP9_DIR" "$MODEL_7B"
run_sweep "$EXP9_DIR" "$MODEL_7B" "exp9_piecewise_cg" "$UNIFORM_DATASET" "${EXP9_CCS[@]}"

# ── Exp 9: batch_cg ──
EXP9_DIR="${BASE_DIR}/exp9/batch_cg"
mkdir -p "$EXP9_DIR"

echo ""
echo "============================================================"
echo "  Exp 9: batch_cg (uniform dataset)"
echo "============================================================"
launch_server "$MODEL_7B" "$EXP9_DIR" "batch_cg" \
    --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph \
    --batch-prefill-batch-sizes 1 2 4 8 16 32 \
    --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 \
    --batch-prefill-max-seq-len 512
warmup "$EXP9_DIR" "$MODEL_7B"
run_sweep "$EXP9_DIR" "$MODEL_7B" "exp9_batch_cg" "$UNIFORM_DATASET" "${EXP9_CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 10: Piecewise CG Max-Tokens Sweep
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 10: Piecewise CG Max-Tokens Sweep"
echo "  Tests H3 (batching degradation)"
echo "################################################################"

EXP10_CCS=(4 8 32 64)

declare -A EXP10_CAPS
EXP10_CAPS[piecewise_cap256]=256
EXP10_CAPS[piecewise_cap512]=512
EXP10_CAPS[piecewise_cap1024]=1024
EXP10_SETTINGS=(piecewise_cap256 piecewise_cap512 piecewise_cap1024)

for setting in "${EXP10_SETTINGS[@]}"; do
    cap=${EXP10_CAPS[$setting]}
    EXP10_DIR="${BASE_DIR}/exp10/${setting}"
    mkdir -p "$EXP10_DIR"

    echo ""
    echo "============================================================"
    echo "  Exp 10: ${setting} (max_tokens=${cap})"
    echo "============================================================"
    launch_server "$MODEL_7B" "$EXP10_DIR" "$setting" \
        --enable-piecewise-cuda-graph --disable-cuda-graph \
        --piecewise-cuda-graph-max-tokens "$cap"
    warmup "$EXP10_DIR" "$MODEL_7B"
    run_sweep "$EXP10_DIR" "$MODEL_7B" "exp10_${setting}" "$DATASET" "${EXP10_CCS[@]}"
done

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 11: Batch CG Grid Density
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 11: Batch CG Grid Density"
echo "  Tests H1 (padding waste) for batch_cg"
echo "################################################################"

EXP11_CCS=(4 8 32 64)

# ── batch_cg_dense: 12 × 11 = 132 graphs ──
EXP11_DIR="${BASE_DIR}/exp11/batch_cg_dense"
mkdir -p "$EXP11_DIR"

echo ""
echo "============================================================"
echo "  Exp 11: batch_cg_dense (132 graphs)"
echo "============================================================"
launch_server "$MODEL_7B" "$EXP11_DIR" "batch_cg_dense" \
    --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph \
    --batch-prefill-batch-sizes 1 2 3 4 5 6 7 8 12 16 24 32 \
    --batch-prefill-seq-lengths 16 24 32 48 64 96 128 192 256 384 512 \
    --batch-prefill-max-seq-len 512
warmup "$EXP11_DIR" "$MODEL_7B"
run_sweep "$EXP11_DIR" "$MODEL_7B" "exp11_batch_cg_dense" "$DATASET" "${EXP11_CCS[@]}"

# ── batch_cg_vdense: 14 × 16 = 224 graphs ──
EXP11_DIR="${BASE_DIR}/exp11/batch_cg_vdense"
mkdir -p "$EXP11_DIR"

echo ""
echo "============================================================"
echo "  Exp 11: batch_cg_vdense (224 graphs)"
echo "============================================================"
launch_server "$MODEL_7B" "$EXP11_DIR" "batch_cg_vdense" \
    --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph \
    --batch-prefill-batch-sizes 1 2 3 4 5 6 7 8 10 12 16 20 24 32 \
    --batch-prefill-seq-lengths 8 12 16 24 32 48 64 80 96 128 160 192 256 320 384 512 \
    --batch-prefill-max-seq-len 512
warmup "$EXP11_DIR" "$MODEL_7B"
run_sweep "$EXP11_DIR" "$MODEL_7B" "exp11_batch_cg_vdense" "$DATASET" "${EXP11_CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 12: KV Cache / Memory Pressure (32B model)
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 12: KV Cache / Memory Pressure"
echo "  Tests H2 (memory pressure) — 32B model"
echo "################################################################"

EXP12_CCS=(4 32 64)

# ── vanilla_kv180k ──
EXP12_DIR="${BASE_DIR}/exp12/vanilla_kv180k"
mkdir -p "$EXP12_DIR"

echo ""
echo "============================================================"
echo "  Exp 12: vanilla_kv180k (32B, max-total-tokens=180000)"
echo "============================================================"
launch_server "$MODEL_32B" "$EXP12_DIR" "vanilla_kv180k" \
    --max-total-tokens 180000
warmup "$EXP12_DIR" "$MODEL_32B"
run_sweep "$EXP12_DIR" "$MODEL_32B" "exp12_vanilla_kv180k" "$DATASET" "${EXP12_CCS[@]}"

# ── vanilla_kv120k ──
EXP12_DIR="${BASE_DIR}/exp12/vanilla_kv120k"
mkdir -p "$EXP12_DIR"

echo ""
echo "============================================================"
echo "  Exp 12: vanilla_kv120k (32B, max-total-tokens=120000)"
echo "============================================================"
launch_server "$MODEL_32B" "$EXP12_DIR" "vanilla_kv120k" \
    --max-total-tokens 120000
warmup "$EXP12_DIR" "$MODEL_32B"
run_sweep "$EXP12_DIR" "$MODEL_32B" "exp12_vanilla_kv120k" "$DATASET" "${EXP12_CCS[@]}"

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY — Experiments 9-12"
echo "################################################################"

$PYTHON -c "
import json, os, sys

base_dir = '${BASE_DIR}'
exp8_dir = '${EXP8_DIR}'

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_exp8(mlabel, setting, cc):
    return load(os.path.join(exp8_dir, mlabel, setting, f'cc{cc}', 'summary.json'))

def load_exp(exp, setting, cc):
    return load(os.path.join(base_dir, exp, setting, f'cc{cc}', 'summary.json'))

def rps(d):
    return d['rps'] if d else None

def prefill_tput(d):
    if d and d['wall_time_sec'] > 0:
        return d['tokens']['total_prompt_tokens'] / d['wall_time_sec']
    return None

def fmt(val, width=8):
    if val is None:
        return f\"{'N/A':>{width}s}\"
    return f'{val:>{width}.1f}'

def fmt_speedup(val, ref):
    if val is None or ref is None or ref == 0:
        return f\"{'N/A':>8s}\"
    return f'{val/ref:>7.3f}x'

# ═══════════════════════════════════════════════════════════════════
# Exp 9: Uniform Dataset
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  EXPERIMENT 9: Uniform Dataset (7B, 40-80 token prompts)')
print('  Tests: H1 (padding) vs H3 (batching)')
print('=' * 100)

exp9_settings = ['vanilla', 'piecewise_cg', 'batch_cg']
exp9_ccs = [4, 32, 64]

print()
print('  --- RPS (requests/sec) ---')
hdr = f\"{'Setting':<20s}\"
for cc in exp9_ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")

for s in exp9_settings:
    row = f'{s:<20s}'
    for cc in exp9_ccs:
        d = load_exp('exp9', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs vanilla
print()
print('  --- Speedup vs Exp9 vanilla ---')
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")
for s in exp9_settings:
    row = f'{s:<20s}'
    for cc in exp9_ccs:
        d_v = load_exp('exp9', 'vanilla', cc)
        d_s = load_exp('exp9', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

# Compare with Exp8 baselines
print()
print('  --- Exp9 (uniform) vs Exp8 (mixed) RPS ---')
hdr2 = f\"{'Setting':<20s}\"
for cc in exp9_ccs:
    hdr2 += f'  exp9/{cc:<3d}  exp8/{cc:<3d}'
print(f'  {hdr2}')
print(f\"  {'-' * len(hdr2)}\")
for s in exp9_settings:
    row = f'{s:<20s}'
    for cc in exp9_ccs:
        d9 = load_exp('exp9', s, cc)
        d8 = load_exp8('7b', s, cc)
        row += f'  {fmt(rps(d9))}  {fmt(rps(d8))}'
    print(f'  {row}')

print()
print('  INTERPRETATION:')
print('  - If CG matches vanilla w/ uniform data → H1 (padding waste) is the cause')
print('  - If CG still slower w/ uniform data   → H3 (batching degradation) is the cause')

# ═══════════════════════════════════════════════════════════════════
# Exp 10: Piecewise CG Max-Tokens Sweep
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  EXPERIMENT 10: Piecewise CG Max-Tokens Sweep (7B)')
print('  Tests: H3 (batching degradation) for piecewise CG')
print('=' * 100)

exp10_settings = ['piecewise_cap256', 'piecewise_cap512', 'piecewise_cap1024']
exp10_ccs = [4, 8, 32, 64]

print()
print('  --- RPS (requests/sec) ---')
hdr = f\"{'Setting':<20s}\"
for cc in exp10_ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")

# Exp8 baselines first
for s in ['vanilla', 'piecewise_cg']:
    row = f'{'exp8_'+s:<20s}'
    for cc in exp10_ccs:
        d = load_exp8('7b', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

print(f\"  {'-' * len(hdr)}\")

for s in exp10_settings:
    row = f'{s:<20s}'
    for cc in exp10_ccs:
        d = load_exp('exp10', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs exp8 vanilla
print()
print('  --- Speedup vs Exp8 vanilla ---')
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")
for s in ['piecewise_cg'] + exp10_settings:
    label = 'exp8_piecewise_cg' if s == 'piecewise_cg' else s
    row = f'{label:<20s}'
    for cc in exp10_ccs:
        d_v = load_exp8('7b', 'vanilla', cc)
        if s == 'piecewise_cg':
            d_s = load_exp8('7b', s, cc)
        else:
            d_s = load_exp('exp10', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

print()
print('  INTERPRETATION:')
print('  - cap256 best at cc=64 → H3 (lower cap forces eager, better batching)')
print('  - cap256 worst at cc=4 → small batches miss CG benefit')
print('  - Crossover point reveals optimal adaptive cap')

# ═══════════════════════════════════════════════════════════════════
# Exp 11: Batch CG Grid Density
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  EXPERIMENT 11: Batch CG Grid Density (7B)')
print('  Tests: H1 (padding waste) for batch_cg')
print('=' * 100)

exp11_settings = ['batch_cg_dense', 'batch_cg_vdense']
exp11_ccs = [4, 8, 32, 64]

print()
print('  --- RPS (requests/sec) ---')
hdr = f\"{'Setting':<20s}\"
for cc in exp11_ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")

# Exp8 baselines
for s in ['vanilla', 'batch_cg']:
    row = f'{'exp8_'+s:<20s}'
    for cc in exp11_ccs:
        d = load_exp8('7b', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

print(f\"  {'-' * len(hdr)}\")

for s in exp11_settings:
    row = f'{s:<20s}'
    for cc in exp11_ccs:
        d = load_exp('exp11', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs exp8 vanilla
print()
print('  --- Speedup vs Exp8 vanilla ---')
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")
for s in ['batch_cg'] + exp11_settings:
    label = 'exp8_batch_cg' if s == 'batch_cg' else s
    row = f'{label:<20s}'
    for cc in exp11_ccs:
        d_v = load_exp8('7b', 'vanilla', cc)
        if s == 'batch_cg':
            d_s = load_exp8('7b', s, cc)
        else:
            d_s = load_exp('exp11', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

print()
print('  INTERPRETATION:')
print('  - Dense grid helps → H1 (padding waste is the bottleneck)')
print('  - Dense grid no help → H3 (batching degradation, not padding)')

# ═══════════════════════════════════════════════════════════════════
# Exp 12: Memory Pressure (32B)
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  EXPERIMENT 12: KV Cache / Memory Pressure (32B)')
print('  Tests: H2 (memory pressure)')
print('=' * 100)

exp12_settings = ['vanilla_kv180k', 'vanilla_kv120k']
exp12_ccs = [4, 32, 64]

print()
print('  --- RPS (requests/sec) ---')
hdr = f\"{'Setting':<20s}\"
for cc in exp12_ccs:
    hdr += f'  cc={cc:<5d}'
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")

# Exp8 baselines
for s in ['vanilla', 'piecewise_cg', 'batch_cg']:
    row = f'{'exp8_'+s:<20s}'
    for cc in exp12_ccs:
        d = load_exp8('32b', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

print(f\"  {'-' * len(hdr)}\")

for s in exp12_settings:
    row = f'{s:<20s}'
    for cc in exp12_ccs:
        d = load_exp('exp12', s, cc)
        row += f'  {fmt(rps(d))}'
    print(f'  {row}')

# Speedup vs exp8 vanilla (full KV)
print()
print('  --- Speedup vs Exp8 vanilla (full KV, 32B) ---')
print(f'  {hdr}')
print(f\"  {'-' * len(hdr)}\")
for s in exp12_settings:
    row = f'{s:<20s}'
    for cc in exp12_ccs:
        d_v = load_exp8('32b', 'vanilla', cc)
        d_s = load_exp('exp12', s, cc)
        row += f'  {fmt_speedup(rps(d_s), rps(d_v))}'
    print(f'  {row}')

print()
print('  INTERPRETATION:')
print('  - No change → H2 refuted (memory is not the issue)')
print('  - Degradation → H2 confirmed (CG memory reduces available KV cache)')

# ═══════════════════════════════════════════════════════════════════
# Diagnosis Matrix
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  DIAGNOSIS MATRIX')
print('=' * 100)
print()
print('  | Result                                    | Cause           | Implication                |')
print('  |-------------------------------------------|-----------------|----------------------------|')
print('  | Exp 9: CG matches vanilla w/ uniform data | H1 (padding)    | Optimize grid granularity  |')
print('  | Exp 9: CG still slower w/ uniform data    | H3 (batching)   | Need scheduling-aware CG   |')
print('  | Exp 10: lower cap -> better at high cc    | H3 (batching)   | Adaptive cap per load      |')
print('  | Exp 11: dense grid helps batch_cg         | H1 (padding)    | Ship denser default grid   |')
print('  | Exp 12: reduced KV has no effect          | H2 refuted      | Memory is not the issue    |')

# ═══════════════════════════════════════════════════════════════════
# Latency comparison
# ═══════════════════════════════════════════════════════════════════
print()
print('=' * 100)
print('  LATENCY SUMMARY (avg ms)')
print('=' * 100)

all_experiments = [
    ('Exp 9', 'exp9', exp9_settings, exp9_ccs, load_exp),
    ('Exp 10', 'exp10', exp10_settings, exp10_ccs, load_exp),
    ('Exp 11', 'exp11', exp11_settings, exp11_ccs, load_exp),
    ('Exp 12', 'exp12', exp12_settings, exp12_ccs, load_exp),
]

for exp_name, exp_key, settings, ccs, loader in all_experiments:
    print(f'  --- {exp_name} ---')
    hdr = f\"{'Setting':<20s}\"
    for cc in ccs:
        hdr += f'  cc={cc:<5d}'
    print(f'  {hdr}')
    print(f\"  {'-' * len(hdr)}\")
    for s in settings:
        row = f'{s:<20s}'
        for cc in ccs:
            d = loader(exp_key, s, cc)
            if d and 'latency_ms' in d:
                row += f'  {d[\"latency_ms\"][\"avg\"]:>8.1f}'
            else:
                row += f\"  {'N/A':>8s}\"
        print(f'  {row}')
    print()

# Fail check
print()
print('=== Fail Count Check ===')
all_ok = True
for exp_key, settings, ccs in [
    ('exp9', exp9_settings, exp9_ccs),
    ('exp10', exp10_settings, exp10_ccs),
    ('exp11', exp11_settings, exp11_ccs),
    ('exp12', exp12_settings, exp12_ccs),
]:
    for s in settings:
        for cc in ccs:
            d = load_exp(exp_key, s, cc)
            if d and d.get('fail', 0) > 0:
                print(f'  WARNING: {exp_key}/{s}/cc{cc} has {d[\"fail\"]} failures!')
                all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENTS 9-12 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
