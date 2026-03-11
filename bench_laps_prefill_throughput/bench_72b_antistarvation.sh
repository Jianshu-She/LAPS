#!/bin/bash
# Benchmark: LAPS anti-starvation (max_consecutive_short=4) vs Batch Prefill CUDA Graph
# Runs 3 times with different seeds, generates aggregate summary.
#
# Usage: bash bench_72b_antistarvation.sh
set -euo pipefail

MODEL="Qwen/Qwen2.5-72B-Instruct"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
TP_SIZE=2
PREFILL_GPUS="0,1"
DECODE_GPUS="2,3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RESULTS_DIR="${SCRIPT_DIR}/results_72b_antistarvation"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

NUM_PROMPTS=500
MAX_NEW_TOKENS=1
NUM_RUNS=3
SEEDS=(42 123 456)

CONCURRENCY_LEVELS="1 2 4 8 16 32 64 128"
LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256 --laps-max-consecutive-short 4"

TODAY=$(date +%Y-%m-%d)

mkdir -p "$BASE_RESULTS_DIR"

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

launch_servers() {
    local label=$1
    local prefill_extra_args=$2

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Model: ${MODEL}, TP=${TP_SIZE}"
    echo "  Prefill GPUs: ${PREFILL_GPUS}, Decode GPUs: ${DECODE_GPUS}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    # ── launch decode (TP=2) ──
    echo "[launch] Decode server on GPUs ${DECODE_GPUS}..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${BASE_RESULTS_DIR}/${label}_decode.log" 2>&1 &

    # ── launch prefill (TP=2) ──
    echo "[launch] Prefill server on GPUs ${PREFILL_GPUS} with: ${prefill_extra_args:-<none>}..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPUS $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --tp-size $TP_SIZE \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9301 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${BASE_RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 900
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  900

    # ── launch router ──
    echo "[launch] Router..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${BASE_RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    # ── warmup ──
    echo "[warmup] 20 requests..."
    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts 20 \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency 4 \
        > /dev/null 2>&1 || true
    sleep 3
}

run_concurrency_sweep() {
    local label=$1
    local run_dir=$2
    local seed=$3

    for cc in $CONCURRENCY_LEVELS; do
        echo ""
        echo "  --- ${label} | cc=${cc} | seed=${seed} ---"
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" \
            --url "http://${HOST}:${ROUTER_PORT}" \
            --num-prompts $NUM_PROMPTS \
            --max-new-tokens $MAX_NEW_TOKENS \
            --concurrency $cc \
            --seed $seed \
            --output "${run_dir}/${label}_cc${cc}.json" \
            2>&1 | tee "${run_dir}/${label}_cc${cc}.txt"
        sleep 2
    done
}

generate_run_summary() {
    local run_dir=$1
    local run_label=$2

    $PYTHON -c "
import json, os

results_dir = '${run_dir}'
settings = ['batch_prefill_cuda_graph', 'laps_antistarvation']
labels = {
    'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
    'laps_antistarvation': 'LAPS (anti-starvation N=4)',
}
ccs = [1, 2, 4, 8, 16, 32, 64, 128]

metrics = [
    ('prefill_throughput_tok_s', 'PREFILL THROUGHPUT (tokens/s)',   '{:>8.0f}'),
    ('request_throughput_req_s', 'REQUEST THROUGHPUT (req/s)',      '{:>8.1f}'),
    ('median_ttft_ms',          'MEDIAN TTFT (ms)',                 '{:>8.1f}'),
    ('mean_ttft_ms',            'MEAN TTFT (ms)',                   '{:>8.1f}'),
    ('p50_ttft_ms',             'P50 TTFT (ms)',                    '{:>8.1f}'),
    ('p90_ttft_ms',             'P90 TTFT (ms)',                    '{:>8.1f}'),
    ('p99_ttft_ms',             'P99 TTFT (ms)',                    '{:>8.1f}'),
    ('total_duration_s',        'TOTAL DURATION (s)',               '{:>8.1f}'),
]

cache = {}
for s in settings:
    for cc in ccs:
        f = os.path.join(results_dir, f'{s}_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                cache[(s, cc)] = json.load(fh)

title = 'Qwen2.5-72B-Instruct, TP=2 (2x H200) — ${run_label}'
print()
print('=' * 110)
print(f'  BENCHMARK SUMMARY — {title}')
print('=' * 110)

for key, section_title, fmt in metrics:
    print()
    print(f'  {section_title}')
    print('-' * 110)
    hdr = f\"{'Setting':<32s}\"
    for cc in ccs:
        hdr += f'  cc={cc:<5d}'
    print(hdr)
    print('-' * 110)

    for s in settings:
        row = f'{labels[s]:<32s}'
        for cc in ccs:
            d = cache.get((s, cc))
            if d and key in d:
                row += '  ' + fmt.format(d[key])
            else:
                row += f'  {\"N/A\":>8s}'
        print(row)

print()
print('=' * 110)
" 2>&1 | tee "${run_dir}/summary.txt"
}

trap cleanup EXIT

# ───────────────────────── main ─────────────────────────

SETTINGS_LABELS=("batch_prefill_cuda_graph" "laps_antistarvation")
SETTINGS_ARGS=(
    "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph"
    "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"
)

for run_idx in $(seq 1 $NUM_RUNS); do
    SEED=${SEEDS[$((run_idx - 1))]}
    RUN_DIR="${BASE_RESULTS_DIR}/${TODAY}_run${run_idx}"
    mkdir -p "$RUN_DIR"

    echo ""
    echo "################################################################"
    echo "  RUN ${run_idx}/${NUM_RUNS} — seed=${SEED} — ${RUN_DIR}"
    echo "################################################################"

    for i in "${!SETTINGS_LABELS[@]}"; do
        label="${SETTINGS_LABELS[$i]}"
        args="${SETTINGS_ARGS[$i]}"

        launch_servers "$label" "$args"
        run_concurrency_sweep "$label" "$RUN_DIR" "$SEED"
    done

    generate_run_summary "$RUN_DIR" "Run ${run_idx} (seed=${SEED})"
done

# ───────────────────────── aggregate summary (mean ± std) ─────────────────────────

echo ""
echo "============================================================"
echo "  GENERATING AGGREGATE SUMMARY (mean ± std across ${NUM_RUNS} runs)"
echo "============================================================"

$PYTHON -c "
import json, os, math

base_dir = '${BASE_RESULTS_DIR}'
today = '${TODAY}'
num_runs = ${NUM_RUNS}
settings = ['batch_prefill_cuda_graph', 'laps_antistarvation']
labels = {
    'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
    'laps_antistarvation': 'LAPS (anti-starvation N=4)',
}
ccs = [1, 2, 4, 8, 16, 32, 64, 128]

metrics = [
    ('prefill_throughput_tok_s', 'PREFILL THROUGHPUT (tokens/s)'),
    ('request_throughput_req_s', 'REQUEST THROUGHPUT (req/s)'),
    ('median_ttft_ms',          'MEDIAN TTFT (ms)'),
    ('mean_ttft_ms',            'MEAN TTFT (ms)'),
    ('p50_ttft_ms',             'P50 TTFT (ms)'),
    ('p90_ttft_ms',             'P90 TTFT (ms)'),
    ('p99_ttft_ms',             'P99 TTFT (ms)'),
    ('total_duration_s',        'TOTAL DURATION (s)'),
]

# Load all runs
all_data = {}
for run_idx in range(1, num_runs + 1):
    run_dir = os.path.join(base_dir, f'{today}_run{run_idx}')
    for s in settings:
        for cc in ccs:
            f = os.path.join(run_dir, f'{s}_cc{cc}.json')
            if os.path.exists(f):
                with open(f) as fh:
                    all_data[(s, cc, run_idx)] = json.load(fh)

def mean_std(values):
    if not values:
        return None, None
    m = sum(values) / len(values)
    if len(values) > 1:
        var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
        s = math.sqrt(var)
    else:
        s = 0.0
    return m, s

title = f'Qwen2.5-72B-Instruct, TP=2 (2x H200) — AGGREGATE ({num_runs} runs)'
lines = []
lines.append('')
lines.append('=' * 130)
lines.append(f'  BENCHMARK SUMMARY — {title}')
lines.append('=' * 130)

for key, section_title in metrics:
    lines.append('')
    lines.append(f'  {section_title}')
    lines.append('-' * 130)
    hdr = f\"{'Setting':<32s}\"
    for cc in ccs:
        hdr += f'  {\"cc=\"+str(cc):>13s}'
    lines.append(hdr)
    lines.append('-' * 130)

    for s in settings:
        row = f'{labels[s]:<32s}'
        for cc in ccs:
            vals = []
            for run_idx in range(1, num_runs + 1):
                d = all_data.get((s, cc, run_idx))
                if d and key in d:
                    vals.append(d[key])
            m, sd = mean_std(vals)
            if m is not None:
                if 'throughput' in key:
                    row += f'  {m:>6.0f}±{sd:>4.0f}'
                elif 'duration' in key:
                    row += f'  {m:>6.1f}±{sd:>4.1f}'
                else:
                    row += f'  {m:>6.1f}±{sd:>4.1f}'
            else:
                row += f'  {\"N/A\":>13s}'
        lines.append(row)

lines.append('')
lines.append('=' * 130)

# Also include the old LAPS results for comparison
old_results_dir = '${SCRIPT_DIR}/results_72b_runs'
old_data = {}
for run_idx in range(1, num_runs + 1):
    run_dir = os.path.join(old_results_dir, f'{today}_run{run_idx}')
    # Try today's date first, then look for any date
    if not os.path.exists(run_dir):
        # Find existing run dirs
        import glob
        dirs = sorted(glob.glob(os.path.join(old_results_dir, '*_run' + str(run_idx))))
        if dirs:
            run_dir = dirs[-1]
    for cc in ccs:
        f = os.path.join(run_dir, f'laps_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                old_data[('laps_old', cc, run_idx)] = json.load(fh)
        f2 = os.path.join(run_dir, f'batch_prefill_cuda_graph_cc{cc}.json')
        if os.path.exists(f2):
            with open(f2) as fh:
                old_data[('batch_prefill_old', cc, run_idx)] = json.load(fh)

if old_data:
    lines.append('')
    lines.append('=' * 130)
    lines.append('  COMPARISON WITH OLD LAPS (strict short-first, no anti-starvation)')
    lines.append('=' * 130)

    compare_settings = ['laps_antistarvation', 'laps_old', 'batch_prefill_cuda_graph', 'batch_prefill_old']
    compare_labels = {
        'laps_antistarvation': 'LAPS (NEW, N=4)',
        'laps_old': 'LAPS (OLD, strict)',
        'batch_prefill_cuda_graph': 'Batch Prefill (NEW run)',
        'batch_prefill_old': 'Batch Prefill (OLD run)',
    }

    for key, section_title in metrics:
        lines.append('')
        lines.append(f'  {section_title}')
        lines.append('-' * 130)
        hdr = f\"{'Setting':<32s}\"
        for cc in ccs:
            hdr += f'  {\"cc=\"+str(cc):>13s}'
        lines.append(hdr)
        lines.append('-' * 130)

        for s in compare_settings:
            if s in ('laps_old', 'batch_prefill_old'):
                src = old_data
            else:
                src = all_data
            row = f'{compare_labels[s]:<32s}'
            for cc in ccs:
                vals = []
                for run_idx in range(1, num_runs + 1):
                    d = src.get((s, cc, run_idx))
                    if d and key in d:
                        vals.append(d[key])
                m, sd = mean_std(vals)
                if m is not None:
                    if 'throughput' in key:
                        row += f'  {m:>6.0f}±{sd:>4.0f}'
                    elif 'duration' in key:
                        row += f'  {m:>6.1f}±{sd:>4.1f}'
                    else:
                        row += f'  {m:>6.1f}±{sd:>4.1f}'
                else:
                    row += f'  {\"N/A\":>13s}'
            lines.append(row)

    lines.append('')
    lines.append('=' * 130)

output = '\n'.join(lines)
print(output)
with open(os.path.join(base_dir, 'aggregate_summary.txt'), 'w') as f:
    f.write(output + '\n')
print(f'\nSaved to {base_dir}/aggregate_summary.txt')
" 2>&1 | tee "${BASE_RESULTS_DIR}/aggregate_summary.txt"

echo ""
echo "All runs complete. Results in: ${BASE_RESULTS_DIR}/"
echo "  Per-run results: ${BASE_RESULTS_DIR}/${TODAY}_run{1..${NUM_RUNS}}/"
echo "  Aggregate summary: ${BASE_RESULTS_DIR}/aggregate_summary.txt"
