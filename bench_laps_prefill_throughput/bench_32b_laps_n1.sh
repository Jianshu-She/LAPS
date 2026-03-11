#!/bin/bash
# Benchmark: LAPS N=1 on Qwen2.5-32B, TP=1, 3 runs
# Runs on GPUs 6,7 with separate ports.
#
# Usage: bash bench_32b_laps_n1.sh
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30500
DECODE_PORT=30501
ROUTER_PORT=30502
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
PREFILL_GPU=6
DECODE_GPU=7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_RESULTS_DIR="${SCRIPT_DIR}/results_32b_laps_n1"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

NUM_PROMPTS=500
MAX_NEW_TOKENS=1
NUM_RUNS=3
SEEDS=(42 123 456)

CONCURRENCY_LEVELS="1 2 4 8 16 32 64 128 256 512"

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
    echo "  Model: ${MODEL}, TP=1"
    echo "  Prefill GPU: ${PREFILL_GPU}, Decode GPU: ${DECODE_GPU}"
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    echo "[launch] Decode server on GPU ${DECODE_GPU}..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9500 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${BASE_RESULTS_DIR}/${label}_decode.log" 2>&1 &

    echo "[launch] Prefill server on GPU ${PREFILL_GPU} with: ${prefill_extra_args:-<none>}..."
    CUDA_VISIBLE_DEVICES=$PREFILL_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode prefill \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9501 \
        --host $HOST --port $PREFILL_PORT \
        --mem-fraction-static 0.85 \
        $prefill_extra_args \
        > "${BASE_RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  600

    echo "[launch] Router..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9501 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${BASE_RESULTS_DIR}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

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

trap cleanup EXIT

# ───────────────────────── main ─────────────────────────

# Only N=1 since N=2 and N=4 are running on other GPUs
SETTINGS_LABELS=("laps_n1")
SETTINGS_ARGS=(
    "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --enable-laps-scheduler --laps-length-threshold 256 --laps-max-consecutive-short 1"
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
done

# ───────────────────────── aggregate summary ─────────────────────────

echo ""
echo "============================================================"
echo "  GENERATING AGGREGATE SUMMARY (mean ± std across ${NUM_RUNS} runs)"
echo "============================================================"

$PYTHON -c "
import json, os, math

base_dir = '${BASE_RESULTS_DIR}'
today = '${TODAY}'
num_runs = ${NUM_RUNS}
settings = ['laps_n1']
labels = {'laps_n1': 'LAPS (N=1)'}
ccs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

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

title = f'Qwen2.5-32B, TP=1 (1x H200) — LAPS N=1 — AGGREGATE ({num_runs} runs)'
lines = []
lines.append('')
lines.append('=' * 170)
lines.append(f'  BENCHMARK SUMMARY — {title}')
lines.append('=' * 170)

for key, section_title in metrics:
    lines.append('')
    lines.append(f'  {section_title}')
    lines.append('-' * 170)
    hdr = f\"{'Setting':<28s}\"
    for cc in ccs:
        hdr += f'  {\"cc=\"+str(cc):>15s}'
    lines.append(hdr)
    lines.append('-' * 170)

    for s in settings:
        row = f'{labels[s]:<28s}'
        for cc in ccs:
            vals = []
            for run_idx in range(1, num_runs + 1):
                d = all_data.get((s, cc, run_idx))
                if d and key in d:
                    vals.append(d[key])
            m, sd = mean_std(vals)
            if m is not None:
                if 'throughput' in key:
                    row += f'  {m:>7.0f}±{sd:>5.0f}'
                elif 'duration' in key:
                    row += f'  {m:>7.1f}±{sd:>5.1f}'
                else:
                    row += f'  {m:>7.1f}±{sd:>5.1f}'
            else:
                row += f'  {\"N/A\":>15s}'
        lines.append(row)

lines.append('')
lines.append('=' * 170)

output = '\n'.join(lines)
print(output)
with open(os.path.join(base_dir, 'aggregate_summary.txt'), 'w') as f:
    f.write(output + '\n')
print(f'\nSaved to {base_dir}/aggregate_summary.txt')
" 2>&1 | tee "${BASE_RESULTS_DIR}/aggregate_summary.txt"

echo ""
echo "All runs complete. Results in: ${BASE_RESULTS_DIR}/"
