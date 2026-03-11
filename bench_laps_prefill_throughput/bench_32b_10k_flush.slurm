#!/bin/bash
#SBATCH --job-name=bench-32b-10k-flush
#SBATCH --account=iq
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/weka/home/jianshu.she/graph/sglang/bench_laps_prefill_throughput/slurm_32b_10k_flush_%j.out
#SBATCH --error=/mnt/weka/home/jianshu.she/graph/sglang/bench_laps_prefill_throughput/slurm_32b_10k_flush_%j.err

# ─────────────────────────────────────────────────────────────────────
# Benchmark: Qwen2.5-32B, TP=1, 10K prompts, KV cache flush enabled
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
SCRIPT_DIR="/mnt/weka/home/jianshu.she/graph/sglang/bench_laps_prefill_throughput"
DATASET="${SCRIPT_DIR}/../data/lmsys_chat_10k.jsonl"
TODAY=$(date +%Y-%m-%d)
BASE_RESULTS_DIR="${SCRIPT_DIR}/results_32b_10k_flush_${TODAY}"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
PREFILL_GPU=0
DECODE_GPU=1

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1
NUM_RUNS=1
SEEDS=(42)

CONCURRENCY_LEVELS="4 8 16 32 64 128"
LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256 --laps-max-consecutive-short 1"

mkdir -p "$BASE_RESULTS_DIR"

# ───────────────────────── record config ─────────────────────────

cat > "${BASE_RESULTS_DIR}/config.txt" <<CONFIGEOF
===============================================================================
  EXPERIMENT CONFIG — $(date '+%Y-%m-%d %H:%M:%S')
===============================================================================
  Launch command: sbatch $(basename "$0")
  Script:        $(readlink -f "$0")
  SLURM Job ID:  ${SLURM_JOB_ID:-N/A}
  SLURM Node:    ${SLURM_NODELIST:-N/A}

  ┌──────────────────┬────────────────────────────────────────────────────────────────────┐
  │    Parameter     │                               Value                                │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Model            │ ${MODEL}                                                           │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ NUM_PROMPTS      │ ${NUM_PROMPTS}                                                     │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ MAX_NEW_TOKENS   │ ${MAX_NEW_TOKENS}                                                  │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ NUM_RUNS         │ ${NUM_RUNS}                                                        │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ SEEDS            │ ${SEEDS[*]}                                                        │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ LAPS N           │ 1 (--laps-max-consecutive-short 1)                                 │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ KV cache flush   │ Yes (between each concurrency level)                               │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Concurrency      │ ${CONCURRENCY_LEVELS}                                              │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Piecewise CG     │ Original sglang defaults (up to 8192 tokens)                       │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Decode CG        │ Disabled (--disable-cuda-graph)                                    │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Batch Prefill CG │ bs=[1,2,4,8,16,32], seq=[1,2,4,8,16,32,64,128,256,512] → 60 graphs│
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Prefill GPU      │ ${PREFILL_GPU}                                                     │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Decode GPU       │ ${DECODE_GPU}                                                      │
  ├──────────────────┼────────────────────────────────────────────────────────────────────┤
  │ Dataset          │ ${DATASET}                                                         │
  └──────────────────┴────────────────────────────────────────────────────────────────────┘

  Settings:
    1. Vanilla SGLang          — (no extra args)
    2. Prefill CUDA Graph      — --enable-piecewise-cuda-graph --disable-cuda-graph
    3. Batch Prefill CUDA Graph— --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph (expanded bs/seq)
    4. Prefill Disaggregation  — --enable-laps-scheduler --laps-length-threshold 256 --laps-max-consecutive-short 1
    5. LAPS (full)             — --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph (expanded) + LAPS N=1
===============================================================================
CONFIGEOF

echo "Config saved to ${BASE_RESULTS_DIR}/config.txt"

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

    echo ""
    echo "============================================================"
    echo "  Launching servers for: ${label}"
    echo "  Model: ${MODEL}, TP=1"
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
        > "${BASE_RESULTS_DIR}/${label}_decode.log" 2>&1 &

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
        > "${BASE_RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  600

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
        # Flush KV cache before each concurrency level
        flush_cache

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
settings = ['vanilla_sglang', 'prefill_cuda_graph', 'batch_prefill_cuda_graph',
            'prefill_disagg', 'laps']
labels = {
    'vanilla_sglang': 'Vanilla SGLang',
    'prefill_cuda_graph': 'Prefill CUDA Graph',
    'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
    'prefill_disagg': 'Prefill Disaggregation',
    'laps': 'LAPS (anti-starvation N=1)',
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

title = 'Qwen2.5-32B, TP=1 — ${run_label}'
print()
print('=' * 140)
print(f'  BENCHMARK SUMMARY — {title}')
print('=' * 140)

for key, section_title, fmt in metrics:
    print()
    print(f'  {section_title}')
    print('-' * 140)
    hdr = f\"{'Setting':<28s}\"
    for cc in ccs:
        hdr += f'  cc={cc:<5d}'
    print(hdr)
    print('-' * 140)

    for s in settings:
        row = f'{labels[s]:<28s}'
        for cc in ccs:
            d = cache.get((s, cc))
            if d and key in d:
                row += '  ' + fmt.format(d[key])
            else:
                row += f'  {\"N/A\":>8s}'
        print(row)

print()
print('=' * 140)
" 2>&1 | tee "${run_dir}/summary.txt"
}

trap cleanup EXIT

# ───────────────────────── main ─────────────────────────

SETTINGS_LABELS=("vanilla_sglang" "prefill_cuda_graph" "batch_prefill_cuda_graph" "prefill_disagg" "laps")
PREFILL_CG_OPTS="--disable-cuda-graph"
BATCH_PREFILL_OPTS="--batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512"
SETTINGS_ARGS=(
    ""
    "--enable-piecewise-cuda-graph $PREFILL_CG_OPTS"
    "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $PREFILL_CG_OPTS $BATCH_PREFILL_OPTS"
    "$LAPS_ARGS"
    "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $PREFILL_CG_OPTS $BATCH_PREFILL_OPTS $LAPS_ARGS"
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
settings = ['vanilla_sglang', 'prefill_cuda_graph', 'batch_prefill_cuda_graph',
            'prefill_disagg', 'laps']
labels = {
    'vanilla_sglang': 'Vanilla SGLang',
    'prefill_cuda_graph': 'Prefill CUDA Graph',
    'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
    'prefill_disagg': 'Prefill Disaggregation',
    'laps': 'LAPS (anti-starvation N=1)',
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

title = f'Qwen2.5-32B, TP=1 — LAPS N=1, KV flush — AGGREGATE ({num_runs} runs)'
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

# Move server logs to subfolder
mkdir -p "${BASE_RESULTS_DIR}/server_logs"
mv "${BASE_RESULTS_DIR}"/*.log "${BASE_RESULTS_DIR}/server_logs/" 2>/dev/null || true

# Generate best summary
$PYTHON "${SCRIPT_DIR}/generate_best_summaries.py" 2>/dev/null || true

echo ""
echo "All runs complete. Results in: ${BASE_RESULTS_DIR}/"
