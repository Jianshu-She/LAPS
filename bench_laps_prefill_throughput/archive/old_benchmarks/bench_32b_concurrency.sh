#!/bin/bash
# Benchmark prefill throughput on Qwen2.5-32B with concurrency sweep.
#
# 5 settings x 8 concurrency levels = 40 data points.
#
# Usage: bash bench_32b_concurrency.sh
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
DATASET="$(cd "$(dirname "$0")/.." && pwd)/data/lmsys_chat_10k.jsonl"
PREFILL_PORT=30300
DECODE_PORT=30301
ROUTER_PORT=30302
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results_32b"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

NUM_PROMPTS=500
MAX_NEW_TOKENS=1
PREFILL_GPU=0
DECODE_GPU=1

CONCURRENCY_LEVELS="1 2 4 8 16 32 64 128"

mkdir -p "$RESULTS_DIR"

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
    echo "  Prefill args: ${prefill_extra_args:-<none>}"
    echo "============================================================"

    cleanup

    # ── launch decode (GPU $DECODE_GPU) ──
    echo "[launch] Decode server on GPU ${DECODE_GPU}..."
    CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${RESULTS_DIR}/${label}_decode.log" 2>&1 &

    # ── launch prefill (GPU $PREFILL_GPU) ──
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
        > "${RESULTS_DIR}/${label}_prefill.log" 2>&1 &

    wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
    wait_ready "http://${HOST}:${DECODE_PORT}/health"  600

    # ── launch router ──
    echo "[launch] Router..."
    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${RESULTS_DIR}/${label}_router.log" 2>&1 &

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

    for cc in $CONCURRENCY_LEVELS; do
        echo ""
        echo "  --- ${label} | cc=${cc} ---"
        $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
            --dataset "$DATASET" \
            --url "http://${HOST}:${ROUTER_PORT}" \
            --num-prompts $NUM_PROMPTS \
            --max-new-tokens $MAX_NEW_TOKENS \
            --concurrency $cc \
            --output "${RESULTS_DIR}/${label}_cc${cc}.json" \
            2>&1 | tee "${RESULTS_DIR}/${label}_cc${cc}.txt"
        sleep 2
    done
}

trap cleanup EXIT

# ───────────────────────── run 5 settings ─────────────────────────

LAPS_ARGS="--enable-laps-scheduler --laps-length-threshold 256"

launch_servers "vanilla_sglang" ""
run_concurrency_sweep "vanilla_sglang"

launch_servers "prefill_cuda_graph" "--enable-piecewise-cuda-graph"
run_concurrency_sweep "prefill_cuda_graph"

launch_servers "batch_prefill_cuda_graph" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph"
run_concurrency_sweep "batch_prefill_cuda_graph"

launch_servers "prefill_disagg" "$LAPS_ARGS"
run_concurrency_sweep "prefill_disagg"

launch_servers "laps" "--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph $LAPS_ARGS"
run_concurrency_sweep "laps"

# ───────────────────────── summary table ─────────────────────────

echo ""
echo "============================================================"
echo "  GENERATING SUMMARY"
echo "============================================================"

$PYTHON -c "
import json, os, sys

results_dir = '${RESULTS_DIR}'
settings = ['vanilla_sglang', 'prefill_cuda_graph', 'batch_prefill_cuda_graph',
            'prefill_disagg', 'laps']
labels = {
    'vanilla_sglang': 'Vanilla SGLang',
    'prefill_cuda_graph': 'Prefill CUDA Graph',
    'batch_prefill_cuda_graph': 'Batch Prefill CUDA Graph',
    'prefill_disagg': 'Prefill Disaggregation',
    'laps': 'LAPS',
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
    ('total_input_tokens',      'TOTAL INPUT TOKENS',              '{:>8d}'),
    ('completed',               'COMPLETED REQUESTS',              '{:>8d}'),
    ('failed',                  'FAILED REQUESTS',                 '{:>8d}'),
    ('total_duration_s',        'TOTAL DURATION (s)',               '{:>8.1f}'),
]

# Cache all results
cache = {}
for s in settings:
    for cc in ccs:
        f = os.path.join(results_dir, f'{s}_cc{cc}.json')
        if os.path.exists(f):
            with open(f) as fh:
                cache[(s, cc)] = json.load(fh)

title = 'Qwen2.5-32B, 1 prefill GPU (H200)'
print()
print('=' * 110)
print(f'  BENCHMARK SUMMARY — {title}')
print('=' * 110)

for key, section_title, fmt in metrics:
    print()
    print(f'  {section_title}')
    print('-' * 110)
    hdr = f\"{'Setting':<28s}\"
    for cc in ccs:
        hdr += f'  cc={cc:<5d}'
    print(hdr)
    print('-' * 110)

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
print('=' * 110)
" 2>&1 | tee "${RESULTS_DIR}/summary.txt"

echo ""
echo "Full results in: ${RESULTS_DIR}/"
echo "Summary saved to: ${RESULTS_DIR}/summary.txt"
