#!/bin/bash
# Experiment 8: All configs × 2 models (single-server, baseline.py)
#
# 4 server configs:
#   1) vanilla       — no CUDA graph
#   2) piecewise_cg  — piecewise CUDA graph only
#   3) batch_cg      — batch prefill CUDA graph (6×10=60 graphs)
#   4) laps          — batch prefill CG + LAPS scheduler
#                      NOTE: LAPS auto-disables in non-disagg mode,
#                      so results will match batch_cg.
#
# 2 models: Qwen2.5-7B-Instruct, Qwen2.5-32B
# CC: 2, 4, 8, 16, 32, 64
# 10K requests, max_tokens=1, baseline.py (HTTP/2)
#
# Usage: bash bench_exp8_all_configs.sh [GPU_ID]
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
BASE_DIR="${SCRIPT_DIR}/results_exp8_${TODAY}"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=10000
MAX_TOKENS=1
CONCURRENCIES=(2 4 8 16 32 64)

MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-32B")
MODEL_LABELS=("7b" "32b")

# ── Server argument presets ──

declare -A SERVER_ARGS
SERVER_ARGS[vanilla]=""
SERVER_ARGS[piecewise_cg]="--enable-piecewise-cuda-graph --disable-cuda-graph"
SERVER_ARGS[batch_cg]="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512"
SERVER_ARGS[laps]="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512 --enable-laps-scheduler --laps-length-threshold 256 --laps-max-consecutive-short 1"

SETTINGS=(vanilla piecewise_cg batch_cg laps)

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

trap cleanup EXIT

# ── Validate ──
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"; exit 1
fi
if [ ! -f "$BASELINE_PY" ]; then
    echo "ERROR: baseline.py not found: $BASELINE_PY"; exit 1
fi

echo "Experiment 8: All Configs × 2 Models (single-server, baseline.py)"
echo "  GPU:      ${GPU}"
echo "  Port:     ${PORT}"
echo "  Dataset:  ${DATASET}"
echo "  Requests: ${NUM_REQUESTS}"
echo "  CC:       ${CONCURRENCIES[*]}"
echo "  Models:   ${MODELS[*]}"
echo "  Settings: ${SETTINGS[*]}"
echo "  Results:  ${BASE_DIR}/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════

for mi in "${!MODELS[@]}"; do
    MODEL="${MODELS[$mi]}"
    MLABEL="${MODEL_LABELS[$mi]}"

    echo ""
    echo "################################################################"
    echo "  MODEL: ${MODEL} (${MLABEL})"
    echo "################################################################"

    for setting in "${SETTINGS[@]}"; do
        SETTING_DIR="${BASE_DIR}/${MLABEL}/${setting}"
        mkdir -p "$SETTING_DIR"

        echo ""
        echo "============================================================"
        echo "  Launching: ${MLABEL} / ${setting}"
        echo "  GPU: ${GPU}, Model: ${MODEL}"
        echo "  Args: ${SERVER_ARGS[$setting]:-<none>}"
        echo "============================================================"

        cleanup

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m sglang.launch_server \
            --model-path "$MODEL" \
            --host $HOST --port $PORT \
            --mem-fraction-static 0.85 \
            ${SERVER_ARGS[$setting]} \
            > "${SETTING_DIR}/${setting}_server.log" 2>&1 &

        wait_ready "http://${HOST}:${PORT}/health" 600

        # Warmup
        echo "[warmup] 20 requests with cc=4..."
        $PYTHON "$BASELINE_PY" \
            --base-url "http://${HOST}:${PORT}" \
            --model "$MODEL" \
            --concurrency 4 \
            --num-requests 20 \
            --max-tokens $MAX_TOKENS \
            --prompt-file "$DATASET" \
            --output-dir "${SETTING_DIR}/warmup" \
            --http2 \
            > /dev/null 2>&1 || true
        sleep 3

        # Sweep
        for cc in "${CONCURRENCIES[@]}"; do
            RUN_DIR="${SETTING_DIR}/cc${cc}"
            mkdir -p "$RUN_DIR"

            flush_cache

            echo "  --- ${MLABEL}/${setting} | cc=${cc} | N=${NUM_REQUESTS} ---"
            $PYTHON "$BASELINE_PY" \
                --base-url "http://${HOST}:${PORT}" \
                --model "$MODEL" \
                --concurrency $cc \
                --num-requests $NUM_REQUESTS \
                --max-tokens $MAX_TOKENS \
                --prompt-file "$DATASET" \
                --output-dir "$RUN_DIR" \
                --http2 \
                --desc "${MLABEL}_${setting}_cc${cc}" \
                2>&1 | tee "${RUN_DIR}/run.log"
            sleep 3
        done
    done
done

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  SUMMARY — Experiment 8: All Configs × 2 Models"
echo "################################################################"

$PYTHON -c "
import json, os

base_dir = '${BASE_DIR}'
model_labels = ['7b', '32b']
settings = ['vanilla', 'piecewise_cg', 'batch_cg', 'laps']
ccs = [2, 4, 8, 16, 32, 64]

def load(mlabel, setting, cc):
    f = os.path.join(base_dir, mlabel, setting, f'cc{cc}', 'summary.json')
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)

for mlabel in model_labels:
    model_name = 'Qwen2.5-7B-Instruct' if mlabel == '7b' else 'Qwen2.5-32B'
    print()
    print('=' * 100)
    print(f'  MODEL: {model_name} ({mlabel})')
    print('=' * 100)

    # RPS
    print()
    print('  --- RPS (requests/sec) ---')
    hdr = f\"{'Setting':<20s}\"
    for cc in ccs:
        hdr += f'  cc={cc:<5d}'
    print(f'  {hdr}')
    print(f\"  {'-' * len(hdr)}\")
    for s in settings:
        row = f'{s:<20s}'
        for cc in ccs:
            d = load(mlabel, s, cc)
            if d:
                row += f'  {d[\"rps\"]:>8.1f}'
            else:
                row += f'  {\"N/A\":>8s}'
        print(f'  {row}')

    # Prefill throughput
    print()
    print('  --- Prefill Throughput (tok/s) ---')
    print(f'  {hdr}')
    print(f\"  {'-' * len(hdr)}\")
    for s in settings:
        row = f'{s:<20s}'
        for cc in ccs:
            d = load(mlabel, s, cc)
            if d and d['wall_time_sec'] > 0:
                tput = d['tokens']['total_prompt_tokens'] / d['wall_time_sec']
                row += f'  {tput:>8.0f}'
            else:
                row += f'  {\"N/A\":>8s}'
        print(f'  {row}')

    # Latency
    print()
    print('  --- Latency (ms): avg / p50 / p90 / p99 ---')
    for s in settings:
        print(f'    {s}:')
        print(f\"    {'cc':<6s}  {'avg':>8s}  {'p50':>8s}  {'p90':>8s}  {'p99':>8s}\")
        print(f\"    {'-' * 42}\")
        for cc in ccs:
            d = load(mlabel, s, cc)
            if d:
                lat = d['latency_ms']
                print(f'    {cc:<6d}  {lat[\"avg\"]:>8.1f}  {lat[\"p50\"]:>8.1f}  {lat[\"p90\"]:>8.1f}  {lat[\"p99\"]:>8.1f}')
            else:
                print(f\"    {cc:<6d}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}\")
        print()

    # Speedup vs vanilla
    print()
    print('  --- Speedup vs vanilla (RPS ratio) ---')
    print(f'  {hdr}')
    print(f\"  {'-' * len(hdr)}\")
    for s in settings:
        row = f'{s:<20s}'
        for cc in ccs:
            d_v = load(mlabel, 'vanilla', cc)
            d_s = load(mlabel, s, cc)
            if d_v and d_s and d_v['rps'] > 0:
                row += f'  {d_s[\"rps\"]/d_v[\"rps\"]:>7.3f}x'
            else:
                row += f'  {\"N/A\":>8s}'
        print(f'  {row}')

# Fail check
print()
print('=== Fail Count Check ===')
all_ok = True
for mlabel in model_labels:
    for s in settings:
        for cc in ccs:
            d = load(mlabel, s, cc)
            if d and d.get('fail', 0) > 0:
                print(f'  WARNING: {mlabel}/{s}/cc{cc} has {d[\"fail\"]} failures!')
                all_ok = False
if all_ok:
    print('  All runs: 0 failures')

print()
print(f'Done. Results in: ${BASE_DIR}/')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  EXPERIMENT 8 COMPLETE"
echo "  Results: ${BASE_DIR}/"
echo "================================================================"
