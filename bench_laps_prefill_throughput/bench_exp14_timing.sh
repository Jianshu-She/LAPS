#!/bin/bash
# Experiment 14a: Profiling Timing Instrumentation
#
# Quick profiling run at cc=32 with timing instrumentation to measure:
#   - Per-forward-call duration (CG path vs eager path)
#   - CG prep overhead (padding + buffer copies) vs CG replay time
#   - Eager path: init_forward_metadata vs model.forward
#
# Only 2 configs: vanilla + batch_cg_v2, N=2000
#
# Usage: bash bench_exp14_timing.sh [GPU_ID]
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
BASE_DIR="${SCRIPT_DIR}/results_exp14a_${TODAY}"
mkdir -p "$BASE_DIR"

NUM_REQUESTS=2000
MAX_TOKENS=1
MODEL="Qwen/Qwen2.5-7B-Instruct"

BATCH_CG_V2_ARGS="--enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph --batch-prefill-batch-sizes 1 2 4 8 --batch-prefill-seq-lengths 8 16 32 64 128 --batch-prefill-max-seq-len 128"

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

run_bench() {
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

trap cleanup EXIT

# ── Validate ──
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"; exit 1
fi
if [ ! -f "$BASELINE_PY" ]; then
    echo "ERROR: baseline.py not found: $BASELINE_PY"; exit 1
fi

echo "Experiment 14a: Profiling Timing (7B, single GPU)"
echo "  GPU:       ${GPU}"
echo "  Port:      ${PORT}"
echo "  Requests:  ${NUM_REQUESTS}"
echo "  Results:   ${BASE_DIR}/"
echo ""

# ═══════════════════════════════════════════════════════════════════════
#  1. Vanilla at cc=32
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  vanilla (cc=32)"
echo "################################################################"

SDIR="${BASE_DIR}/vanilla"
mkdir -p "$SDIR"
launch_server "$SDIR" "vanilla"
warmup "$SDIR"
run_bench "$SDIR" 32 "exp14a_vanilla"

# ═══════════════════════════════════════════════════════════════════════
#  2. batch_cg_v2 at cc=32
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  batch_cg_v2 (cc=32)"
echo "################################################################"

SDIR="${BASE_DIR}/batch_cg_v2"
mkdir -p "$SDIR"
launch_server "$SDIR" "batch_cg_v2" $BATCH_CG_V2_ARGS
warmup "$SDIR"
run_bench "$SDIR" 32 "exp14a_batch_cg_v2"

# ═══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  RESULTS"
echo "################################################################"

$PYTHON -c "
import json, os

base = '${BASE_DIR}'

def load(setting, cc):
    path = os.path.join(base, setting, f'cc{cc}', 'summary.json')
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

for setting in ['vanilla', 'batch_cg_v2']:
    d = load(setting, 32)
    if d:
        print(f'{setting:20s}  RPS={d[\"rps\"]:.1f}  avg_latency={d[\"latency_ms\"][\"avg\"]:.1f}ms  fail={d.get(\"fail\", 0)}')
    else:
        print(f'{setting:20s}  N/A')
" 2>&1 | tee "${BASE_DIR}/summary.txt"

echo ""
echo "================================================================"
echo "  Now analyze server logs for FWD-TIME and BP-CG-TIMING lines:"
echo "    grep 'FWD-TIME\|BP-CG-TIMING' ${BASE_DIR}/*/batch_cg_v2_server.log"
echo "    grep 'FWD-TIME' ${BASE_DIR}/*/vanilla_server.log"
echo "================================================================"
