#!/bin/bash
# Debug experiments 2, 3, 4 (Exp 1 already completed)
#
# Experiment 2: Uniform-length dataset (batch_prefill_cg vs piecewise, cc=8,32)
# Experiment 3: 500 prompts with cache flush (batch_prefill_cg vs vanilla, all cc)
# Experiment 4: 10K prompts with cache flush (batch_prefill_cg vs vanilla, cc=8,128)
#
# Usage: bash bench_debug_exp234.sh
set -uo pipefail  # no -e so we don't abort on server crashes

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

check_servers_alive() {
    curl -s "http://${HOST}:${PREFILL_PORT}/health" > /dev/null 2>&1 && \
    curl -s "http://${HOST}:${DECODE_PORT}/health" > /dev/null 2>&1 && \
    curl -s "http://${HOST}:${ROUTER_PORT}/health" > /dev/null 2>&1
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

    CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
        --model-path "$MODEL" \
        --disaggregation-mode decode \
        --disaggregation-transfer-backend $BACKEND \
        --disaggregation-ib-device $IB_DEVICE \
        --disaggregation-bootstrap-port 9300 \
        --host $HOST --port $DECODE_PORT \
        --mem-fraction-static 0.85 \
        > "${log_dir}/${label}_decode.log" 2>&1 &

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

    $PYTHON -m sglang_router.launch_router \
        --pd-disaggregation --mini-lb \
        --prefill "http://${HOST}:${PREFILL_PORT}" 9301 \
        --decode  "http://${HOST}:${DECODE_PORT}" \
        --host $HOST --port $ROUTER_PORT \
        > "${log_dir}/${label}_router.log" 2>&1 &

    wait_ready "http://${HOST}:${ROUTER_PORT}/health" 60

    # warmup
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

    # Check if servers are still alive
    if ! check_servers_alive; then
        echo "  [WARN] Servers appear down, skipping ${label} cc=${cc}"
        return 1
    fi

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
# EXPERIMENT 2: Uniform-length dataset
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

# 2a: batch_prefill_cg
echo ""
echo "  --- Exp 2a: batch_prefill_cg with uniform dataset ---"
launch_servers "exp2a_batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP2_DIR"

for cc in 8 32; do
    run_bench "uniform_batch_prefill_cg" "$UNIFORM_DATASET" $UNIFORM_COUNT $cc "$EXP2_DIR" 42 false
done

# 2b: piecewise_cg
echo ""
echo "  --- Exp 2b: piecewise_cg with uniform dataset ---"
launch_servers "exp2b_piecewise_cg" "$PIECEWISE_CG_ARGS" "$EXP2_DIR"

for cc in 8 32; do
    run_bench "uniform_piecewise_cg" "$UNIFORM_DATASET" $UNIFORM_COUNT $cc "$EXP2_DIR" 42 false
done

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: 500 prompts with cache flush
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
launch_servers "exp3a_batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP3_DIR"

for cc in 1 2 4 8 16 32 64 128; do
    run_bench "flush_batch_prefill_cg" "$LMSYS_10K" 500 $cc "$EXP3_DIR" 42 true
done

# 3b: vanilla_sglang
echo ""
echo "  --- Exp 3b: vanilla_sglang, 500 prompts, flush ---"
launch_servers "exp3b_vanilla_sglang" "" "$EXP3_DIR"

for cc in 1 2 4 8 16 32 64 128; do
    run_bench "flush_vanilla_sglang" "$LMSYS_10K" 500 $cc "$EXP3_DIR" 42 true
done

# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: 10K prompts with cache flush
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "################################################################"
echo "  EXPERIMENT 4: 10K prompts with flush (cross-check)"
echo "################################################################"

EXP4_DIR="${BASE_RESULTS_DIR}/exp4_10k_flush"
mkdir -p "$EXP4_DIR"

# 4a: vanilla_sglang (servers still running from exp3b)
echo ""
echo "  --- Exp 4a: vanilla_sglang, 10K prompts, flush ---"
for cc in 8 128; do
    run_bench "flush_vanilla_sglang" "$LMSYS_10K" 10000 $cc "$EXP4_DIR" 42 true
done

# 4b: batch_prefill_cg
echo ""
echo "  --- Exp 4b: batch_prefill_cg, 10K prompts, flush ---"
launch_servers "exp4b_batch_prefill_cg" "$BATCH_PREFILL_CG_ARGS" "$EXP4_DIR"

for cc in 8 128; do
    run_bench "flush_batch_prefill_cg" "$LMSYS_10K" 10000 $cc "$EXP4_DIR" 42 true
done

# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "================================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results: ${BASE_RESULTS_DIR}/"
echo "================================================================"

# Print summary tables
$PYTHON -c "
import json, os

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def tput(d):
    return d.get('prefill_throughput_tok_s', 0) if d else 0

base = '${BASE_RESULTS_DIR}'

print()
print('=== Exp 2: Uniform dataset (batch_prefill_cg vs piecewise_cg) ===')
for cc in [8, 32]:
    bp = load(f'{base}/exp2_uniform/uniform_batch_prefill_cg_cc{cc}.json')
    pw = load(f'{base}/exp2_uniform/uniform_piecewise_cg_cc{cc}.json')
    print(f'  cc={cc}: batch_prefill_cg={tput(bp):.0f} tok/s, piecewise_cg={tput(pw):.0f} tok/s')

print()
print('=== Exp 3: 500 prompts with flush ===')
for label_short, label_full in [('BP-CG', 'flush_batch_prefill_cg'), ('Vanilla', 'flush_vanilla_sglang')]:
    print(f'  {label_short}:')
    for cc in [1, 2, 4, 8, 16, 32, 64, 128]:
        d = load(f'{base}/exp3_500_flush/{label_full}_cc{cc}.json')
        print(f'    cc={cc:<4d} → {tput(d):>8.0f} tok/s')

print()
print('=== Exp 4: 10K prompts with flush ===')
for cc in [8, 128]:
    bp = load(f'{base}/exp4_10k_flush/flush_batch_prefill_cg_cc{cc}.json')
    vn = load(f'{base}/exp4_10k_flush/flush_vanilla_sglang_cc{cc}.json')
    print(f'  cc={cc}: batch_prefill_cg={tput(bp):.0f} tok/s, vanilla={tput(vn):.0f} tok/s')
" 2>&1 | tee "${BASE_RESULTS_DIR}/summary_exp234.txt"
