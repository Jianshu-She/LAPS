#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# Benchmark: Batch Prefill CUDA Graph Hit Rate Analysis
# Tests at different concurrency levels and reports hit/miss statistics
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="Qwen/Qwen2.5-32B"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET="${SCRIPT_DIR}/../data/lmsys_chat_10k.jsonl"
TODAY=$(date +%Y-%m-%d)
BASE_RESULTS_DIR="${SCRIPT_DIR}/results_bp_hitrate_${TODAY}"
PYTHON="/mnt/weka/home/jianshu.she/miniconda3/envs/graph/bin/python"

PREFILL_PORT=30310
DECODE_PORT=30311
ROUTER_PORT=30312
HOST="127.0.0.1"
IB_DEVICE="mlx5_0"
BACKEND="mooncake"
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}

NUM_PROMPTS=10000
MAX_NEW_TOKENS=1

# Concurrency levels to test
CONCURRENCY_LEVELS="1 2 4 8 16 32 64 128"

# Batch prefill CG settings to test
BATCH_PREFILL_OPTS="--batch-prefill-batch-sizes 1 2 4 8 16 32 --batch-prefill-seq-lengths 1 2 4 8 16 32 64 128 256 512 --batch-prefill-max-seq-len 512"

mkdir -p "$BASE_RESULTS_DIR"

# ───────────────────────── record config ─────────────────────────
cat > "${BASE_RESULTS_DIR}/config.txt" <<CONFIGEOF
===============================================================================
  BATCH PREFILL CG HIT RATE ANALYSIS — $(date '+%Y-%m-%d %H:%M:%S')
===============================================================================
  Model:             ${MODEL}
  Dataset:           ${DATASET}
  NUM_PROMPTS:       ${NUM_PROMPTS}
  MAX_NEW_TOKENS:    ${MAX_NEW_TOKENS}
  Prefill GPU:       ${PREFILL_GPU}
  Decode GPU:        ${DECODE_GPU}
  Concurrency:       ${CONCURRENCY_LEVELS}
  Batch Prefill:     ${BATCH_PREFILL_OPTS}
  CG settings:       --enable-piecewise-cuda-graph --enable-batch-prefill-cuda-graph --disable-cuda-graph
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
    echo "[flush] Flushing KV cache (also triggers BP-CG stats dump)..."
    curl -s -X POST "http://${HOST}:${PREFILL_PORT}/flush_cache" > /dev/null 2>&1 || true
    curl -s -X POST "http://${HOST}:${DECODE_PORT}/flush_cache" > /dev/null 2>&1 || true
    sleep 2
}

trap cleanup EXIT

PREFILL_LOG="${BASE_RESULTS_DIR}/prefill_server.log"
DECODE_LOG="${BASE_RESULTS_DIR}/decode_server.log"
ROUTER_LOG="${BASE_RESULTS_DIR}/router.log"

# ───────────────────────── launch servers ─────────────────────────

echo ""
echo "============================================================"
echo "  Launching servers with Batch Prefill CUDA Graph"
echo "  Model: ${MODEL}, TP=1"
echo "  Prefill GPU: ${PREFILL_GPU}, Decode GPU: ${DECODE_GPU}"
echo "============================================================"

cleanup

echo "[launch] Decode server on GPU ${DECODE_GPU}..."
CUDA_VISIBLE_DEVICES=$DECODE_GPU $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend $BACKEND \
    --disaggregation-ib-device $IB_DEVICE \
    --disaggregation-bootstrap-port 9310 \
    --host $HOST --port $DECODE_PORT \
    --mem-fraction-static 0.85 \
    > "$DECODE_LOG" 2>&1 &

echo "[launch] Prefill server on GPU ${PREFILL_GPU}..."
CUDA_VISIBLE_DEVICES=$PREFILL_GPU $PYTHON -m sglang.launch_server \
    --model-path "$MODEL" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend $BACKEND \
    --disaggregation-ib-device $IB_DEVICE \
    --disaggregation-bootstrap-port 9311 \
    --host $HOST --port $PREFILL_PORT \
    --mem-fraction-static 0.85 \
    --enable-piecewise-cuda-graph \
    --enable-batch-prefill-cuda-graph \
    --disable-cuda-graph \
    $BATCH_PREFILL_OPTS \
    > "$PREFILL_LOG" 2>&1 &

wait_ready "http://${HOST}:${PREFILL_PORT}/health" 600
wait_ready "http://${HOST}:${DECODE_PORT}/health" 600

echo "[launch] Router..."
$PYTHON -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill "http://${HOST}:${PREFILL_PORT}" 9311 \
    --decode  "http://${HOST}:${DECODE_PORT}" \
    --host $HOST --port $ROUTER_PORT \
    > "$ROUTER_LOG" 2>&1 &

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

# Flush after warmup to reset stats baseline
flush_cache

# ───────────────────────── run benchmark ─────────────────────────

echo ""
echo "============================================================"
echo "  Running hit rate benchmark at different concurrency levels"
echo "============================================================"

for cc in $CONCURRENCY_LEVELS; do
    echo ""
    echo "  --- cc=${cc} ---"

    # Record line count before benchmark to find new BP-CG-STATS lines
    LINES_BEFORE=$(wc -l < "$PREFILL_LOG" 2>/dev/null || echo 0)

    $PYTHON "${SCRIPT_DIR}/bench_prefill_only.py" \
        --dataset "$DATASET" \
        --url "http://${HOST}:${ROUTER_PORT}" \
        --num-prompts $NUM_PROMPTS \
        --max-new-tokens $MAX_NEW_TOKENS \
        --concurrency $cc \
        --seed 42 \
        --output "${BASE_RESULTS_DIR}/bench_cc${cc}.json" \
        2>&1 | tee "${BASE_RESULTS_DIR}/bench_cc${cc}.txt"

    sleep 2

    # Flush cache to trigger stats dump and reset
    flush_cache

    # Extract BP-CG-STATS lines from this run
    LINES_AFTER=$(wc -l < "$PREFILL_LOG" 2>/dev/null || echo 0)
    echo ""
    echo "  [BP-CG hit rate stats for cc=${cc}]"
    tail -n +$((LINES_BEFORE + 1)) "$PREFILL_LOG" | grep "BP-CG-STATS" || echo "  (no stats lines found)"
    echo ""

    sleep 2
done

# ───────────────────────── generate summary ─────────────────────────

echo ""
echo "============================================================"
echo "  Generating Hit Rate Summary"
echo "============================================================"

$PYTHON -c "
import re, json, os

results_dir = '${BASE_RESULTS_DIR}'
prefill_log = '${PREFILL_LOG}'
ccs = [int(x) for x in '${CONCURRENCY_LEVELS}'.split()]

# Parse BP-CG-STATS-FLUSH lines from the log
stats_lines = []
if os.path.exists(prefill_log):
    with open(prefill_log) as f:
        for line in f:
            if 'BP-CG-STATS-FLUSH' in line:
                stats_lines.append(line.strip())

# Parse periodic BP-CG-STATS lines
periodic_lines = []
if os.path.exists(prefill_log):
    with open(prefill_log) as f:
        for line in f:
            if 'BP-CG-STATS]' in line and 'FLUSH' not in line:
                periodic_lines.append(line.strip())

print()
print('=' * 80)
print('  BATCH PREFILL CUDA GRAPH HIT RATE ANALYSIS')
print('=' * 80)
print()

# Parse flush stats
print('  Stats from flush_cache (cumulative):')
print('-' * 80)
for line in stats_lines:
    # Extract key metrics
    m = re.search(r'eligible=(\d+)\s+hit=(\d+)\s+miss=(\d+)\s+hit_rate=([\d.]+)%\s+skip_bs_le_1=(\d+)\s+miss_bs_large=(\d+)\s+miss_seq_long=(\d+)', line)
    if m:
        eligible, hit, miss = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hit_rate = float(m.group(4))
        skip_bs1 = int(m.group(5))
        miss_bs = int(m.group(6))
        miss_seq = int(m.group(7))
        print(f'  eligible={eligible:>6d}  hit={hit:>6d}  miss={miss:>6d}  hit_rate={hit_rate:>5.1f}%  skip_bs<=1={skip_bs1:>5d}  miss_bs_large={miss_bs:>4d}  miss_seq_long={miss_seq:>4d}')

print()
print('  Periodic stats (every 500 eligible calls):')
print('-' * 80)
for line in periodic_lines:
    # Extract the useful part
    idx = line.find('[BP-CG-STATS]')
    if idx >= 0:
        print(f'  {line[idx:]}')

# Also load benchmark results
print()
print('  Benchmark throughput by concurrency:')
print('-' * 80)
print(f\"{'cc':>6s}  {'throughput':>12s}  {'req/s':>8s}  {'median_ttft':>12s}\")
print('-' * 80)
for cc in ccs:
    jf = os.path.join(results_dir, f'bench_cc{cc}.json')
    if os.path.exists(jf):
        with open(jf) as f:
            d = json.load(f)
        thr = d.get('prefill_throughput_tok_s', 0)
        rps = d.get('request_throughput_req_s', 0)
        ttft = d.get('median_ttft_ms', 0)
        print(f'{cc:>6d}  {thr:>12.0f}  {rps:>8.1f}  {ttft:>12.1f}')

print()
print('=' * 80)
" 2>&1 | tee "${BASE_RESULTS_DIR}/hitrate_summary.txt"

echo ""
echo "All done. Results in: ${BASE_RESULTS_DIR}/"
echo "  - Server logs: ${BASE_RESULTS_DIR}/"
echo "  - Hit rate summary: ${BASE_RESULTS_DIR}/hitrate_summary.txt"
