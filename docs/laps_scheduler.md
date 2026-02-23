# LAPS: Length-Aware Prefill Scheduler

Based on [LAPS paper (arxiv 2601.11589)](https://arxiv.org/abs/2601.11589), optimizing PD disaggregation prefill instances.

## Branches

| Branch | Base | What it adds |
|--------|------|-------------|
| `laps-dual-queue` | `batch-prefill-attn-in-graph` | Dual-queue short/long separation |
| `laps-waiting-window` | `laps-dual-queue` | Batch waiting window for short requests |
| `laps-dynamic-alloc` | `laps-dual-queue` | Router-level dynamic GPU allocation |

## Feature 1: Dual-Queue Scheduler (`laps-dual-queue`)

### Problem
A single waiting queue mixes short (memory-bound) and long (compute-bound) prefill requests. Long prefills block short ones, hurting latency.

### Solution
Split into `short_queue` and `long_queue` by token length. Short batches dispatch first; long requests only run when no short requests are pending.

### Files changed
- `python/sglang/srt/server_args.py` — config fields + CLI args + validation
- `python/sglang/srt/disaggregation/prefill.py` — dual-queue logic in `SchedulerDisaggregationPrefillMixin`

### Key methods added to `prefill.py`
| Method | Purpose |
|--------|---------|
| `_init_laps_state()` | Initialize short/long queues and threshold |
| `_laps_classify_requests(reqs)` | Route requests by `len(origin_input_ids)` vs threshold |
| `_laps_rebuild_from_waiting_queue()` | Reconcile sub-queues after aborts modify `waiting_queue` |
| `_laps_sync_waiting_queue()` | Set `waiting_queue = short + long` for abort/metrics compat |
| `_build_batch_from_queue(queue)` | Queue-swap pattern: swap `waiting_queue`, call `get_new_batch_prefill()`, restore |
| `_get_next_laps_batch_to_run()` | Short-first priority scheduling |

### Design: queue-swap pattern
Instead of duplicating the 200+ line `_get_new_batch_prefill_raw()`, we temporarily swap `self.waiting_queue` to point to the target sub-queue, call existing `get_new_batch_prefill()`, then restore. Safe because the event loop is single-threaded.

### Abort handling
The abort handler (`scheduler.py:2751`) directly pops from `self.waiting_queue`. We call `_laps_sync_waiting_queue()` to keep it in sync, and `_laps_rebuild_from_waiting_queue()` at the start of each iteration to filter out aborted requests from sub-queues.

---

## Feature 2: Batch Waiting Window (`laps-waiting-window`)

### Problem
With dual queues, short requests dispatch as soon as one arrives. Under light load, this means batch size = 1 every time, missing CUDA graph batching opportunities.

### Solution
Wait up to a configurable time window for short requests to accumulate. During the window, long requests can still be processed. Dispatch triggers:
1. Window timer expires (`laps_wait_window_ms`)
2. Batch size target reached (`laps_wait_max_batch`)

### Additional methods
| Method | Purpose |
|--------|---------|
| `_laps_should_dispatch_short()` | Check window expiry or batch size target |
| `_laps_is_waiting()` | True if in active waiting window (suppresses idle check) |

---

## Feature 3: Dynamic Multi-GPU Allocation (`laps-dynamic-alloc`)

### Problem
With multiple prefill instances, a static assignment doesn't adapt to workload skew. If most requests are short, the "long" instances sit idle while "short" instances are overloaded.

### Solution
The router (MiniLB) classifies each request by estimated prompt length and routes it to a dedicated prefill group (short or long). Periodically, instances are moved between groups based on outstanding request pressure.

### Architecture
```
Requests → [mini_lb router] → classify short/long → route to group
                                                      ↓
                              short_group: prefill[0], prefill[1]  (dynamic)
                              long_group:  prefill[2]              (dynamic)
                                                      ↓
                              Rebalance every T seconds based on pending request pressure
```

### Files changed
- `sgl-model-gateway/bindings/python/src/sglang_router/router_args.py` — config fields + CLI args
- `sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py` — LAPS routing, classification, rebalancing

### Key functions/methods
| Function/Method | Purpose |
|-----------------|---------|
| `_estimate_prompt_tokens(request_data)` | Estimate token count from input_ids, text, or messages |
| `MiniLoadBalancer._laps_classify(request_data)` | Returns `"short"` or `"long"` based on token count vs threshold |
| `MiniLoadBalancer._laps_maybe_rebalance()` | Time-gated inline check — moves at most 1 instance between groups |
| `MiniLoadBalancer.select_pair_laps(request_data)` | LAPS-aware pair selection: classify, pick from group, track pending |
| `MiniLoadBalancer.laps_request_done(category)` | Decrement pending counter after request completes |

### Rebalancing algorithm
- Checked inline on every `select_pair_laps` call, time-gated to `laps_rebalance_interval_s`
- If `short_pending > long_pending * ratio` and long group has >1 instance: move one long→short
- If `long_pending > short_pending * ratio` and short group has >1 instance: move one short→long
- Each group keeps ≥1 instance. At most 1 move per interval.

### Interaction with per-server dual-queue
Router-level allocation and server-level dual-queue are complementary:
- **Router**: Decides *which server* handles the request (coarse-grained, spatial)
- **Server**: Prioritizes short over long within its local queue (fine-grained, temporal)

### Observability
`GET /laps_status` returns group assignments, pending counts, and configuration.

---

## How to Enable

### Dual-queue only (Branch: `laps-dual-queue`)

```bash
git checkout laps-dual-queue
```

Launch prefill server with:
```bash
python -m sglang.launch_server \
    --model-path <model> \
    --disaggregation-mode prefill \
    --enable-piecewise-cuda-graph \
    --enable-batch-prefill-cuda-graph \
    --enable-laps-scheduler \
    --laps-length-threshold 256
```

### Dual-queue + waiting window (Branch: `laps-waiting-window`)

```bash
git checkout laps-waiting-window
```

Launch prefill server with:
```bash
python -m sglang.launch_server \
    --model-path <model> \
    --disaggregation-mode prefill \
    --enable-piecewise-cuda-graph \
    --enable-batch-prefill-cuda-graph \
    --enable-laps-scheduler \
    --laps-length-threshold 256 \
    --laps-wait-window-ms 10.0 \
    --laps-wait-max-batch 4
```

### Dynamic allocation at router (Branch: `laps-dynamic-alloc`)

```bash
git checkout laps-dynamic-alloc
```

Launch router with 3+ prefill instances:
```bash
python -m sglang_router.launch_router \
    --pd-disaggregation --mini-lb \
    --prefill http://host:30100 --prefill http://host:30200 --prefill http://host:30300 \
    --decode http://host:30400 \
    --enable-laps-alloc \
    --laps-alloc-threshold 256 \
    --laps-rebalance-interval-s 5.0 \
    --laps-rebalance-ratio 2.0 \
    --port 8000
```

Each prefill server (optional: also enable server-level LAPS):
```bash
python -m sglang.launch_server \
    --model-path <model> \
    --disaggregation-mode prefill \
    --enable-laps-scheduler \
    --laps-length-threshold 256 \
    --port 301XX
```

### CLI Arguments Reference

#### Prefill server args

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-laps-scheduler` | flag | `False` | Enable LAPS dual-queue scheduling |
| `--laps-length-threshold` | int | `256` | Token length cutoff: `<= threshold` → short, `> threshold` → long |
| `--laps-wait-window-ms` | float | `5.0` | Max wait time (ms) for short requests to accumulate |
| `--laps-wait-max-batch` | int | `8` | Dispatch immediately when this many short requests are queued |

#### Router args (MiniLB)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable-laps-alloc` | flag | `False` | Enable LAPS dynamic GPU allocation across prefill instances |
| `--laps-alloc-threshold` | int | `256` | Token length cutoff for router classification |
| `--laps-rebalance-interval-s` | float | `5.0` | Min interval (seconds) between rebalancing checks |
| `--laps-rebalance-ratio` | float | `2.0` | Pending ratio threshold to trigger rebalancing |

### Notes
- Server-level LAPS (`--enable-laps-scheduler`) only works in `--disaggregation-mode prefill`. If set in other modes, it auto-disables with a warning.
- Server-level LAPS is disabled for pipeline parallelism (`pp_size > 1`); the PP event loop has its own flow.
- When `--laps-wait-window-ms 0`, the waiting window effectively disables (immediate dispatch), behaving like dual-queue only.
- Router-level LAPS (`--enable-laps-alloc`) requires >=2 prefill instances. With only 1 prefill, it auto-disables.
- Router and server LAPS are complementary: the router does spatial (which server) and the server does temporal (scheduling priority).
