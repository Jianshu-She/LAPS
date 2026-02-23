# Code Changes: LAPS vs Vanilla SGLang

This document describes all code modifications made to SGLang to implement the LAPS (Length-Aware Prefill Scheduling) system. Changes span three features: batch prefill CUDA graph v2, dual-queue scheduling, and dynamic GPU allocation.

---

## Summary of Modified Files

| File | Feature | What Changed |
|------|---------|-------------|
| `python/sglang/srt/server_args.py` | All | New CLI args and config fields |
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | Batch Prefill v2 | Graph capture/replay with attention inside |
| `python/sglang/srt/layers/attention/flashattention_backend.py` | Batch Prefill v2 | Stable-pointer buffer management for FA metadata |
| `python/sglang/srt/layers/attention/base_attn_backend.py` | Batch Prefill v2 | Abstract interface for batch prefill support |
| `python/sglang/srt/model_executor/forward_batch_info.py` | Batch Prefill v2 | `padded_static_len` field |
| `python/sglang/srt/disaggregation/prefill.py` | Dual-Queue | Short/long queue separation in event loop |
| `sgl-model-gateway/.../sglang_router/mini_lb.py` | Dynamic Alloc | Request classification, group rebalancing |
| `sgl-model-gateway/.../sglang_router/router_args.py` | Dynamic Alloc | Router CLI args |

---

## Feature 1: Batch Prefill CUDA Graph v2 (Attention Inside the Graph)

### Background: The Problem with Piecewise CUDA Graphs

Vanilla SGLang uses **piecewise** CUDA graphs for prefill: the model is split into segments at each attention layer, and separate graphs are captured for the non-attention parts. Attention itself runs **outside** the graph because its metadata (sequence lengths, page tables, cumulative offsets) changes every batch — CUDA graphs require fixed tensor addresses.

This means every prefill step incurs CPU-side kernel launch overhead at each attention layer boundary. For short prefills (the common case in PD disaggregation), this overhead dominates.

### Solution: Monolithic Graph with Stable-Pointer Buffers

Batch prefill v2 captures the **entire** forward pass — including attention — inside a single CUDA graph. The key idea: pre-allocate fixed-address buffers for all attention metadata, and update them **in-place** before each replay so the pointers captured in the graph remain valid.

### The Right-Alignment Insight

When batching sequences of different lengths into a CUDA graph, they must be padded to a uniform `seq_len`. The critical question is: where do the real tokens go within each padded slot?

**Flash attention uses a bottom-right aligned causal mask.** For `flash_attn_with_kvcache`, query position `p` can attend to KV position `j` where:

```
j <= p + cache_seqlens - seqlen_q
```

**Left-alignment breaks this.** If a 10-token sequence is placed at positions `0..9` in a 64-token slot (`seqlen_q=64`), then the mask condition becomes `j <= p + cache_seqlens - 64`. Since `cache_seqlens ≈ 10` for a new prefill, this yields `j <= p - 54`, which is negative for the first 54 positions — the real tokens are **masked out**.

**Right-alignment fixes this.** Placing the 10-token sequence at positions `54..63` means position 63 (the last real token) can see `j <= 63 + 10 - 64 = 9`, which covers the full KV cache. The causal mask works correctly.

```
Left-aligned (BROKEN):         Right-aligned (CORRECT):
[real real real ... pad pad]    [pad pad ... real real real]
 0    1    2       62  63        0   1       61   62   63
 ↑ masked out by causal mask     ↑ padding    ↑ correctly visible
```

### File-by-File Changes

#### `piecewise_cuda_graph_runner.py`

**New fields in `__init__` (lines 287-345):**
- `self.batch_prefill_enabled` — feature flag from server_args
- Pre-allocated padded buffers: `bp_input_ids`, `bp_positions`, `bp_out_cache_loc`, `bp_seq_lens`, `bp_req_pool_indices`, `bp_extend_seq_lens` — sized for `max_bs * max_seq_len`
- `self.batch_prefill_graphs: Dict[(bs, seq_len), CUDAGraph]` — captured graphs
- `self.batch_prefill_outputs: Dict[(bs, seq_len), output]` — captured output tensors

**`can_run_batch_prefill(forward_batch)` (lines 786-814):**
Checks if an incoming batch qualifies:
- `batch_size > 1` (single sequences use the existing decode graph)
- `forward_mode == EXTEND` (prefill, not decode)
- Not multimodal, no logprobs
- A matching captured graph exists via `_find_batch_prefill_graph()`

**`_find_batch_prefill_graph(actual_bs, max_extend_len)` (lines 815-844):**
Finds the smallest captured `(bs, seq_len)` where `bs >= actual_bs` and `seq_len >= max_extend_len`.

**`capture_one_batch_prefill(bs, seq_len, stream)` (lines 861-982):**
1. Slices pre-allocated buffers to `(bs, seq_len)` shape
2. Creates a `ForwardBatch` with uniform `seq_lens = seq_len` and `padded_static_len = seq_len`
3. Calls `attn_backend.init_forward_metadata_capture_batch_prefill_cuda_graph()` to set up stable-pointer FA metadata
4. Defines `run_once()` — a full forward pass **without** piecewise context (attention runs normally inside)
5. Runs 2 warmup iterations, then captures with `torch.cuda.graph()`
6. Stores graph and output tensor

**`replay_batch_prefill(forward_batch)` (lines 984-1069):**
1. **Right-aligns** real tokens into padded slots:
   ```python
   dst_offset = i * target_seq_len + (target_seq_len - actual_len)
   bp_input_ids[dst_offset:dst_offset+actual_len].copy_(real_input_ids)
   bp_positions[dst_offset:dst_offset+actual_len].copy_(real_positions)
   bp_out_cache_loc[dst_offset:dst_offset+actual_len].copy_(real_cache_locs)
   ```
2. Sets `extend_seq_lens = target_seq_len` for all sequences — LogitsProcessor reads the **last** position of each slot, which with right-alignment is the last real token
3. Pads dummy sequences (beyond `actual_bs`) with `seq_lens=1` and a valid `req_pool_index` (first real sequence's index) so the page table lookup doesn't fault
4. Calls `attn_backend.init_forward_metadata_replay_batch_prefill_cuda_graph()` to update metadata **in-place**
5. Replays the graph: `self.batch_prefill_graphs[(target_bs, target_seq_len)].replay()`
6. Slices output to `[:actual_bs]`

#### `flashattention_backend.py`

**`init_batch_prefill_cuda_graph_state(max_bs, max_seq_len)` (lines 2197-2222):**
Pre-allocates buffers that will be captured inside the graph:
- `cache_seqlens` — `(max_bs,)` int32, KV cache length per sequence
- `cu_seqlens_k` — `(max_bs+1,)` int32, cumulative sum of cache_seqlens
- `page_table` — `(max_bs, max_num_pages)` int32, page indices for paged attention
- `strided_indices` — pre-computed page stride pattern
- `cu_seqlens_q` — stored per `(bs, seq_len)` key, fixed after capture

**`init_forward_metadata_capture_batch_prefill_cuda_graph(bs, seq_len, ...)` (lines 2224-2276):**
Creates a `FlashAttentionMetadata` object using the pre-allocated buffers. Sets `max_seq_len_q = seq_len` (uniform). Computes `cu_seqlens_q` once and stores it per key.

**`init_forward_metadata_replay_batch_prefill_cuda_graph(bs, seq_len, ...)` (lines 2278-2322):**
Updates the **same** buffer objects in-place:
- `cache_seqlens.copy_(new_seq_lens)` — pointer unchanged, values updated
- `cu_seqlens_k[1:].copy_(cumsum(...))` — same
- `page_table[:, :max_pages].copy_(new_page_indices)` — same

The CUDA graph captured pointers to these buffers during capture. Since we update values without reallocating, the graph sees the new data.

#### `base_attn_backend.py`

Three new abstract methods:
- `supports_batch_prefill_cuda_graph()` → `False` by default
- `init_batch_prefill_cuda_graph_state(...)` → raises `NotImplementedError`
- `init_forward_metadata_replay_batch_prefill_cuda_graph(...)` → raises `NotImplementedError`

Only FlashAttention backend overrides these; FlashInfer and other backends return `False`.

#### `forward_batch_info.py`

New field `padded_static_len: int = -1`. When set to a positive value, indicates the batch uses uniform padded sequence length for batch prefill graph capture.

#### `server_args.py`

New config fields:
```python
enable_batch_prefill_cuda_graph: bool = False
batch_prefill_max_seq_len: int = 256        # max sequence length to capture
batch_prefill_batch_sizes: List[int]         # default [1, 2, 4, 8]
batch_prefill_seq_lengths: List[int]         # default [16, 32, 64, 128, 256]
```

CLI arguments: `--enable-batch-prefill-cuda-graph`, `--batch-prefill-max-seq-len`, `--batch-prefill-batch-sizes`, `--batch-prefill-seq-lengths`.

---

## Feature 2: LAPS Dual-Queue Scheduler

### Background

In PD disaggregation, prefill workers process prompts of widely varying lengths. A single FIFO queue causes head-of-line blocking: short requests wait behind long ones, inflating TTFT for short requests.

### Solution

Split the waiting queue into two sub-queues by token length. Short requests are served with higher priority, reducing their latency without starving long requests.

### File Changes

#### `disaggregation/prefill.py` (lines 361-532)

All methods are added to `SchedulerDisaggregationPrefillMixin`:

| Method | Purpose |
|--------|---------|
| `_init_laps_state()` | Create `_laps_short_queue`, `_laps_long_queue`, set threshold |
| `_laps_classify_requests(reqs)` | Route new requests: `len(origin_input_ids) <= threshold` → short, else long |
| `_laps_rebuild_from_waiting_queue()` | After abort handlers modify `waiting_queue`, filter aborted requests from sub-queues |
| `_laps_sync_waiting_queue()` | Set `waiting_queue = short + long` for compatibility with abort handlers and metrics |
| `_build_batch_from_queue(queue)` | **Queue-swap pattern**: temporarily replace `self.waiting_queue` with the target sub-queue, call the existing `get_new_batch_prefill()` (200+ lines of batch building logic), then restore. Safe because the event loop is single-threaded. |
| `_get_next_laps_batch_to_run()` | Short-first priority: try short queue, fall back to long queue |

**Modified event loops:**
- `event_loop_normal_disagg_prefill()` (line 436): if `enable_laps_scheduler`, uses the dual-queue flow instead of the default single-queue
- `event_loop_overlap_disagg_prefill()` (line 474): same dual-queue integration for the overlap scheduler variant

#### `server_args.py`

New config fields:
```python
enable_laps_scheduler: bool = False
laps_length_threshold: int = 256    # tokens; <= threshold → short
```

Validation: auto-disables with warning if `disaggregation_mode != "prefill"` or `pp_size > 1`.

CLI arguments: `--enable-laps-scheduler`, `--laps-length-threshold`.

---

## Feature 3: LAPS Dynamic GPU Allocation (Router Level)

### Background

With multiple prefill instances, a static short/long assignment wastes capacity when the workload shifts. If most requests are short, the "long" instances sit idle.

### Solution

The router classifies each incoming request by estimated token count and routes it to a dedicated prefill group (short or long). Periodically, it moves instances between groups based on pending request pressure.

### File Changes

#### `sglang_router/mini_lb.py`

**LAPS initialization in `__init__` (lines 73-97):**
- Splits prefill instances: first half → short group, second half → long group
- Requires >=2 prefill instances; auto-disables with 1
- State: `_laps_short_group`, `_laps_long_group` (lists of instance indices), `_laps_short_pending`, `_laps_long_pending` (counters), `_laps_last_rebalance` (timestamp)

| Method | Purpose |
|--------|---------|
| `_estimate_prompt_tokens(request_data)` | Estimate token count from `input_ids`, text length, or messages |
| `_laps_classify(request_data)` | Returns `"short"` or `"long"` based on estimated tokens vs threshold |
| `_laps_maybe_rebalance()` | Time-gated (every `rebalance_interval_s`). If `short_pending > long_pending * ratio` and long group has >1 instance: move one long→short. Vice versa. Each group keeps >=1 instance. At most 1 move per interval. |
| `select_pair_laps(request_data)` | LAPS-aware routing: classify request, pick instance from matching group, increment pending counter |
| `laps_request_done(category)` | Decrement pending counter after request completes |

**`/laps_status` endpoint** returns: `enabled`, `threshold`, `short_group`, `long_group`, `short_pending`, `long_pending`.

#### `sglang_router/router_args.py`

New config fields:
```python
enable_laps_alloc: bool = False
laps_alloc_threshold: int = 256          # tokens
laps_rebalance_interval_s: float = 5.0   # seconds between rebalance checks
laps_rebalance_ratio: float = 2.0        # pending ratio to trigger move
```

CLI arguments: `--enable-laps-alloc`, `--laps-alloc-threshold`, `--laps-rebalance-interval-s`, `--laps-rebalance-ratio`.

---

## How the Three Features Interact

```
                     ┌─────────────────────────────┐
    Requests ──────►│  Router (Feature 3)          │
                     │  Classify short/long         │
                     │  Route to prefill group      │
                     │  Rebalance groups over time  │
                     └──────┬──────────┬────────────┘
                            │          │
                    ┌───────▼──┐  ┌────▼───────┐
                    │ Prefill  │  │ Prefill    │
                    │ Server A │  │ Server B   │
                    │          │  │            │
                    │ Feature 2│  │ Feature 2  │
                    │ Dual-Q   │  │ Dual-Q     │
                    │ short>   │  │ short>long │
                    │ long     │  │            │
                    │          │  │            │
                    │ Feature 1│  │ Feature 1  │
                    │ Batch PF │  │ Batch PF   │
                    │ CUDA Grph│  │ CUDA Graph │
                    └──────────┘  └────────────┘
```

- **Feature 3 (router)**: Spatial — decides *which server* handles each request
- **Feature 2 (scheduler)**: Temporal — decides *scheduling priority* within a server
- **Feature 1 (CUDA graph)**: Execution — accelerates the actual prefill computation

The three levels are complementary and can be enabled independently.
