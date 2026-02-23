# LAPS: Length-Aware Prefill Scheduling for LLM Serving

**Accepted to MLSys 2026** | [arXiv](https://arxiv.org/abs/2502.12345)

**Author:** Jianshu She

## Overview

In prefill-decode (PD) disaggregated LLM serving, prefill workers process prompts of widely varying lengths. When short and long prefills share a batch, GPU utilization suffers: short sequences finish early and waste compute while long sequences dominate latency, creating head-of-line blocking that degrades both throughput and time-to-first-token (TTFT).

LAPS introduces a three-level scheduling framework that addresses this interference. **Dual-Queue Scheduling** separates short and long prefills into distinct queues so each batch contains similarly-sized sequences. **Waiting Window** adds a configurable delay to accumulate enough same-length requests for high-utilization batches. **Dynamic Allocation** adjusts the prefill-decode GPU split at runtime based on queue pressure, preventing either pipeline stage from becoming a bottleneck.

![Architecture](docs/figures/architecture.png)

## Features

| Feature | Branch | Description | Docs |
|---|---|---|---|
| Dual-Queue Scheduling | `laps-dual-queue` | Separates prefill requests into short/long queues to eliminate length interference | [docs/laps_scheduler.md](docs/laps_scheduler.md) |
| Waiting Window | `laps-waiting-window` | Accumulates same-length requests before dispatching to maximize batch utilization | [docs/laps_scheduler.md](docs/laps_scheduler.md) |
| Dynamic Allocation | `laps-dynamic-alloc` | Adjusts prefill/decode GPU ratio at runtime based on queue pressure | [docs/laps_scheduler.md](docs/laps_scheduler.md) |

## Benchmark Results

Prefill throughput and median TTFT on **Qwen2.5-32B** with **1 prefill GPU (H200)** across varying concurrency levels:

### Prefill Throughput (tokens/s)

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| No CUDA Graph | 1940 | 3936 | 5839 | 9740 | 16567 | 25754 | 24839 | 27690 |
| Piecewise | 1946 | 4190 | 6225 | 11183 | 19398 | 29867 | 31760 | 29146 |
| Piecewise+BatchPF | 1951 | 4194 | 6210 | 11091 | 18534 | 30633 | 27865 | **41181** |

### Median TTFT (ms)

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| No CUDA Graph | 44.8 | 46.1 | 68.0 | 84.9 | 93.7 | 118.7 | 172.0 | 327.8 |
| Piecewise | 45.5 | 46.2 | 64.5 | 70.0 | 80.1 | 100.7 | 135.2 | 324.4 |
| Piecewise+BatchPF | 45.6 | 46.2 | 64.4 | 69.2 | 82.4 | 95.8 | 149.0 | **292.0** |

## Quick Start

```bash
# Level 1: Dual-Queue Scheduling
python -m sglang.launch_server --model <model> --enable-dual-queue

# Level 2: + Waiting Window
python -m sglang.launch_server --model <model> --enable-dual-queue --waiting-window-ms 50

# Level 3: + Dynamic Allocation
python -m sglang.launch_server --model <model> --enable-dual-queue --waiting-window-ms 50 --enable-dynamic-alloc
```

## Documentation

See [docs/laps_scheduler.md](docs/laps_scheduler.md) for detailed design and configuration options.

## Acknowledgements

Built on [SGLang](https://github.com/sgl-project/sglang). Licensed under Apache 2.0.
