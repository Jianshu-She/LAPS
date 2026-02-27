# LAPS: Length-Aware Prefill Scheduling for LLM Serving

**Accepted to MLSys 2026** | [arXiv](https://arxiv.org/abs/2601.11589)

**Authors:** Jianshu She, Zonghang Li, Hongchao Du, Shangyu Wu, Wenhao Zheng, Eric Xing, Zhengzhong Liu, Huaxiu Yao, Jason Xue, Qirong Ho

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

All experiments run under **PD disaggregation** (Mooncake KV transfer backend) with **500 requests** from LMSYS-Chat-1M (`max_new_tokens=1`).

### Qwen2.5-32B — 1 Prefill GPU (H200)

**Prefill Throughput (tokens/s)**

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| Vanilla SGLang | 1884 | 3626 | 5688 | 9539 | 15344 | 24254 | 29834 | 30172 |
| Prefill CUDA Graph | 1942 | 4257 | 6203 | 11191 | 20665 | 28060 | 23002 | 40471 |
| Batch Prefill CUDA Graph | 1927 | 4206 | 6240 | 11517 | 18483 | 29868 | 23693 | 40975 |
| Prefill Disaggregation | 1951 | 4097 | 5752 | 8727 | 14899 | 22301 | 24032 | 37307 |
| **LAPS** | 1922 | 4256 | 5993 | 10100 | 16763 | 27515 | 21357 | **40938** |

**Median TTFT (ms)**

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| Vanilla SGLang | 46.1 | 52.2 | 68.5 | 88.9 | 101.9 | 123.2 | 187.7 | 339.0 |
| Prefill CUDA Graph | 45.7 | 45.7 | 64.6 | 68.1 | 72.1 | 103.7 | 168.2 | 297.8 |
| Batch Prefill CUDA Graph | 46.0 | 46.2 | 64.7 | 67.5 | 82.4 | 99.3 | 151.9 | 276.2 |
| Prefill Disaggregation | 44.5 | 46.0 | 67.7 | 85.0 | 101.2 | 120.7 | 184.1 | 305.9 |
| **LAPS** | 46.2 | 45.8 | 65.2 | 76.1 | 90.4 | 100.3 | 156.7 | **258.7** |

### Qwen2.5-72B-Instruct — TP=2, 1 Prefill Instance (2x H200)

**Prefill Throughput (tokens/s)**

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| Vanilla SGLang | 1401 | 2485 | 4191 | 6344 | 9839 | 17332 | 19698 | 33544 |
| Prefill CUDA Graph | 1598 | 3544 | 5254 | 9618 | 17572 | 24118 | 25670 | 36453 |
| Batch Prefill CUDA Graph | 1600 | 3545 | 5326 | 9444 | 16074 | 25569 | 25763 | 37516 |
| Prefill Disaggregation | 1412 | 2421 | 3298 | 4779 | 7068 | 11672 | 20707 | 20790 |
| **LAPS** | 1598 | 3510 | 5058 | 8561 | 13941 | 22796 | 27708 | 31264 |

**Median TTFT (ms)**

| Setting | cc=1 | cc=2 | cc=4 | cc=8 | cc=16 | cc=32 | cc=64 | cc=128 |
|---|---|---|---|---|---|---|---|---|
| Vanilla SGLang | 64.3 | 75.3 | 98.7 | 133.3 | 171.6 | 184.5 | 246.9 | 344.6 |
| Prefill CUDA Graph | 54.8 | 54.4 | 76.4 | 81.3 | 85.9 | 126.3 | 180.7 | 314.8 |
| Batch Prefill CUDA Graph | 54.8 | 54.6 | 75.6 | 81.5 | 93.5 | 116.8 | 158.7 | 310.0 |
| Prefill Disaggregation | 63.4 | 82.5 | 136.3 | 149.8 | 155.8 | 172.6 | 214.2 | 418.7 |
| **LAPS** | 55.0 | 55.0 | 76.9 | 91.4 | 101.6 | 119.5 | **147.1** | **171.0** |

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

- [docs/laps_scheduler.md](docs/laps_scheduler.md) — Detailed design and configuration options
- [docs/code_changes.md](docs/code_changes.md) — Code changes vs vanilla SGLang (with attention-in-graph deep dive)
- [bench_laps_prefill_throughput/README.md](bench_laps_prefill_throughput/README.md) — Test and benchmark scripts

## Acknowledgements

Built on [SGLang](https://github.com/sgl-project/sglang). Licensed under Apache 2.0.
