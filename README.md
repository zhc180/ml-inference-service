# ML Inference Service

A lightweight, production-oriented LLM inference server built from the ground up in Python. Designed as a readable, extensible alternative to projects like [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), with an emphasis on explicit scheduler design, observable internals, and a clean separation between memory management, scheduling, and serving.

## Features

- **FastAPI async HTTP serving** — `/generate` and `/health` endpoints with Pydantic request validation
- **Prefill/decode scheduler** — separate queues with configurable token and batch budgets per step
- **Paged KV-cache block manager** — ref-counted block allocation with FIFO eviction and prefix reuse hooks
- **Dynamic batching** — continuous batching across concurrent requests
- **INT8 quantization hooks** — pluggable quantization path (bitsandbytes / ONNX / TensorRT-LLM)
- **Prometheus metrics** — latency histograms, queue wait, block utilization, preemption counters
- **Grafana dashboard** — pre-built panels for key inference metrics
- **Load testing** — Locust scenarios for throughput and latency benchmarking
- **Docker + Kubernetes** — containerized deployment with a k8s manifest

## Architecture

```
HTTP Request
    │
    ▼
app.py          FastAPI routes (/generate, /health, /metrics)
    │
    ▼
inference.py    InferenceEngine — orchestrates scheduler and block manager
    ├── scheduler.py     Prefill/decode queue management with budget enforcement
    ├── sequence.py      Per-request state machine (WAITING_PREFILL → FINISHED)
    └── block_manager.py Paged KV-cache allocator with ref-counting and eviction

optimization/
    ├── caching.py       Prompt-level response cache
    └── quantization.py  INT8 quantization entry point

monitoring/
    └── metrics.py       Prometheus counters, histograms, and gauges
```

## Project Structure

```text
ml-inference-service/
├── README.md
├── pyproject.toml
├── src/
│   ├── server/
│   │   ├── app.py
│   │   ├── inference.py
│   │   ├── scheduler.py
│   │   ├── sequence.py
│   │   ├── block_manager.py
│   │   └── batching.py
│   ├── optimization/
│   │   ├── quantization.py
│   │   └── caching.py
│   └── monitoring/
│       └── metrics.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── k8s/
│   └── deployment.yaml
├── load_tests/
│   └── locustfile.py
├── grafana/
│   └── dashboard.json
└── benchmarks/
    └── results.md
```

## Quick Start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn src.server.app:app --reload --port 8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Generate

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Explain paged attention", "max_new_tokens": 64}'
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

### Load Testing

```bash
locust -f load_tests/locustfile.py --host http://localhost:8000
```

## Design Decisions

**Explicit state machines.** Each request is a `Sequence` object that moves through well-defined states (`WAITING_PREFILL → RUNNING_PREFILL → WAITING_DECODE → RUNNING_DECODE → FINISHED`). State transitions are never implicit.

**Separate prefill and decode budgets.** The scheduler enforces independent token and batch size limits for prefill and decode phases each engine step, following the pattern established by vLLM's continuous batching design.

**Ref-counted block manager.** KV-cache blocks use reference counting to support prefix sharing across sequences. Blocks are only returned to the free pool when their ref count reaches zero, enabling safe reuse without copies.

**Observable by default.** Prometheus metrics are instrumented at the scheduler, block manager, and engine loop level — queue wait times, block utilization, preemption counts, and step latencies are all tracked out of the box.

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request for significant changes.

## License

MIT
