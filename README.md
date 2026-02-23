# ML Inference Service

Project 3 from your 8-week portfolio plan.

This repository is a starter scaffold for building an optimized LLM inference API with:
- FastAPI async serving
- Dynamic batching
- INT8 quantization hooks
- KV-cache management hooks
- Prometheus metrics
- Load testing with Locust

## Planned Milestones

- Week 5: server setup, batching, quantization basics
- Week 6: KV-cache optimization, streaming, observability, load tests

## Project Structure

```text
ml-inference-service/
├── README.md
├── pyproject.toml
├── .gitignore
├── src/
│   ├── server/
│   │   ├── app.py
│   │   ├── inference.py
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

### Test Generate Endpoint

```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello model","max_new_tokens":32}'
```

## Next Build Steps

1. Replace placeholder inference logic in `src/server/inference.py`.
2. Wire continuous batching behavior in `src/server/batching.py`.
3. Add real model quantization path in `src/optimization/quantization.py`.
4. Add Prometheus scraping + Grafana dashboard panels.
5. Run and record load-test benchmarks in `benchmarks/results.md`.
