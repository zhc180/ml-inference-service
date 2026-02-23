from contextlib import contextmanager
from time import perf_counter

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

metrics_router = APIRouter()

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests",
)


@contextmanager
def observe_request():
    REQUEST_COUNT.inc()
    start = perf_counter()
    try:
        yield
    finally:
        REQUEST_LATENCY.observe(perf_counter() - start)


@metrics_router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
