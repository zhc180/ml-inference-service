from contextlib import contextmanager
from time import perf_counter
from typing import Literal

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

metrics_router = APIRouter()

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests",
)
PREFILL_TOKENS = Counter(
    "inference_prefill_tokens_total",
    "Total prefill tokens processed by the engine",
)
DECODE_TOKENS = Counter(
    "inference_decode_tokens_total",
    "Total decode tokens processed by the engine",
)
PREFILL_STEPS = Counter(
    "inference_prefill_steps_total",
    "Total prefill steps executed",
)
DECODE_STEPS = Counter(
    "inference_decode_steps_total",
    "Total decode steps executed",
)
PREFILL_STEP_LATENCY = Histogram(
    "inference_prefill_step_latency_seconds",
    "Latency of one prefill step",
)
DECODE_STEP_LATENCY = Histogram(
    "inference_decode_step_latency_seconds",
    "Latency of one decode step",
)
PREFILL_BATCH_SIZE = Histogram(
    "inference_prefill_batch_size",
    "Number of sequences processed per prefill step",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
DECODE_BATCH_SIZE = Histogram(
    "inference_decode_batch_size",
    "Number of sequences processed per decode step",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)
QUEUE_WAIT = Histogram(
    "inference_scheduler_queue_wait_seconds",
    "Queue wait time before a sequence is scheduled",
    labelnames=("phase",),
)
ACTIVE_SEQUENCES = Gauge(
    "inference_active_sequences",
    "Current number of active (non-terminal) sequences",
)
PREEMPTIONS = Counter(
    "inference_preemptions_total",
    "Total number of scheduler preemption events",
)
KV_BLOCKS_TOTAL = Gauge(
    "inference_kv_blocks_total",
    "Total KV-cache blocks available",
)
KV_BLOCKS_USED = Gauge(
    "inference_kv_blocks_used",
    "KV-cache blocks currently in use",
)
KV_BLOCK_UTILIZATION = Gauge(
    "inference_kv_block_utilization_ratio",
    "Used/total KV-cache block ratio",
)


@contextmanager
def observe_request():
    REQUEST_COUNT.inc()
    start = perf_counter()
    try:
        yield
    finally:
        REQUEST_LATENCY.observe(perf_counter() - start)


@contextmanager
def observe_prefill_step(tokens: int, batch_size: int):
    """Record one prefill step latency and throughput units."""
    start = perf_counter()
    try:
        yield
    finally:
        PREFILL_STEPS.inc()
        PREFILL_TOKENS.inc(max(tokens, 0))
        PREFILL_BATCH_SIZE.observe(max(batch_size, 0))
        PREFILL_STEP_LATENCY.observe(perf_counter() - start)


@contextmanager
def observe_decode_step(tokens: int, batch_size: int):
    """Record one decode step latency and throughput units."""
    start = perf_counter()
    try:
        yield
    finally:
        DECODE_STEPS.inc()
        DECODE_TOKENS.inc(max(tokens, 0))
        DECODE_BATCH_SIZE.observe(max(batch_size, 0))
        DECODE_STEP_LATENCY.observe(perf_counter() - start)


def observe_queue_wait(phase: Literal["prefill", "decode"], wait_seconds: float) -> None:
    """Record queue wait time for a sequence before scheduling."""
    QUEUE_WAIT.labels(phase=phase).observe(max(wait_seconds, 0.0))


def set_active_sequences(count: int) -> None:
    """Publish current active-sequence count from the scheduler."""
    ACTIVE_SEQUENCES.set(max(count, 0))


def inc_preemptions(count: int = 1) -> None:
    """Increment preemption counter by number of preemption events."""
    PREEMPTIONS.inc(max(count, 0))


def set_kv_block_usage(used_blocks: int, total_blocks: int) -> None:
    """Publish KV-cache block usage and derived utilization ratio."""
    bounded_total = max(total_blocks, 0)
    bounded_used = max(min(used_blocks, bounded_total), 0)
    KV_BLOCKS_TOTAL.set(bounded_total)
    KV_BLOCKS_USED.set(bounded_used)
    if bounded_total == 0:
        KV_BLOCK_UTILIZATION.set(0.0)
    else:
        KV_BLOCK_UTILIZATION.set(bounded_used / bounded_total)


@metrics_router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
