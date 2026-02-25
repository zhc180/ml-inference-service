import asyncio
from time import perf_counter

from src.optimization.caching import KVCache
from src.server.scheduler import Scheduler, SchedulerConfig
from src.server.sequence import Sequence


class InferenceEngine:
    """Placeholder async inference engine.

    Replace this with your real model loading/token generation flow.
    """

    def __init__(self) -> None:
        self.cache = KVCache(max_entries=1024)
        self.scheduler: Scheduler | None = None

    async def generate(self, prompt: str, max_new_tokens: int = 64) -> dict:
        started = perf_counter()

        cache_key = f"{prompt}:{max_new_tokens}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {
                "text": cached,
                "cached": True,
                "latency_ms": round((perf_counter() - started) * 1000, 2),
            }

        await asyncio.sleep(0.05)
        text = f"[stubbed-generation] {prompt}"
        self.cache.put(cache_key, text)

        return {
            "text": text,
            "cached": False,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }

    def configure_scheduler(self, config: SchedulerConfig) -> None:
        """Create scheduler instance used by the internal engine loop."""
        raise NotImplementedError

    async def submit_sequence(self, prompt: str, max_new_tokens: int) -> str:
        """Tokenize input and enqueue a new sequence; return request ID."""
        raise NotImplementedError

    async def run_engine_step(self) -> None:
        """Execute one engine tick: schedule, prefill/decode, and bookkeeping."""
        raise NotImplementedError

    async def _run_prefill(self, batch: list[Sequence]) -> None:
        """Execute model prefill for a scheduled prefill batch."""
        raise NotImplementedError

    async def _run_decode(self, batch: list[Sequence]) -> None:
        """Execute one decode token step for a scheduled decode batch."""
        raise NotImplementedError

    async def _finalize_finished(self, sequences: list[Sequence]) -> None:
        """Collect outputs and free scheduler/block resources for terminal sequences."""
        raise NotImplementedError
