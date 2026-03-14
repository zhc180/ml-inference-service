import asyncio
from contextlib import AbstractContextManager
from time import perf_counter

from src.monitoring.metrics import observe_decode_step, observe_prefill_step
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
        self.scheduler = Scheduler(config)

    async def submit_sequence(self, prompt: str, max_new_tokens: int) -> str:
        """Tokenize input and enqueue a new sequence; return request ID."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not configured")

        sequence = Sequence(prompt=prompt, max_new_tokens=max_new_tokens)
        self.scheduler.add_sequence(sequence)
        return sequence.id

    async def run_engine_step(self) -> None:
        """Execute one engine tick: schedule, prefill/decode, and bookkeeping."""
        # Example:
        # prefill_batch = self.scheduler.pop_prefill_batch()
        # if prefill_batch:
        #     prefill_tokens = sum(len(seq.prompt_token_ids) for seq in prefill_batch)
        #     with self._observe_prefill_step(tokens=prefill_tokens, batch_size=len(prefill_batch)):
        #         await self._run_prefill(prefill_batch)
        #
        # decode_batch = self.scheduler.pop_decode_batch()
        # if decode_batch:
        #     decode_tokens = len(decode_batch)  # one token/seq for single-step decode
        #     with self._observe_decode_step(tokens=decode_tokens, batch_size=len(decode_batch)):
        #         await self._run_decode(decode_batch)
        prefill_batch = self.scheduler.pop_prefill_batch() if self.scheduler else []
        if prefill_batch:
            prefill_tokens = sum(len(seq.prompt_token_ids) for seq in prefill_batch)
            with self._observe_prefill_step(tokens=prefill_tokens, batch_size=len(prefill_batch)):
                await self._run_prefill(prefill_batch)

        decode_batch = self.scheduler.pop_decode_batch() if self.scheduler else []
        if decode_batch:
            decode_tokens = len(decode_batch)  # one token/seq for single-step decode
            with self._observe_decode_step(tokens=decode_tokens, batch_size=len(decode_batch)):
                await self._run_decode(decode_batch)

        

    async def _run_prefill(self, batch: list[Sequence]) -> None:
        """Execute model prefill for a scheduled prefill batch."""
        # Example:
        # tokenizer -> model forward(prefill) -> write KV blocks -> scheduler.on_prefill_complete
        # Keep this method model-centric and avoid scheduler policy logic here.
        raise NotImplementedError

    async def _run_decode(self, batch: list[Sequence]) -> None:
        """Execute one decode token step for a scheduled decode batch."""
        # Example:
        # model forward(decode one step) -> append KV -> scheduler.on_decode_step_complete
        # Then identify finished sequences and call _finalize_finished.
        raise NotImplementedError

    async def _finalize_finished(self, sequences: list[Sequence]) -> None:
        """Collect outputs and free scheduler/block resources for terminal sequences."""
        raise NotImplementedError

    def _observe_prefill_step(self, tokens: int, batch_size: int) -> AbstractContextManager[None]:
        """Context manager helper for prefill step metrics."""
        return observe_prefill_step(tokens=tokens, batch_size=batch_size)

    def _observe_decode_step(self, tokens: int, batch_size: int) -> AbstractContextManager[None]:
        """Context manager helper for decode step metrics."""
        return observe_decode_step(tokens=tokens, batch_size=batch_size)
