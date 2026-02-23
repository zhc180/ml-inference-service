import asyncio
from time import perf_counter

from src.optimization.caching import KVCache


class InferenceEngine:
    """Placeholder async inference engine.

    Replace this with your real model loading/token generation flow.
    """

    def __init__(self) -> None:
        self.cache = KVCache(max_entries=1024)

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
