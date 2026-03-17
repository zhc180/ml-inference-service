import asyncio
import sys
import types
import unittest
from typing import Optional


class _FakeMetric:
    def inc(self, count: int = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, value: float) -> None:
        pass

    def labels(self, **_: object) -> "_FakeMetric":
        return self


class _FakeAPIRouter:
    def get(self, _: str):
        def decorator(func):
            return func

        return decorator


class _FakeResponse:
    def __init__(self, content: Optional[bytes] = None, media_type: Optional[str] = None) -> None:
        self.content = content
        self.media_type = media_type


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    loaded_model_id = None

    @classmethod
    def from_pretrained(cls, model_id: str) -> "_FakeTokenizer":
        cls.loaded_model_id = model_id
        return cls()

    def __call__(self, prompt: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        del add_special_tokens
        tokens = [len(part) for part in prompt.split()]
        return {"input_ids": tokens or [0]}


class _FakeModel:
    loaded_model_id = None

    @classmethod
    def from_pretrained(cls, model_id: str) -> "_FakeModel":
        cls.loaded_model_id = model_id
        return cls()


fastapi = types.ModuleType("fastapi")
fastapi.APIRouter = _FakeAPIRouter
fastapi.Response = _FakeResponse
sys.modules.setdefault("fastapi", fastapi)

prometheus_client = types.ModuleType("prometheus_client")
prometheus_client.CONTENT_TYPE_LATEST = "text/plain"
prometheus_client.Counter = lambda *args, **kwargs: _FakeMetric()
prometheus_client.Gauge = lambda *args, **kwargs: _FakeMetric()
prometheus_client.Histogram = lambda *args, **kwargs: _FakeMetric()
prometheus_client.generate_latest = lambda: b""
sys.modules.setdefault("prometheus_client", prometheus_client)

transformers = types.ModuleType("transformers")
transformers.AutoTokenizer = _FakeTokenizer
transformers.AutoModelForCausalLM = _FakeModel
sys.modules.setdefault("transformers", transformers)

caching = types.ModuleType("src.optimization.caching")


class _FakeKVCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max_entries


caching.KVCache = _FakeKVCache
sys.modules.setdefault("src.optimization.caching", caching)

from src.server.inference import InferenceEngine
from src.server.scheduler import SchedulerConfig


class InferenceEngineTests(unittest.TestCase):
    def test_engine_eagerly_loads_small_llama_model(self) -> None:
        engine = InferenceEngine()

        self.assertEqual(_FakeTokenizer.loaded_model_id, "meta-llama/Llama-3.2-3B")
        self.assertEqual(_FakeModel.loaded_model_id, "meta-llama/Llama-3.2-3B")
        self.assertIs(engine.model.__class__, _FakeModel)

    def test_submit_sequence_tokenizes_and_enqueues_request(self) -> None:
        engine = InferenceEngine()
        engine.configure_scheduler(SchedulerConfig())

        request_id = asyncio.run(engine.submit_sequence("hello world", max_new_tokens=3))

        assert engine.scheduler is not None
        sequence = engine.scheduler._active_sequences[request_id]
        self.assertEqual(sequence.request_id, request_id)
        self.assertEqual(sequence.prompt, "hello world")
        self.assertEqual(sequence.max_new_tokens, 3)
        self.assertEqual(sequence.prompt_token_ids, [5, 5])


if __name__ == "__main__":
    unittest.main()
