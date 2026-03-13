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

from src.server.scheduler import Scheduler, SchedulerConfig
from src.server.sequence import Sequence, SequenceStatus


class SchedulerTests(unittest.TestCase):
    def test_oversized_prefill_request_does_not_block_following_work(self) -> None:
        scheduler = Scheduler(
            SchedulerConfig(max_batch_size=4, max_prefill_tokens_per_step=2),
        )
        oversized = Sequence(
            request_id="too-big",
            prompt="oversized",
            max_new_tokens=4,
            prompt_token_ids=[1, 2, 3],
        )
        small = Sequence(
            request_id="small",
            prompt="small",
            max_new_tokens=4,
            prompt_token_ids=[1],
        )

        scheduler.add_sequence(oversized)
        scheduler.add_sequence(small)

        batch = scheduler.pop_prefill_batch()

        self.assertEqual([seq.request_id for seq in batch], ["small"])
        self.assertEqual(oversized.status, SequenceStatus.ABORTED)
        self.assertNotIn("too-big", scheduler._active_sequences)

    def test_add_sequence_rejects_duplicate_request_ids(self) -> None:
        scheduler = Scheduler(SchedulerConfig())
        first = Sequence("req-1", "hello", 4, prompt_token_ids=[1])
        duplicate = Sequence("req-1", "world", 4, prompt_token_ids=[2])

        scheduler.add_sequence(first)

        with self.assertRaises(ValueError):
            scheduler.add_sequence(duplicate)

    def test_on_prefill_complete_requires_running_prefill(self) -> None:
        scheduler = Scheduler(SchedulerConfig())
        sequence = Sequence("req-1", "hello", 4, prompt_token_ids=[1])
        scheduler.add_sequence(sequence)

        with self.assertRaises(ValueError):
            scheduler.on_prefill_complete(sequence)

    def test_on_sequences_finished_removes_waiting_decode_entries(self) -> None:
        scheduler = Scheduler(SchedulerConfig())
        sequence = Sequence("req-1", "hello", 4, prompt_token_ids=[1])

        scheduler.add_sequence(sequence)
        prefill_batch = scheduler.pop_prefill_batch()

        self.assertEqual(prefill_batch, [sequence])

        scheduler.on_prefill_complete(sequence)
        scheduler.on_sequences_finished([sequence])

        self.assertEqual(scheduler.pop_decode_batch(), [])
        self.assertFalse(scheduler.has_pending_work())


if __name__ == "__main__":
    unittest.main()
