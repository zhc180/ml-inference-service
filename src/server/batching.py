from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class InferenceRequest:
    prompt: str
    max_new_tokens: int
    metadata: dict[str, Any] | None = None


class DynamicBatcher:
    """Queue-based placeholder for continuous batching.

    Integrate this with your model forward pass to process grouped requests.
    """

    def __init__(self, max_batch_size: int = 8) -> None:
        self.max_batch_size = max_batch_size
        self._queue: deque[InferenceRequest] = deque()

    def enqueue(self, request: InferenceRequest) -> None:
        self._queue.append(request)

    def pop_batch(self) -> list[InferenceRequest]:
        batch: list[InferenceRequest] = []
        while self._queue and len(batch) < self.max_batch_size:
            batch.append(self._queue.popleft())
        return batch
