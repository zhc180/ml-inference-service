from contextlib import AbstractContextManager
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from src.monitoring.metrics import (
    inc_preemptions,
    observe_decode_step,
    observe_prefill_step,
    observe_queue_wait,
    set_active_sequences,
)
from src.server.sequence import Sequence


@dataclass
class SchedulerConfig:
    """Scheduling limits used to build each engine step."""

    max_batch_size: int = 8
    max_prefill_tokens_per_step: int = 2048
    max_decode_tokens_per_step: int = 512
    max_active_sequences: int = 64


class Scheduler:
    """Prefill/decode scheduler interface.

    Notes for implementation:
    - Maintain separate waiting/running queues for prefill and decode.
    - Treat prefill and decode budgeting independently each step.
    - Keep state transitions explicit to simplify debugging/metrics.
    """

    def __init__(self, config: SchedulerConfig) -> None:
        """Initialize queues and sequence index structures."""
        # Example: self._publish_active_sequences(0)
        raise NotImplementedError

    def add_sequence(self, sequence: Sequence) -> None:
        """Register a new sequence in WAITING_PREFILL state."""
        # Example: self._publish_active_sequences(len(self._active_sequences))
        raise NotImplementedError

    def pop_prefill_batch(self) -> list[Sequence]:
        """Select the next prefill micro-batch based on policy and budgets."""
        # Example:
        # self._record_queue_wait("prefill", wait_seconds)
        # with self._observe_prefill_step(tokens=prefill_tokens, batch_size=len(batch)):
        #     ... run prefill ...
        raise NotImplementedError

    def pop_decode_batch(self) -> list[Sequence]:
        """Select the next decode micro-batch based on policy and budgets."""
        # Example:
        # self._record_queue_wait("decode", wait_seconds)
        # with self._observe_decode_step(tokens=decode_tokens, batch_size=len(batch)):
        #     ... run decode ...
        raise NotImplementedError

    def on_prefill_complete(self, sequence: Sequence) -> None:
        """Move sequence from prefill phase into decode-ready state."""
        raise NotImplementedError

    def on_decode_step_complete(self, sequences: Iterable[Sequence]) -> None:
        """Update sequence states after one decode step for a batch."""
        raise NotImplementedError

    def on_sequences_finished(self, sequences: Iterable[Sequence]) -> None:
        """Finalize terminal sequences and remove from active queues."""
        # Example: self._publish_active_sequences(len(self._active_sequences))
        raise NotImplementedError

    def preempt(self, sequence_ids: Iterable[str]) -> None:
        """Mark selected sequences as preempted and re-queue if policy allows."""
        # Example: self._record_preemptions(count=len(preempted_ids))
        raise NotImplementedError

    def has_pending_work(self) -> bool:
        """Return whether scheduler has any sequence still in-flight."""
        raise NotImplementedError

    def _record_queue_wait(
        self,
        phase: Literal["prefill", "decode"],
        wait_seconds: float,
    ) -> None:
        """Emit scheduler queue wait metric for one sequence scheduling event."""
        observe_queue_wait(phase=phase, wait_seconds=wait_seconds)

    def _observe_prefill_step(self, tokens: int, batch_size: int) -> AbstractContextManager[None]:
        """Context manager helper to instrument one prefill step."""
        return observe_prefill_step(tokens=tokens, batch_size=batch_size)

    def _observe_decode_step(self, tokens: int, batch_size: int) -> AbstractContextManager[None]:
        """Context manager helper to instrument one decode step."""
        return observe_decode_step(tokens=tokens, batch_size=batch_size)

    def _publish_active_sequences(self, count: int) -> None:
        """Publish active sequence gauge after scheduler state transitions."""
        set_active_sequences(count)

    def _record_preemptions(self, count: int = 1) -> None:
        """Increment preemption counter after preemption decisions."""
        inc_preemptions(count=count)
