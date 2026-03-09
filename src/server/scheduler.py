from collections import deque
from contextlib import AbstractContextManager
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from src.monitoring.metrics import (
    inc_preemptions,
    observe_decode_step,
    observe_prefill_step,
    observe_queue_wait,
    set_active_sequences,
)
from src.server.sequence import Sequence, SequenceStatus


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
        self._config = config
        self._waiting_prefill: deque[Sequence] = deque()   # new requests
        self._waiting_decode: deque[Sequence] = deque()    # prefill done, waiting decode turn
        self._active_sequences: dict[str, Sequence] = {}   # request_id → seq, all in-flight
        self._publish_active_sequences(0)

    def add_sequence(self, sequence: Sequence) -> None:
        """Register a new sequence in WAITING_PREFILL state."""
        self._waiting_prefill.append(sequence)
        self._active_sequences[sequence.request_id] = sequence

        sequence.status = SequenceStatus.WAITING_PREFILL

        self._publish_active_sequences(len(self._active_sequences))
        

    def pop_prefill_batch(self) -> list[Sequence]:
        """Select the next prefill micro-batch based on policy and budgets."""
        token_budget = self._config.max_prefill_tokens_per_step
        batch = []
        now = perf_counter()
        while self._waiting_prefill and len(batch) < self._config.max_batch_size:
            seq = self._waiting_prefill[0]
            if len(seq.prompt_token_ids) > token_budget:
                break
            token_budget -= len(seq.prompt_token_ids)
            batch.append(self._waiting_prefill.popleft())
            seq.status = SequenceStatus.RUNNING_PREFILL
            self._record_queue_wait("prefill", now - seq.created_at)

        return batch

    def pop_decode_batch(self) -> list[Sequence]:
        """Select the next decode micro-batch based on policy and budgets."""
        batch = []
        now = perf_counter()
        while (self._waiting_decode
               and len(batch) < self._config.max_batch_size
               and len(batch) < self._config.max_decode_tokens_per_step):
            seq = self._waiting_decode[0]
            # only as a guard, we expect sequences with zero remaining decode budget to have been marked finished and removed already
            if seq.remaining_decode_budget() == 0:
                self._waiting_decode.popleft()
                self._active_sequences.pop(seq.request_id, None)
                seq.mark_finished()
                continue
            batch.append(self._waiting_decode.popleft())
            seq.status = SequenceStatus.RUNNING_DECODE
            self._record_queue_wait("decode", now - seq.created_at)

        return batch

    def on_prefill_complete(self, sequence: Sequence) -> None:
        """Move sequence from prefill phase into decode-ready state."""
        sequence.status = SequenceStatus.WAITING_DECODE
        self._waiting_decode.append(sequence)

    def on_decode_step_complete(self, sequences: Iterable[Sequence]) -> None:
        """Update sequence states after one decode step for a batch."""
        for seq in sequences:
            if seq.remaining_decode_budget() == 0:
                seq.mark_finished()
            else:
                seq.status = SequenceStatus.WAITING_DECODE
                self._waiting_decode.append(seq)

    def on_sequences_finished(self, sequences: Iterable[Sequence]) -> None:
        """Finalize terminal sequences and remove from active queues."""
        # Example: self._publish_active_sequences(len(self._active_sequences))
        for seq in sequences:
            seq.mark_finished()
            self._active_sequences.pop(seq.request_id, None)
        self._publish_active_sequences(len(self._active_sequences))

    def preempt(self, sequence_ids: Iterable[str]) -> None:
        """Mark selected sequences as preempted and re-queue if policy allows."""
        # Example: self._record_preemptions(count=len(preempted_ids))
        raise NotImplementedError

    def has_pending_work(self) -> bool:
        """Return whether scheduler has any sequence still in-flight."""
        return bool(self._active_sequences)

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
