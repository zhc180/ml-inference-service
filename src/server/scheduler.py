from collections.abc import Iterable
from dataclasses import dataclass

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
        raise NotImplementedError

    def add_sequence(self, sequence: Sequence) -> None:
        """Register a new sequence in WAITING_PREFILL state."""
        raise NotImplementedError

    def pop_prefill_batch(self) -> list[Sequence]:
        """Select the next prefill micro-batch based on policy and budgets."""
        raise NotImplementedError

    def pop_decode_batch(self) -> list[Sequence]:
        """Select the next decode micro-batch based on policy and budgets."""
        raise NotImplementedError

    def on_prefill_complete(self, sequence: Sequence) -> None:
        """Move sequence from prefill phase into decode-ready state."""
        raise NotImplementedError

    def on_decode_step_complete(self, sequences: Iterable[Sequence]) -> None:
        """Update sequence states after one decode step for a batch."""
        raise NotImplementedError

    def on_sequences_finished(self, sequences: Iterable[Sequence]) -> None:
        """Finalize terminal sequences and remove from active queues."""
        raise NotImplementedError

    def preempt(self, sequence_ids: Iterable[str]) -> None:
        """Mark selected sequences as preempted and re-queue if policy allows."""
        raise NotImplementedError

    def has_pending_work(self) -> bool:
        """Return whether scheduler has any sequence still in-flight."""
        raise NotImplementedError
