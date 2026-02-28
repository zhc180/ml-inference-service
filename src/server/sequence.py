from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter


class SequenceStatus(str, Enum):
    """Lifecycle states for a generation request."""

    WAITING_PREFILL = "waiting_prefill"
    RUNNING_PREFILL = "running_prefill"
    WAITING_DECODE = "waiting_decode"
    RUNNING_DECODE = "running_decode"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Sequence:
    """State container for one request being served by the engine.

    This class is intentionally lightweight so you can evolve fields as your
    scheduler and block manager solidify.
    """

    request_id: str
    prompt: str
    max_new_tokens: int
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    kv_block_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING_PREFILL
    created_at: float = field(default_factory=perf_counter)

    def total_tokens(self) -> int:
        """Return prompt + generated token count for this sequence."""
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def remaining_decode_budget(self) -> int:
        """Return how many decode tokens can still be generated."""
        return max(self.max_new_tokens - len(self.generated_token_ids), 0)

    def mark_finished(self) -> None:
        """Transition sequence state to terminal finished."""
        self.status = SequenceStatus.FINISHED

    def mark_aborted(self) -> None:
        """Transition sequence state to terminal aborted."""
        self.status = SequenceStatus.ABORTED
