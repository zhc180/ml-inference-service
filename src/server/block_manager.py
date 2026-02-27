from dataclasses import dataclass, field

from src.monitoring.metrics import set_kv_block_usage
from src.server.sequence import Sequence


@dataclass
class KVBlock:
    """Represents one KV-cache block in paged attention layout."""

    block_id: int
    token_capacity: int
    used_tokens: int = 0
    ref_count: int = 0
    hash_key: str | None = None
    sequence_ids: set[str] = field(default_factory=set)


class BlockManager:
    """KV-cache block allocator/recycler interface.

    Notes for implementation:
    - Keep an explicit free-list for O(1) pop/push.
    - Separate logical sequence->block mapping from physical block storage.
    - Add a deterministic preemption/eviction policy for reproducibility.
    """

    def __init__(self, num_blocks: int, block_token_capacity: int) -> None:
        """Initialize block pool metadata and free lists."""
        # Example: self._publish_kv_usage(used_blocks=0, total_blocks=num_blocks)
        raise NotImplementedError

    def allocate_for_prefill(self, sequence: Sequence, num_tokens: int) -> list[int]:
        """Allocate enough blocks to hold prefilling `num_tokens`."""
        # Example: self._publish_kv_usage(used_blocks=current_used, total_blocks=self._num_blocks)
        raise NotImplementedError

    def append_for_decode(self, sequence: Sequence, num_tokens: int = 1) -> list[int]:
        """Append decode tokens, extending sequence block table if needed."""
        # Example: self._publish_kv_usage(used_blocks=current_used, total_blocks=self._num_blocks)
        raise NotImplementedError

    def free_sequence(self, sequence: Sequence) -> None:
        """Release blocks owned by a completed/aborted sequence."""
        # Example: self._publish_kv_usage(used_blocks=current_used, total_blocks=self._num_blocks)
        raise NotImplementedError

    def try_reuse_prefix(self, sequence: Sequence) -> bool:
        """Attempt hashed prefix-block reuse; return True on cache hit."""
        raise NotImplementedError

    def evict_or_preempt(self, required_blocks: int) -> list[str]:
        """Free enough blocks for `required_blocks`; return affected sequence IDs."""
        # Example: self._publish_kv_usage(used_blocks=current_used, total_blocks=self._num_blocks)
        raise NotImplementedError

    def available_block_count(self) -> int:
        """Expose current free block count for scheduler decisions/metrics."""
        raise NotImplementedError

    def _publish_kv_usage(self, used_blocks: int, total_blocks: int) -> None:
        """Publish KV block usage and utilization ratio."""
        set_kv_block_usage(used_blocks=used_blocks, total_blocks=total_blocks)
