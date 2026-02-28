from collections import deque
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
        self._publish_kv_usage(used_blocks=0, total_blocks=num_blocks)
        
        self._num_blocks = num_blocks
        self._block_token_capacity = block_token_capacity
        self._blocks = [KVBlock(block_id=i, token_capacity=block_token_capacity) for i in range(num_blocks)]
        self._free_block_ids = deque(range(num_blocks))
    
    def _allocate_blocks(self, sequence: Sequence, num_tokens: int) -> list[int]:
        """Allocate enough blocks to hold `num_tokens` for the given sequence."""
        block_needed = (num_tokens + self._block_token_capacity - 1) // self._block_token_capacity
        if block_needed > len(self._free_block_ids):
            raise RuntimeError("Not enough free blocks for allocation")
        for _ in range(block_needed):
            block_id = self._free_block_ids.popleft()
            block = self._blocks[block_id]
            block.ref_count += 1
            block.sequence_ids.add(sequence.request_id)
            block.used_tokens = min(num_tokens, block.token_capacity)
            num_tokens -= block.token_capacity
            sequence.kv_block_ids.append(block_id)

        
        self._publish_kv_usage(used_blocks=self._num_blocks - len(self._free_block_ids), total_blocks=self._num_blocks)

        return sequence.kv_block_ids[-block_needed:]

    def allocate_for_prefill(self, sequence: Sequence, num_tokens: int) -> list[int]:
        """Allocate enough blocks to hold prefilling `num_tokens`."""
        return self._allocate_blocks(sequence, num_tokens)

    def append_for_decode(self, sequence: Sequence, num_tokens: int = 1) -> list[int]:
        """Append decode tokens, extending sequence block table if needed."""
        last_block_id = sequence.kv_block_ids[-1] if sequence.kv_block_ids else None
        last_block = self._blocks[last_block_id] if last_block_id is not None else None

        if last_block and last_block.used_tokens + num_tokens <= last_block.token_capacity:
            last_block.used_tokens += num_tokens
            return []
        else:
            if last_block_id is None:
                raise RuntimeError("No blocks allocated for this sequence to append decode tokens")
            remaining_capacity = last_block.token_capacity - last_block.used_tokens
            tokens_to_append = num_tokens - remaining_capacity
            last_block.used_tokens += remaining_capacity

            new_block_ids = self._allocate_blocks(sequence, tokens_to_append)

            return new_block_ids

    def free_sequence(self, sequence: Sequence) -> None:
        """Release blocks owned by a completed/aborted sequence."""
        for block_id in sequence.kv_block_ids:
            block = self._blocks[block_id]
            block.ref_count -= 1
            block.sequence_ids.discard(sequence.request_id)
            # If no more sequences reference this block, reset and return to free pool
            if block.ref_count == 0:
                block.used_tokens = 0
                block.sequence_ids.clear()
                block.hash_key = None
                self._free_block_ids.append(block_id)

        sequence.kv_block_ids.clear()
        self._publish_kv_usage(used_blocks=self._num_blocks - len(self._free_block_ids), total_blocks=self._num_blocks)


    def try_reuse_prefix(self, sequence: Sequence) -> bool:
        """Attempt hashed prefix-block reuse; return True on cache hit."""
        raise NotImplementedError

    def evict_or_preempt(self, required_blocks: int) -> list[str]:
        """Free enough blocks for `required_blocks`; return affected sequence IDs."""
        raise NotImplementedError

    def available_block_count(self) -> int:
        """Expose current free block count for scheduler decisions/metrics."""
        return len(self._free_block_ids)

    def _publish_kv_usage(self, used_blocks: int, total_blocks: int) -> None:
        """Publish KV block usage and utilization ratio."""
        set_kv_block_usage(used_blocks=used_blocks, total_blocks=total_blocks)
