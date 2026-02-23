class KVCache:
    """Simple fixed-size dict cache to bootstrap KV-cache experiments."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max_entries
        self._store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        if len(self._store) >= self.max_entries:
            first_key = next(iter(self._store))
            self._store.pop(first_key)
        self._store[key] = value
