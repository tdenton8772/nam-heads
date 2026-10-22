from typing import Optional, Dict, Tuple

from .base import ContextStore


class MemoryContextStore(ContextStore):
    """In-memory context store. Fast but not persistent across restarts."""

    def __init__(self):
        self._store: Dict[Tuple[str, str], str] = {}

    def get(self, raw_phrase: str, dimension: str) -> Optional[str]:
        return self._store.get((raw_phrase, dimension))

    def put(self, raw_phrase: str, dimension: str, canonical_value: str, confidence: float = 1.0) -> None:
        self._store[(raw_phrase, dimension)] = canonical_value

    def has(self, raw_phrase: str, dimension: str) -> bool:
        return (raw_phrase, dimension) in self._store

    def count(self) -> int:
        return len(self._store)
