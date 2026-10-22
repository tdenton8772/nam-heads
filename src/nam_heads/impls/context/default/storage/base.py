from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class ContextStore(ABC):
    """
    Canonical interface for context dimension storage.

    Maps (raw_phrase, dimension) pairs to canonical values.
    Backends: in-memory (dev/test), SQLite (persistent), Couchbase KV (v0.2.0).

    ALL backends must implement this interface exactly.
    """

    @abstractmethod
    def get(self, raw_phrase: str, dimension: str) -> Optional[str]:
        """Return the canonical value for *raw_phrase* in *dimension*, or None."""
        pass

    @abstractmethod
    def put(self, raw_phrase: str, dimension: str, canonical_value: str, confidence: float = 1.0) -> None:
        """Store or update the canonical mapping."""
        pass

    @abstractmethod
    def has(self, raw_phrase: str, dimension: str) -> bool:
        """Return True if a mapping exists for (raw_phrase, dimension)."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored mappings."""
        pass

    def get_batch(
        self, queries: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Optional[str]]:
        """Batch lookup of (raw_phrase, dimension) -> canonical_value.

        Default: serial fallback. CouchbaseKV overrides with batch_get.
        """
        results: Dict[Tuple[str, str], Optional[str]] = {}
        for phrase, dim in queries:
            results[(phrase, dim)] = self.get(phrase, dim)
        return results
