from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class CategoryStore(ABC):
    """
    Canonical interface for verb category storage.

    Maps verb lemmas to semantic categories (e.g. ``"repair"`` -> ``"physical"``).
    Backends range from in-memory (dev/test) to SQLite (persistent) to Couchbase KV
    (distributed, v0.2.0).

    ALL backends must implement this interface exactly.
    """

    @abstractmethod
    def get_category(self, lemma: str) -> Optional[str]:
        """Return the semantic category for *lemma*, or ``None`` if unknown."""
        pass

    @abstractmethod
    def put_category(self, lemma: str, category: str, confidence: float = 1.0) -> None:
        """Store or update the category mapping for *lemma*."""
        pass

    @abstractmethod
    def get_category_weighted(self, lemma: str, ontology_type: Optional[str] = None) -> Optional[Tuple[str, float]]:
        """Return (category, confidence) for lemma, optionally scoped by ontology_type.

        If ontology_type provided and an override exists, return that.
        Otherwise fall back to the default (ontology_type=None) row.
        """
        pass

    @abstractmethod
    def put_category_weighted(self, lemma: str, category: str, confidence: float = 1.0,
                              ontology_type: Optional[str] = None) -> None:
        """Store mapping, optionally scoped to an ontology type. None = global default."""
        pass

    @abstractmethod
    def has_lemma(self, lemma: str) -> bool:
        """Return True if *lemma* has a stored category."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored lemma->category mappings."""
        pass

    def get_categories_weighted_batch(
        self, queries: List[Tuple[str, Optional[str]]]
    ) -> Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]]:
        """Batch lookup of (lemma, ontology_type) -> (category, confidence).

        Default: serial fallback. CouchbaseKV overrides with batch_get.
        """
        results: Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]] = {}
        for lemma, ont in queries:
            results[(lemma, ont)] = self.get_category_weighted(lemma, ont)
        return results

    # ---- predicate documents (affordance predicate mode) --------------------
    # Backends that hold the predicate lexicon implement these three. A store
    # without them leaves predicate extraction on its seed lexicon.
    def get_docs_batch(self, names: List[str]) -> Dict[str, Optional[dict]]:
        raise NotImplementedError

    def put_doc(self, name: str, doc: dict) -> None:
        raise NotImplementedError

    def seed_docs(self, entries: List[Tuple[str, dict]]) -> int:
        raise NotImplementedError
