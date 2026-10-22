from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class CategoryStore(ABC):
    """
    Canonical interface for attribute category storage.

    Maps adjective lemmas to semantic categories (e.g. ``"angry"`` -> ``"emotional"``).
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

    # ---- full value documents ------------------------------------------
    # The global row of a lemma is a JSON document. ``category`` and
    # ``confidence`` are what every reader uses; predication mode adds what
    # it has learned about the value (evidence kinds, the ontology classes it
    # was observed with, whether it names a kind of person and why). Extra
    # fields are additive: older readers ignore them.

    def get_doc(self, lemma: str) -> Optional[dict]:
        r = self.get_category_weighted(lemma, None)
        return {"category": r[0], "confidence": r[1]} if r else None

    def get_docs_batch(self, lemmas: List[str]) -> Dict[str, Optional[dict]]:
        return {lemma: self.get_doc(lemma) for lemma in lemmas}

    def put_doc(self, lemma: str, doc: dict) -> None:
        self.put_category_weighted(lemma, doc.get("category", "uncategorized"),
                                   float(doc.get("confidence", 0.5)), ontology_type=None)
