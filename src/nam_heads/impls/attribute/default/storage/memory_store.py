from typing import Optional, Dict, Tuple

from .base import CategoryStore


class MemoryCategoryStore(CategoryStore):
    """In-memory category store. Fast but not persistent across restarts."""

    def __init__(self):
        # Keyed by (lemma, ontology_type_or_empty) -> (category, confidence)
        self._store: Dict[Tuple[str, str], Tuple[str, float]] = {}
        # Global-row extras (predication-mode fields), keyed by lemma
        self._extras: Dict[str, dict] = {}
        self._lemmas: set = set()

    def get_category(self, lemma: str) -> Optional[str]:
        result = self.get_category_weighted(lemma, None)
        return result[0] if result else None

    def put_category(self, lemma: str, category: str, confidence: float = 1.0) -> None:
        self.put_category_weighted(lemma, category, confidence, ontology_type=None)

    def get_category_weighted(self, lemma: str, ontology_type: Optional[str] = None) -> Optional[Tuple[str, float]]:
        if ontology_type:
            result = self._store.get((lemma, ontology_type))
            if result is not None:
                return result
        return self._store.get((lemma, ""))

    def put_category_weighted(self, lemma: str, category: str, confidence: float = 1.0,
                              ontology_type: Optional[str] = None) -> None:
        key = (lemma, ontology_type or "")
        self._store[key] = (category, confidence)
        self._lemmas.add(lemma)

    def has_lemma(self, lemma: str) -> bool:
        return lemma in self._lemmas

    def get_doc(self, lemma: str) -> Optional[dict]:
        r = self._store.get((lemma, ""))
        if r is None:
            return None
        doc = dict(self._extras.get(lemma, {}))
        doc["category"], doc["confidence"] = r
        return doc

    def put_doc(self, lemma: str, doc: dict) -> None:
        self.put_category_weighted(lemma, doc.get("category", "uncategorized"),
                                   float(doc.get("confidence", 0.5)))
        self._extras[lemma] = {k: v for k, v in doc.items() if k not in ("category", "confidence")}

    def seed_batch(self, entries) -> int:
        """(lemma, category, confidence, ontology_type[, extras]) -- writes missing
        rows; merges extras into an existing global row that lacks them."""
        from .seeding import merge_seed
        added = 0
        for e in entries:
            lemma, cat, conf, ont = e[:4]
            extras = e[4] if len(e) > 4 else None
            if ont:
                self.put_category_weighted(lemma, cat, conf, ontology_type=ont)
                added += 1
                continue
            old = self.get_doc(lemma)
            new = merge_seed(old, cat, conf, extras)
            if new != old:
                self.put_doc(lemma, new)
                added += 1
        return added

    def count(self) -> int:
        return len(self._store)
