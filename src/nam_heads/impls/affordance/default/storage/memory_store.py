import json
from typing import Optional, Dict, List, Tuple

from .base import CategoryStore


class MemoryCategoryStore(CategoryStore):
    """In-memory category store. Fast but not persistent across restarts."""

    def __init__(self):
        # Keyed by (lemma, ontology_type_or_empty) -> (category, confidence)
        self._store: Dict[Tuple[str, str], Tuple[str, float]] = {}
        # Predicate documents (affordance predicate mode): name -> JSON text
        self._docs: Dict[str, str] = {}
        self._seeded: Dict[Tuple[str, str], str] = {}   # seed row -> seed version
        self.doc_batches = 0   # get_docs_batch calls (one per head batch)

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
        self._seeded.pop(key, None)              # a learned write is never a seed row

    def seed_categories(self, entries, version: str) -> int:
        """Category seed rows: written when absent, or when seeded under
        another version; learned rows (put_category_weighted) are kept."""
        n = 0
        for lemma, category, confidence, ontology_type in entries:
            key = (lemma, ontology_type or "")
            if key not in self._store or self._seeded.get(key, version) != version:
                self._store[key] = (category, confidence)
                self._seeded[key] = version
                n += 1
        return n

    def has_lemma(self, lemma: str) -> bool:
        return (lemma, "") in self._store or any(k[0] == lemma for k in self._store)

    def count(self) -> int:
        return len(self._store)

    # ---- predicate documents ------------------------------------------------
    def get_docs_batch(self, names: List[str]) -> Dict[str, Optional[dict]]:
        """Docs by name; a category entry (``lemma`` or ``lemma::ontology``,
        the KV key layout) reads as {category, confidence} like on KV."""
        self.doc_batches += 1
        out = {}
        for n in names:
            if n in self._docs:
                out[n] = json.loads(self._docs[n])
                continue
            lemma, _, ont = n.partition("::")
            cat = self._store.get((lemma, ont))
            out[n] = {"category": cat[0], "confidence": cat[1]} if cat else None
        return out

    def put_doc(self, name: str, doc: dict) -> None:
        from .merge import merge_docs
        old = json.loads(self._docs[name]) if name in self._docs else None
        self._docs[name] = json.dumps(merge_docs(old, doc), sort_keys=True)

    def seed_docs(self, entries: List[Tuple[str, dict]]) -> int:
        """Write seed docs that are absent or seeded under another version
        (counters kept); learned docs are left alone."""
        from .merge import merge_docs
        from .seed_version import needs_seed
        added = 0
        for n, doc in entries:
            old = json.loads(self._docs[n]) if n in self._docs else None
            if needs_seed(old, doc.get("seed_version")):
                self._docs[n] = json.dumps(merge_docs(old, doc, prefer_b=True), sort_keys=True)
                added += 1
        return added

    def doc_items(self) -> Dict[str, dict]:
        """Every predicate doc (tests and measurement)."""
        return {n: json.loads(v) for n, v in sorted(self._docs.items())}
