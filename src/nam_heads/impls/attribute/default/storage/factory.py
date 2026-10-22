import logging
from pathlib import Path
from typing import Optional

from .base import CategoryStore

logger = logging.getLogger(__name__)


def create_category_store(
    backend: str,
    fallback: Optional[str] = "memory",
    path: Optional[str] = None,
    seed: bool = False,
    kv_on_cache_miss: bool = True,
) -> CategoryStore:
    """Create a CategoryStore instance.

    Args:
        backend: ``"couchbase_kv"`` or ``"memory"``
        fallback: backend to use if primary fails (default ``"memory"``)
        path: unused (retained for API compatibility)
        seed: if True, populate the store with seed categories from definitions.py
    """
    backend = backend.lower()

    if path:
        p = Path(path)
        (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)

    try:
        store = _make_store(backend, path, kv_on_cache_miss=kv_on_cache_miss)
    except Exception as e:
        if not fallback:
            raise
        logger.warning(
            "Primary category store '%s' failed (%s); falling back to '%s'",
            backend, e, fallback,
        )
        store = _make_store(fallback.lower(), None)

    if seed:
        # Deferred (retried in the background) instead of fatal when the
        # process opts in with NAM_SEED_DEFER_ON_ERROR=1 (nam-query).
        from src.storage.deferred_seed import run_seed
        run_seed("attribute-store", lambda: _seed_store(store))

    return store


def _make_store(backend: str, path: Optional[str], kv_on_cache_miss: bool = True) -> CategoryStore:
    if backend in ("couchbase_kv", "couchbase"):
        from .couchbase_kv_store import CouchbaseKVCategoryStore
        from src.storage.backends.kv_registry import KVRegistry, kv_kwargs_from_env
        import os
        kv = KVRegistry.get_or_create(
            "ingest-session",
            pool_workers=int(os.getenv("NAM_KV_POOL_INGEST_SESSION_WORKERS", "4")),
            **kv_kwargs_from_env(
                os.getenv("NAM_SESSION_BUCKET", "session"),
                vbuckets=int(os.getenv("NAM_SESSION_VBUCKETS", "256")),
            ),
        )
        lmdb_cache = None
        lmdb_path = os.getenv("NAM_ATTRIBUTE_LMDB_PATH", "")
        if lmdb_path:
            from src.storage.lmdb_node_cache import LmdbAttributeCache
            map_size_mb = int(os.getenv("NAM_ATTRIBUTE_LMDB_MAP_SIZE_MB", "256"))
            lmdb_readonly = os.getenv("NAM_ATTRIBUTE_LMDB_READONLY", "0").lower() in ("1", "true", "yes")
            lmdb_cache = LmdbAttributeCache(lmdb_path, map_size=map_size_mb * 1024 * 1024, readonly=lmdb_readonly)
        return CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb_cache, kv_on_cache_miss=kv_on_cache_miss)

    if backend == "memory":
        from .memory_store import MemoryCategoryStore
        return MemoryCategoryStore()

    raise ValueError(f"Unknown category store backend: {backend}")


def _seed_store(store: CategoryStore) -> None:
    """Seed the store with built-in category definitions (no-op for existing entries)."""
    from ..definitions import ATTRIBUTE_CATEGORIES, ONTOLOGY_CATEGORY_OVERRIDES

    # Collect all seed entries: (lemma, category, confidence, ontology_type)
    all_entries = []
    for category, lemmas in ATTRIBUTE_CATEGORIES.items():
        for lemma in lemmas:
            all_entries.append((lemma, category, 1.0, None))
    for ontology_type, overrides in ONTOLOGY_CATEGORY_OVERRIDES.items():
        for lemma, category in overrides.items():
            all_entries.append((lemma, category, 1.0, ontology_type))

    # Person nouns (person_nouns.txt, generated from WordNet): the seed of
    # what the predication path treats as a kind of person. Merged into an
    # existing row as extras, so an adjective seed ("official") keeps its
    # category and gains the role flag.
    from ..predication import person_nouns, abstract_nouns
    for noun in sorted(person_nouns()):
        all_entries.append((noun, "role", 1.0, None, {"role": True, "role_source": "seed"}))
    # Abstract / relational nouns (abstract_nouns.txt): the seed of the kind
    # flag (kind=False) that kind-noun anaphors, "the <noun> of X" and
    # "<nouns> such as X" consult. Learned rows (kind=True from evidence)
    # and the role seed keep what they hold; the row itself is an ordinary
    # learned-confidence row, so its category still follows observations.
    from ..learning import LEARNED_CONFIDENCE
    for noun in sorted(abstract_nouns()):
        all_entries.append((noun, "uncategorized", LEARNED_CONFIDENCE, None,
                            {"kind": False, "kind_source": "seed"}))

    # Use batch seeding if available (KV stores), else fall back to per-entry
    if hasattr(store, "seed_batch"):
        added = store.seed_batch(all_entries)
    else:
        count_before = store.count()
        added = 0
        for lemma, category, confidence, ontology_type, *_ in all_entries:
            if ontology_type is None:
                if not store.has_lemma(lemma):
                    store.put_category_weighted(lemma, category, confidence, ontology_type=None)
                    added += 1
            else:
                store.put_category_weighted(lemma, category, confidence, ontology_type=ontology_type)
                added += 1

    if added:
        logger.info("Seeded category store with %d entries", added)
