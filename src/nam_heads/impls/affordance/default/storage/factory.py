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
            "Primary verb category store '%s' failed (%s); falling back to '%s'",
            backend, e, fallback,
        )
        store = _make_store(fallback.lower(), None)

    if seed:
        # Deferred (retried in the background) instead of fatal when the
        # process opts in with NAM_SEED_DEFER_ON_ERROR=1 (nam-query).
        from src.storage.deferred_seed import run_seed
        run_seed("affordance-store", lambda: _seed_store(store))

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
        lmdb_path = os.getenv("NAM_AFFORDANCE_LMDB_PATH", "")
        if lmdb_path:
            from src.storage.lmdb_node_cache import LmdbAttributeCache
            map_size_mb = int(os.getenv("NAM_AFFORDANCE_LMDB_MAP_SIZE_MB", "256"))
            lmdb_readonly = os.getenv("NAM_AFFORDANCE_LMDB_READONLY", "0").lower() in ("1", "true", "yes")
            lmdb_cache = LmdbAttributeCache(lmdb_path, map_size=map_size_mb * 1024 * 1024, readonly=lmdb_readonly)
        return CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb_cache, kv_on_cache_miss=kv_on_cache_miss)

    if backend == "memory":
        from .memory_store import MemoryCategoryStore
        return MemoryCategoryStore()

    raise ValueError(f"Unknown verb category store backend: {backend}")


def _seed_store(store: CategoryStore) -> None:
    """Seed the store with built-in verb category definitions (no-op for existing entries)."""
    from ..definitions import VERB_CATEGORIES, ONTOLOGY_CATEGORY_OVERRIDES

    # Collect all seed entries: (lemma, category, confidence, ontology_type)
    all_entries = []
    for category, lemmas in VERB_CATEGORIES.items():
        for lemma in lemmas:
            all_entries.append((lemma, category, 1.0, None))
    for ontology_type, overrides in ONTOLOGY_CATEGORY_OVERRIDES.items():
        for lemma, category in overrides.items():
            all_entries.append((lemma, category, 1.0, ontology_type))

    # Versioned: a changed seed artifact re-seeds seeded rows (never learned
    # ones). Batch seeding on KV stores, version-aware seeding on memory.
    from .seed_version import seed_version
    version = seed_version()
    if hasattr(store, "seed_batch"):
        added = store.seed_batch(all_entries, version=version)
    elif hasattr(store, "seed_categories"):
        added = store.seed_categories(all_entries, version)
    else:
        count_before = store.count()
        added = 0
        for lemma, category, confidence, ontology_type in all_entries:
            if ontology_type is None:
                if not store.has_lemma(lemma):
                    store.put_category_weighted(lemma, category, confidence, ontology_type=None)
                    added += 1
            else:
                store.put_category_weighted(lemma, category, confidence, ontology_type=ontology_type)
                added += 1

    if added:
        logger.info("Seeded verb category store with %d entries", added)

    # Predicate lexicon seed: verb bases, synonym pairs, phrasal values.
    if hasattr(store, "seed_docs"):
        try:
            from ..predicate_store import seed_entries
            n = store.seed_docs(seed_entries())
            if n:
                logger.info("Seeded predicate lexicon with %d entries", n)
        except NotImplementedError:
            pass
