import logging
from pathlib import Path
from typing import Optional

from .base import ContextStore

logger = logging.getLogger(__name__)


def create_context_store(
    backend: str,
    fallback: Optional[str] = "memory",
    path: Optional[str] = None,
    seed: bool = False,
    kv_on_cache_miss: bool = True,
) -> ContextStore:
    """Create a ContextStore instance.

    Args:
        backend: ``"couchbase_kv"`` or ``"memory"``
        fallback: backend to use if primary fails (default ``"memory"``)
        path: unused (retained for API compatibility)
        seed: if True, populate the store with seed data from definitions.py
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
            "Primary context store '%s' failed (%s); falling back to '%s'",
            backend, e, fallback,
        )
        store = _make_store(fallback.lower(), None)

    if seed:
        # Deferred (retried in the background) instead of fatal when the
        # process opts in with NAM_SEED_DEFER_ON_ERROR=1 (nam-query).
        from src.storage.deferred_seed import run_seed
        run_seed("context-store", lambda: _seed_store(store))

    return store


def _make_store(backend: str, path: Optional[str], kv_on_cache_miss: bool = True) -> ContextStore:
    if backend in ("couchbase_kv", "couchbase"):
        from .couchbase_kv_store import CouchbaseKVContextStore
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
        # Optional L2 LMDB cache (shared memory, same pattern as attribute/affordance)
        lmdb_cache = None
        lmdb_path = os.getenv("NAM_CONTEXT_LMDB_PATH", "")
        if lmdb_path:
            try:
                from src.storage.lmdb_node_cache import LmdbAttributeCache
                readonly = os.getenv("NAM_CONTEXT_LMDB_READONLY", "1") == "1"
                map_size_mb = int(os.getenv("NAM_CONTEXT_LMDB_MAP_SIZE_MB", "256"))
                lmdb_cache = LmdbAttributeCache(
                    lmdb_path, map_size=map_size_mb * 1024 * 1024, readonly=readonly,
                )
                if not lmdb_cache.available:
                    logger.warning("Context LMDB cache not available at %s", lmdb_path)
                    lmdb_cache = None
                else:
                    logger.info("Context LMDB L2 cache enabled at %s (readonly=%s)", lmdb_path, readonly)
            except Exception as e:
                logger.warning("Failed to initialize context LMDB cache: %s", e)
        return CouchbaseKVContextStore(kv, lmdb_cache=lmdb_cache, kv_on_cache_miss=kv_on_cache_miss)

    if backend == "memory":
        from .memory_store import MemoryContextStore
        return MemoryContextStore()

    raise ValueError(f"Unknown context store backend: {backend}")


def _seed_store(store: ContextStore) -> None:
    """Seed the store with built-in definitions (no-op for existing entries)."""
    from ..definitions import (
        GEOGRAPHIC_SEED,
        TEMPORAL_PERIODS,
        DOMAIN_SEEDS,
        INSTITUTIONAL_KEYWORDS,
    )

    # Collect all seed entries: (raw_phrase, dimension, canonical_value, confidence)
    all_entries = []
    for surface_form, canonical in GEOGRAPHIC_SEED.items():
        all_entries.append((surface_form, "geographic", canonical, 1.0))
    for surface_form, canonical in TEMPORAL_PERIODS.items():
        all_entries.append((surface_form, "temporal", canonical, 1.0))
    for ontology_type, domain_label in DOMAIN_SEEDS.items():
        all_entries.append((f"domain::{ontology_type}", "domain", domain_label, 1.0))
    for inst_type, keywords_str in INSTITUTIONAL_KEYWORDS.items():
        for keyword in keywords_str.split():
            all_entries.append((f"inst::{keyword}", "institutional", inst_type, 1.0))

    # Use batch seeding if available (KV stores), else fall back to per-entry
    if hasattr(store, "seed_batch"):
        added = store.seed_batch(all_entries)
    else:
        added = 0
        for phrase, dim, canonical, conf in all_entries:
            if not store.has(phrase, dim):
                store.put(phrase, dim, canonical, confidence=conf)
                added += 1

    if added:
        logger.info("Seeded context store with %d entries", added)
