import logging
from typing import Optional

from .base import EntityStore

logger = logging.getLogger(__name__)


def create_entity_store(
    backend: str,
    fallback: Optional[str],
    kv_on_cache_miss: bool = True,
) -> EntityStore:
    backend = backend.lower()

    try:
        if backend in ("couchbase_kv", "couchbase"):
            from .couchbase_kv_store import CouchbaseKVEntityStore
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
            lmdb_path = os.getenv("NAM_ENTITY_LMDB_PATH", "")
            if lmdb_path:
                from .lmdb_cache import LmdbEntityCache
                map_size_mb = int(os.getenv("NAM_ENTITY_LMDB_MAP_SIZE_MB", "512"))
                lmdb_readonly = os.getenv("NAM_ENTITY_LMDB_READONLY", "0").lower() in ("1", "true", "yes")
                lmdb_cache = LmdbEntityCache(lmdb_path, map_size=map_size_mb * 1024 * 1024, readonly=lmdb_readonly)
            return CouchbaseKVEntityStore(kv, lmdb_cache=lmdb_cache, kv_on_cache_miss=kv_on_cache_miss)

        if backend == "lmdb":
            import os
            from .lmdb_cache import LmdbEntityCache
            from .lmdb_store import LmdbEntityStore
            lmdb_path = os.getenv("NAM_ENTITY_LMDB_PATH", "")
            if not lmdb_path:
                raise ValueError("lmdb backend requires NAM_ENTITY_LMDB_PATH")
            map_size_mb = int(os.getenv("NAM_ENTITY_LMDB_MAP_SIZE_MB", "512"))
            lmdb_cache = LmdbEntityCache(lmdb_path, map_size=map_size_mb * 1024 * 1024, readonly=True)
            return LmdbEntityStore(lmdb_cache)

        if backend == "memory":
            from .memory_store import MemoryEntityStore
            return MemoryEntityStore()

        raise ValueError(f"Unknown db_backend: {backend}")

    except Exception as e:
        if not fallback:
            raise

        fallback = fallback.lower()

        if fallback == "memory":
            # Not silent: on the ingest entity container (backend lmdb) or
            # nam-query (couchbase_kv) a memory store means no shared aliases,
            # redirects or session reads -- the head still runs, so this log
            # line is the only sign of it.
            logger.warning("Primary entity store '%s' failed (%s); falling back to '%s'",
                           backend, e, fallback)
            from .memory_store import MemoryEntityStore
            return MemoryEntityStore()

        raise RuntimeError(
            f"Primary backend '{backend}' failed ({e}); "
            f"fallback '{fallback}' also failed"
        )