"""
CouchbaseKV-backed context store.

Uses the session bucket with key prefix ``ctx::``:
  ctx::<raw_phrase>::<dimension>  -> JSON {canonical_value, confidence}

In-memory cache eliminates repeat KV lookups for common contexts across batches.
Deferred writes: put() updates cache immediately and buffers the KV write.
Pending writes are flushed in a single batch_set on the next batch lookup.

L2 LMDB cache (optional): node-level shared memory between L1 (in-process dict)
and L3 (CouchbaseKV). Same pattern as attribute/affordance stores.
"""
import json
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

from .base import ContextStore

logger = logging.getLogger(__name__)

_PREFIX = "ctx"

_MISS = object()


class _Miss:
    """A cached negative lookup that expires (NAM_CONTEXT_MISS_TTL_S, default
    300 s), so an entry another pod writes later becomes visible. Only KV
    misses (NAM_CONTEXT_KV_ON_MISS=1) and explicit LMDB miss sentinels are
    cached; a plain LMDB miss is never cached."""
    __slots__ = ("expires",)

    def __init__(self, ttl: float):
        self.expires = time.monotonic() + ttl


def _miss_ttl() -> float:
    try:
        return float(os.getenv("NAM_CONTEXT_MISS_TTL_S", "300"))
    except ValueError:
        return 300.0


class CouchbaseKVContextStore(ContextStore):
    def __init__(self, kv, lmdb_cache=None, kv_on_cache_miss: bool = True):
        self._kv = kv
        self._lmdb = lmdb_cache  # Optional LmdbAttributeCache (L2)
        self._kv_on_miss = kv_on_cache_miss  # If False, skip L3 KV on L2 miss
        # Size-bounded L1 (storage/bounded_cache.py): clean entries LRU-evicted
        # beyond NAM_CONTEXT_L1_MAX, local writes pinned until relayed.
        from src.storage.bounded_cache import BoundedL1
        self._cache = BoundedL1.from_env("context")
        self._cache_lock = threading.Lock()
        # Deferred writes: accumulated (key, payload) pairs flushed on next batch read
        self._pending_writes: List[Tuple[str, bytes]] = []
        self._pending_lock = threading.Lock()
        # Pending LMDB relay writes (drained by NATS worker for context-lmdb-sync)
        self._pending_lmdb_writes: List[Tuple[str, bytes]] = []
        self._pending_lmdb_lock = threading.Lock()

    def _flush_pending(self):
        """Flush any accumulated deferred writes in a single batch_set."""
        with self._pending_lock:
            if not self._pending_writes:
                return
            batch = list(self._pending_writes)
            self._pending_writes.clear()
        from .merge import batch_cas_merge_set, is_mergeable
        plain = [(k, v) for k, v in batch if not is_mergeable(v)]
        votes = [(k, v) for k, v in batch if is_mergeable(v)]
        try:
            if plain:
                self._kv.pipeline_set(plain)
            if votes:
                # One read-merge-CAS round for every vote set in the flush
                # (it was a GET + CAS SET per key).
                batch_cas_merge_set(self._kv, votes)
        except Exception as e:
            logger.warning("ctx deferred write flush failed: %s (%d items)", e, len(batch))

    def _key(self, raw_phrase: str, dimension: str) -> str:
        return f"{_PREFIX}::{raw_phrase}::{dimension}"

    def get(self, raw_phrase: str, dimension: str) -> Optional[str]:
        key = self._key(raw_phrase, dimension)
        # Check cache first
        with self._cache_lock:
            cached = self._cache.get(key)
            if isinstance(cached, _Miss) and cached.expires <= time.monotonic():
                del self._cache[key]
                cached = None
        if isinstance(cached, _Miss):
            return None
        if cached is not None:
            try:
                return json.loads(cached.decode("utf-8"))["canonical_value"]
            except Exception:
                pass
        if self._kv_on_miss:
            raw, _ = self._kv.get(key)
            with self._cache_lock:
                self._cache[key] = raw if raw is not None else _Miss(_miss_ttl())
            if raw:
                try:
                    return json.loads(raw.decode("utf-8"))["canonical_value"]
                except Exception:
                    pass
        return None

    def get_batch(
        self, queries: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Optional[str]]:
        """Batch lookup with in-memory cache. Only fetches cache misses from KV."""
        # Flush any deferred writes first (from previous batch's auto-classify)
        self._flush_pending()

        if not queries:
            return {}

        all_keys: List[str] = []
        key_to_query: Dict[str, Tuple[str, str]] = {}
        for phrase, dim in queries:
            k = self._key(phrase, dim)
            if k not in key_to_query:
                all_keys.append(k)
                key_to_query[k] = (phrase, dim)

        # Check cache, collect misses
        # The answer is assembled as each tier answers, never re-read from
        # the bounded L1 (which may evict this batch's own entries).
        uncached: List[str] = []
        snap: Dict[str, object] = {}
        now = time.monotonic()
        with self._cache_lock:
            for key in all_keys:
                v = self._cache.get(key)
                if isinstance(v, _Miss) and v.expires <= now:
                    del self._cache[key]
                    v = None
                if v is None:
                    uncached.append(key)
                else:
                    snap[key] = v

        # L2: check LMDB for uncached keys before hitting KV
        if uncached and self._lmdb and self._lmdb.available:
            from src.storage.lmdb_node_cache import _MISS_SENTINEL
            lmdb_results = self._lmdb.batch_get(uncached)
            lmdb_hits = []
            still_uncached = []
            for key in uncached:
                val = lmdb_results.get(key)
                if val is not None:
                    if val == _MISS_SENTINEL:
                        lmdb_hits.append((key, _Miss(_miss_ttl())))
                    else:
                        lmdb_hits.append((key, val))
                else:
                    still_uncached.append(key)
            with self._cache_lock:
                for key, val in lmdb_hits:
                    self._cache[key] = val
                    snap[key] = val
            uncached = still_uncached

        # L3: fetch remaining uncached keys from KV (only if policy allows)
        if uncached and self._kv_on_miss:
            raw_results = self._kv.pipeline_get(uncached)
            kv_results_for_lmdb = []
            with self._cache_lock:
                for key in uncached:
                    val, _ = raw_results.get(key, (None, None))
                    v = val if val is not None else _Miss(_miss_ttl())
                    self._cache[key] = v
                    snap[key] = v
                    # Backfill LMDB with KV results
                    if self._lmdb and self._lmdb.available:
                        from src.storage.lmdb_node_cache import _MISS_SENTINEL
                        lmdb_val = val if val is not None else _MISS_SENTINEL
                        kv_results_for_lmdb.append((key, lmdb_val))
            if kv_results_for_lmdb and self._lmdb and not self._lmdb._readonly:
                self._lmdb.batch_put(kv_results_for_lmdb)

        cache_snap = {k: snap.get(k, _MISS) for k in all_keys}
        cache_snap = {k: (_MISS if isinstance(v, _Miss) else v) for k, v in cache_snap.items()}

        results: Dict[Tuple[str, str], Optional[str]] = {}
        for k, query in key_to_query.items():
            raw = cache_snap.get(k, _MISS)
            if raw is not _MISS and raw is not None:
                try:
                    results[query] = json.loads(raw.decode("utf-8"))["canonical_value"]
                    continue
                except Exception:
                    pass
            results[query] = None

        return results

    def put(self, raw_phrase: str, dimension: str, canonical_value: str, confidence: float = 1.0) -> None:
        doc = {"canonical_value": canonical_value, "confidence": confidence}
        payload = json.dumps(doc).encode("utf-8")
        key = self._key(raw_phrase, dimension)
        # Update cache immediately (for same-process reads)
        with self._cache_lock:
            self._cache.set_dirty(key, payload)
        # Defer the KV write to be flushed in batch on next batch lookup
        with self._pending_lock:
            self._pending_writes.append((key, payload))
        # Accumulate for LMDB relay (drained by NATS worker)
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.append((key, payload))

    def seed_batch(self, entries):
        """Batch-seed: check all keys in one batch_get, write only missing.

        Args:
            entries: list of (raw_phrase, dimension, canonical_value, confidence) tuples
        Returns:
            Number of entries written (were missing).
        """
        if not entries:
            return 0
        keys = [self._key(phrase, dim) for phrase, dim, _, _ in entries]
        existing = self._kv.pipeline_get(keys)
        missing = []
        for i, (_, _, cv, conf) in enumerate(entries):
            if existing.get(keys[i], (None, None))[0] is None:
                payload = json.dumps({"canonical_value": cv, "confidence": conf}).encode("utf-8")
                missing.append((keys[i], payload))
        if missing:
            self._kv.pipeline_set(missing)
            with self._cache_lock:
                for k, v in missing:
                    self._cache.set_dirty(k, v)
        return len(missing)

    def reset_session_state(self) -> Dict[str, int]:
        """Drop everything this process holds from the session bucket: the L1
        and the writes not yet sent or relayed. Called after the bucket is
        flushed (cold reset, src/storage/session_reset.py). A write made before
        the flush must not land in the emptied bucket. Returns the number
        dropped from each."""
        with self._cache_lock:
            l1 = len(self._cache)
            self._cache.clear()
        with self._pending_lock:
            pending = len(self._pending_writes)
            self._pending_writes = []
        with self._pending_lmdb_lock:
            relay = len(self._pending_lmdb_writes)
            self._pending_lmdb_writes = []
        return {"l1": l1, "pending_writes": pending, "pending_relay": relay}

    def drain_pending_lmdb_writes(self) -> List[Tuple[str, bytes]]:
        """Return and clear accumulated writes for external LMDB sync.

        Called by NATS worker after encode() to publish writes to
        the context-lmdb-sync worker via NATS.
        """
        with self._pending_lmdb_lock:
            writes = self._pending_lmdb_writes
            self._pending_lmdb_writes = []
        return writes

    def has(self, raw_phrase: str, dimension: str) -> bool:
        raw, _ = self._kv.get(self._key(raw_phrase, dimension))
        return raw is not None

    def count(self) -> int:
        return -1
