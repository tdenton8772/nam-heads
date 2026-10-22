"""
CouchbaseKV-backed attribute category store.

Uses the session bucket with key prefix ``acat::``:
  acat::<lemma>              -> JSON {category, confidence}      (global default)
  acat::<lemma>::<ontology>  -> JSON {category, confidence}      (ontology-scoped)

In-memory cache eliminates repeat KV lookups for common adjectives across batches.
Deferred writes: put_category_weighted updates cache immediately and buffers the
KV write. Pending writes are flushed in a single batch_set on the next batch lookup.
"""
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple

from .base import CategoryStore

logger = logging.getLogger(__name__)

_PREFIX = "acat"

# Sentinel for "key was fetched but not found" (distinguishes from never-fetched)
_MISS = object()
_ABSENT = object()


class CouchbaseKVCategoryStore(CategoryStore):
    def __init__(self, kv, lmdb_cache=None, kv_on_cache_miss: bool = True):
        self._kv = kv
        self._lmdb = lmdb_cache  # Optional LmdbAttributeCache (L2)
        self._kv_on_miss = kv_on_cache_miss  # If False, skip L3 KV on L2 miss
        # In-memory cache: kv_key -> raw_bytes or _MISS. Size-bounded
        # (storage/bounded_cache.py): clean entries LRU-evicted beyond
        # NAM_ATTRIBUTE_L1_MAX, local writes pinned until relayed.
        from src.storage.bounded_cache import BoundedL1
        self._cache = BoundedL1.from_env("attribute")
        self._cache_lock = threading.Lock()
        # Deferred writes: accumulated (key, payload) pairs flushed on next batch read
        self._pending_writes: List[Tuple[str, bytes]] = []
        self._pending_lock = threading.Lock()
        # Pending LMDB relay writes (drained by NATS worker for attribute-lmdb-sync)
        self._pending_lmdb_writes: List[Tuple[str, bytes]] = []
        self._pending_lmdb_lock = threading.Lock()

    def _flush_pending(self):
        """Flush any accumulated deferred writes in a single batch_set."""
        with self._pending_lock:
            if not self._pending_writes:
                return
            batch = list(self._pending_writes)
            self._pending_writes.clear()
        try:
            self._kv.pipeline_set(batch)
        except Exception as e:
            logger.warning("acat deferred write flush failed: %s (%d items)", e, len(batch))

    def _key(self, lemma: str, ontology_type: Optional[str] = None) -> str:
        if ontology_type:
            return f"{_PREFIX}::{lemma}::{ontology_type}"
        return f"{_PREFIX}::{lemma}"

    def get_category(self, lemma: str) -> Optional[str]:
        key = self._key(lemma)
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is None:
            raw, _ = self._kv.get(key)
            with self._cache_lock:
                self._cache[key] = raw if raw is not None else _MISS
            cached = raw if raw is not None else _MISS
        if cached is not _MISS and cached is not None:
            try:
                return json.loads(cached.decode("utf-8"))["category"]
            except Exception:
                pass
        return None

    def put_category(self, lemma: str, category: str, confidence: float = 1.0) -> None:
        doc = {"category": category, "confidence": confidence}
        self._kv.set(self._key(lemma), json.dumps(doc).encode("utf-8"))

    def _lookup_key(self, key: str) -> object:
        """L1 cache check, then optionally L2 LMDB and L3 KV based on policy."""
        with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            return cached
        # L2: LMDB
        if self._lmdb and self._lmdb.available:
            val = self._lmdb.get(key)
            if val is not None:
                from src.storage.lmdb_node_cache import _MISS_SENTINEL
                result = _MISS if val == _MISS_SENTINEL else val
                with self._cache_lock:
                    self._cache[key] = result
                return result
        # L3: KV (only if policy allows)
        if self._kv_on_miss:
            raw, _ = self._kv.get(key)
            with self._cache_lock:
                self._cache[key] = raw if raw is not None else _MISS
            return raw if raw is not None else _MISS
        return None

    def get_category_weighted(self, lemma: str, ontology_type: Optional[str] = None) -> Optional[Tuple[str, float]]:
        # Check ontology-scoped key
        if ontology_type:
            ont_key = self._key(lemma, ontology_type)
            cached = self._lookup_key(ont_key)
            if cached is not None and cached is not _MISS:
                try:
                    doc = json.loads(cached.decode("utf-8"))
                    return doc["category"], float(doc["confidence"])
                except Exception:
                    pass
        # Fall back to global key
        global_key = self._key(lemma)
        cached = self._lookup_key(global_key)
        if cached is not None and cached is not _MISS:
            try:
                doc = json.loads(cached.decode("utf-8"))
                return doc["category"], float(doc["confidence"])
            except Exception:
                pass
        return None

    def _fetch_raw(self, all_keys: List[str]) -> Dict[str, object]:
        """L1 cache, then L2 LMDB, then ONE L3 KV batch_get for the misses.
        Returns key -> raw bytes or _MISS, assembled as each tier answers
        (never re-read from the bounded L1, which may evict this batch's
        own entries when the batch is large)."""
        from src.storage.lmdb_node_cache import _MISS_SENTINEL
        uncached: List[str] = []
        snap: Dict[str, object] = {}
        with self._cache_lock:
            for key in all_keys:
                v = self._cache.get(key, _ABSENT)
                if v is _ABSENT:
                    uncached.append(key)
                else:
                    snap[key] = v

        # L2: check LMDB for uncached keys before hitting KV
        if uncached and self._lmdb and self._lmdb.available:
            lmdb_results = self._lmdb.batch_get(uncached)
            still_uncached = []
            with self._cache_lock:
                for key in uncached:
                    val = lmdb_results.get(key)
                    if val is not None:
                        v = _MISS if val == _MISS_SENTINEL else val
                        self._cache[key] = v
                        snap[key] = v
                    else:
                        still_uncached.append(key)
            uncached = still_uncached

        # L3: fetch remaining uncached keys from KV (only if policy allows)
        if uncached and self._kv_on_miss:
            raw_results = self._kv.pipeline_get(uncached)
            kv_results_for_lmdb = []
            with self._cache_lock:
                for key in uncached:
                    val, _ = raw_results.get(key, (None, None))
                    v = val if val is not None else _MISS
                    self._cache[key] = v
                    snap[key] = v
                    # Backfill LMDB with KV results
                    if self._lmdb and self._lmdb.available:
                        kv_results_for_lmdb.append((key, val if val is not None else _MISS_SENTINEL))
            if kv_results_for_lmdb and self._lmdb and not self._lmdb._readonly:
                self._lmdb.batch_put(kv_results_for_lmdb)
        return {k: snap.get(k, _MISS) for k in all_keys}

    def get_categories_weighted_batch(
        self, queries: List[Tuple[str, Optional[str]]]
    ) -> Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]]:
        """Batch lookup with in-memory cache. Only fetches cache misses from KV."""
        # Flush any deferred writes first (from previous batch's auto-classify)
        self._flush_pending()

        if not queries:
            return {}

        # Build all keys: ontology-scoped + global fallback for each query
        all_keys: List[str] = []
        key_set: set = set()
        ont_keys: List[Optional[str]] = []  # ontology-scoped key per query (or None)

        for qi, (lemma, ontology_type) in enumerate(queries):
            ont_key = None
            if ontology_type:
                ont_key = self._key(lemma, ontology_type)
                if ont_key not in key_set:
                    key_set.add(ont_key)
                    all_keys.append(ont_key)
            ont_keys.append(ont_key)

            global_key = self._key(lemma)
            if global_key not in key_set:
                key_set.add(global_key)
                all_keys.append(global_key)

        cache_snap = self._fetch_raw(all_keys)

        results: Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]] = {}
        for qi, (lemma, ontology_type) in enumerate(queries):
            query_key = (lemma, ontology_type)
            if query_key in results:
                continue

            # Try ontology-scoped first
            if ont_keys[qi]:
                raw = cache_snap.get(ont_keys[qi], _MISS)
                if raw is not _MISS and raw is not None:
                    try:
                        doc = json.loads(raw.decode("utf-8"))
                        results[query_key] = (doc["category"], float(doc["confidence"]))
                        continue
                    except Exception:
                        pass

            # Fall back to global
            global_key = self._key(lemma)
            raw = cache_snap.get(global_key, _MISS)
            if raw is not _MISS and raw is not None:
                try:
                    doc = json.loads(raw.decode("utf-8"))
                    results[query_key] = (doc["category"], float(doc["confidence"]))
                    continue
                except Exception:
                    pass

            results[query_key] = None

        return results

    def put_category_weighted(self, lemma: str, category: str, confidence: float = 1.0,
                              ontology_type: Optional[str] = None) -> None:
        doc = {"category": category, "confidence": confidence}
        payload = json.dumps(doc).encode("utf-8")
        key = self._key(lemma, ontology_type)
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
            entries: list of (lemma, category, confidence, ontology_type[, extras])
                tuples. ``extras`` (global rows only) are merged into an
                existing row that lacks them (seeding.merge_seed).
        Returns:
            Number of entries written.
        """
        from .seeding import merge_seed
        if not entries:
            return 0
        keys = [self._key(e[0], e[3]) for e in entries]
        existing = self._kv.pipeline_get(keys)
        missing = []
        for i, e in enumerate(entries):
            lemma, cat, conf, ont = e[:4]
            extras = e[4] if len(e) > 4 else None
            raw = existing.get(keys[i], (None, None))[0]
            if raw is None:
                payload = json.dumps(merge_seed(None, cat, conf, extras),
                                     sort_keys=True).encode("utf-8")
                missing.append((keys[i], payload))
            elif extras:
                try:
                    old = json.loads(raw.decode("utf-8"))
                except Exception:
                    continue
                new = merge_seed(old, cat, conf, extras)
                if new != old:
                    missing.append((keys[i], json.dumps(new, sort_keys=True).encode("utf-8")))
        if missing:
            self._kv.pipeline_set(missing)
        written = dict(missing)
        # Every seeded row must be readable by THIS process (L1) and by
        # every reader on the node (LMDB L2, filled only by the relay):
        # readers do not fall through to KV (NAM_ATTRIBUTE_KV_ON_MISS=0).
        final = {}
        for i, e in enumerate(entries):
            k = keys[i]
            v = written.get(k, existing.get(k, (None, None))[0])
            if v is not None:
                final[k] = v
        with self._cache_lock:
            self._cache.update_dirty(final)
        relay = list(written.items())
        if self._lmdb and self._lmdb.available:
            have = self._lmdb.batch_get([k for k in final if k not in written])
            relay += [(k, v) for k, v in final.items() if k not in written and have.get(k) is None]
        if relay:
            with self._pending_lmdb_lock:
                self._pending_lmdb_writes.extend(relay)
        return len(missing)

    def _decode(self, raw) -> Optional[dict]:
        if raw is None or raw is _MISS:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def get_doc(self, lemma: str) -> Optional[dict]:
        return self._decode(self._lookup_key(self._key(lemma)))

    def get_docs_batch(self, lemmas: List[str]) -> Dict[str, Optional[dict]]:
        """Full global rows; one batched L2/L3 fetch for cache misses."""
        if not lemmas:
            return {}
        self._flush_pending()
        raw = self._fetch_raw(list(dict.fromkeys(self._key(lemma) for lemma in lemmas)))
        return {lemma: self._decode(raw.get(self._key(lemma), _MISS)) for lemma in lemmas}

    def put_doc(self, lemma: str, doc: dict) -> None:
        """Write a full global row through the same deferred-KV and LMDB-relay
        path as put_category_weighted."""
        payload = json.dumps(doc, sort_keys=True).encode("utf-8")
        key = self._key(lemma)
        with self._cache_lock:
            self._cache.set_dirty(key, payload)
        with self._pending_lock:
            self._pending_writes.append((key, payload))
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.append((key, payload))

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
        the attribute-lmdb-sync worker via NATS.
        """
        with self._pending_lmdb_lock:
            writes = self._pending_lmdb_writes
            self._pending_lmdb_writes = []
        return writes

    def has_lemma(self, lemma: str) -> bool:
        raw, _ = self._kv.get(self._key(lemma))
        return raw is not None

    def count(self) -> int:
        return -1
