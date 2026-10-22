"""
CouchbaseKV-backed verb category store.

Uses the SESSION bucket (``NAM_SESSION_BUCKET``, default "session") with key
prefix ``vcat::``:
  vcat::<lemma>              -> JSON {category, confidence}      (global default)
  vcat::<lemma>::<ontology>  -> JSON {category, confidence}      (ontology-scoped)

The only construction path is the attribute head's storage/factory.py, which
passes the session KV (KVRegistry "ingest-session"). This matters for flush
safety: session is deliberately preserved as the warm head-store cache; a
standard reset flushes the derived buckets only (main, nam, graph, vector: the
supervisor flush task's default). This store survives a standard reset.

In-memory cache eliminates repeat KV lookups for common verbs across batches.
Deferred writes: put_category_weighted updates cache immediately and buffers the
KV write. Pending writes are flushed in a single batch_set on the next batch lookup.
"""
import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

from .base import CategoryStore

logger = logging.getLogger(__name__)

_PREFIX = "vcat"

_MISS = object()
_ABSENT = object()

# Parsed predicate docs kept beside L1 (key -> (raw bytes, doc)), so a doc is
# decoded once per distinct value instead of on every batch read, put and
# flush. Bounded: cleared when full.
_PARSED_MAX = int(os.getenv("NAM_AFFORDANCE_PARSED_CACHE_MAX", "20000"))


class CouchbaseKVCategoryStore(CategoryStore):
    def __init__(self, kv, lmdb_cache=None, kv_on_cache_miss: bool = True):
        self._kv = kv
        self._lmdb = lmdb_cache  # Optional LmdbAttributeCache (L2)
        self._kv_on_miss = kv_on_cache_miss  # If False, skip L3 KV on L2 miss
        # Size-bounded L1 (storage/bounded_cache.py): clean entries LRU-evicted
        # beyond NAM_AFFORDANCE_L1_MAX, local writes pinned until relayed.
        from src.storage.bounded_cache import BoundedL1
        self._cache = BoundedL1.from_env("affordance")
        # Re-entrant: the KV flush may run on its own thread (enable_async_flush)
        # and merges under this lock, as put_doc does.
        self._cache_lock = threading.RLock()
        # Deferred writes: accumulated (key, payload) pairs flushed on next batch read
        self._pending_writes: List[Tuple[str, bytes]] = []
        self._pending_lock = threading.Lock()
        # Pending LMDB relay writes (drained by NATS worker for affordance-lmdb-sync)
        self._pending_lmdb_writes: List[Tuple[str, bytes]] = []
        self._pending_lmdb_lock = threading.Lock()
        self._flush_exec = None
        self._flush_future = None
        self._parsed: Dict[str, Tuple[bytes, object]] = {}      # latest L1 value
        self._parsed_kv: Dict[str, Tuple[bytes, object]] = {}   # last value written to KV
        # key -> (payload bytes, payload doc) of the last queued put_doc
        self._payload_docs: Dict[str, Tuple[bytes, dict]] = {}
        # key -> (current doc, payload, merged bytes, merged doc) of its last merge
        self._merge_memo: Dict[str, tuple] = {}

    # ---- parsed-doc cache ---------------------------------------------------
    def _parse(self, key: str, raw) -> Optional[object]:
        """json.loads(raw), reusing the parsed doc when ``raw`` is the value
        last parsed (or merged) for ``key``. The returned doc is shared:
        callers must not modify it."""
        for cache in (self._parsed, self._parsed_kv):
            hit = cache.get(key)
            if hit is not None and (hit[0] is raw or hit[0] == raw):
                return hit[1]
        try:
            doc = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:
            return None
        self._remember(key, raw, doc)
        return doc

    def _remember(self, key: str, raw, doc, kv: bool = False) -> None:
        cache = self._parsed_kv if kv else self._parsed
        if len(cache) >= _PARSED_MAX:
            cache.clear()
        cache[key] = (raw, doc)

    def _merge_kv(self, key: str, cur, payload) -> bytes:
        """``_merge`` for the KV flush: the stored value is remembered as the
        key's last KV value (the next flush reads it back from KV)."""
        with self._cache_lock:
            return self._merge(key, cur, payload, kv=True)

    def _merge(self, key: str, cur, payload, kv: bool = False) -> bytes:
        """merge.merge_payloads(cur, payload), on cached parsed docs.

        A key is merged twice per batch with the same inputs -- into L1 by
        put_doc, then into KV by the flush, whose current KV value is the one
        L1 held -- so the last merge per key is memoized by the identity of
        its parsed current doc and its payload."""
        from .merge import encode, merge_payload_docs
        pd = self._payload_docs.get(key)
        b = pd[1] if pd is not None and pd[0] is payload else self._parse_free(payload)
        a = self._parse(key, cur) if cur is not None else None
        memo = self._merge_memo.get(key)
        if memo is not None and memo[1] is payload and (memo[0] is a or memo[3] is a):
            # Same inputs as the last merge, or the current value IS its
            # result (another writer -- affordance-lmdb-sync's write-through
            # -- stored it first): merging the payload in again is a no-op.
            raw, m = memo[2], memo[3]
        else:
            m = merge_payload_docs(a, b)
            if m is b:
                raw = payload if isinstance(payload, (bytes, bytearray)) else encode(b or {})
            else:
                raw = encode(m)
            self._merge_memo[key] = (a, payload, raw, m)
        if isinstance(m, dict):
            self._remember(key, raw, m)
            if kv:
                self._remember(key, raw, m, kv=True)
        return raw

    def _is_mergeable(self, key: str, payload) -> bool:
        """merge.is_mergeable(payload), from the queued doc when it is known."""
        pd = self._payload_docs.get(key)
        if pd is not None and pd[0] is payload:
            d = pd[1]
            return isinstance(d, dict) and ("_g" in d or "_go" in d)
        from .merge import is_mergeable
        return is_mergeable(payload)

    @staticmethod
    def _parse_free(raw) -> Optional[object]:
        try:
            return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)
        except Exception:
            return None

    # ---- deferred KV writes --------------------------------------------------
    def enable_async_flush(self) -> None:
        """Run the KV flush on a background thread instead of inside the next
        batch read. The worker turns this on (services/nats_worker.run_head):
        the head's L1 already holds every write, the flush only publishes
        them to the session bucket, and its read-merge-CAS round trips cost
        the head wall time per batch (on kind, ~0.65 ms per key read: ~15
        ms/record). Flushes stay ordered (one thread) and an explicit
        ``_flush_pending()`` still waits for them."""
        if self._flush_exec is None:
            from concurrent.futures import ThreadPoolExecutor
            self._flush_exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vcat-flush")

    def _flush_soon(self) -> None:
        """What a batch read does with deferred writes: flush them now, or
        hand them to the flush thread (enable_async_flush)."""
        if self._flush_exec is None:
            self._flush_pending()
            return
        with self._pending_lock:
            if not self._pending_writes:
                return
        self._flush_future = self._flush_exec.submit(self._do_flush)

    def _flush_pending(self):
        """Flush deferred writes now (after any background flush in flight)."""
        fut = self._flush_future
        if fut is not None:
            try:
                fut.result()
            except Exception:
                pass
        self._do_flush()

    def _do_flush(self):
        """Flush deferred writes: plain category payloads in one batch_set
        (last writer wins, as before); predicate-lexicon docs with counters
        through cas_merge_set (storage/merge.py), so a concurrent pod's
        counts are merged, not overwritten. The merged value refreshes L1
        (joined with L1 when the flush ran beside the head's puts)."""
        from .merge import batch_cas_merge_set, is_mergeable
        with self._pending_lock:
            if not self._pending_writes:
                return
            batch = list(self._pending_writes)
            self._pending_writes.clear()
        flags = [self._is_mergeable(k, v) for k, v in batch]
        plain = [kv for kv, f in zip(batch, flags) if not f]
        merged = [kv for kv, f in zip(batch, flags) if f]
        try:
            if plain:
                self._kv.pipeline_set(plain)
        except Exception as e:
            logger.warning("vcat deferred write flush failed: %s (%d items)", e, len(plain))
        # Counter docs: one batched read-merge-CAS round for the whole flush
        # (a per-key GET + CAS SET put ~30 serial session round trips on every
        # record). Lost races fall back to the per-key loop inside.
        try:
            stored_by_key = batch_cas_merge_set(self._kv, merged, merge=self._merge_kv) if merged else {}
        except Exception as e:
            logger.warning("vcat merged write failed (%d items): %s", len(merged), e)
            stored_by_key = {}
        with self._cache_lock:
            for k, stored in stored_by_key.items():
                if not stored:
                    continue
                cur = self._cache.get(k)
                if self._flush_exec is not None and isinstance(cur, (bytes, bytearray)) \
                        and cur is not stored and cur != stored:
                    # put_doc may have merged a newer payload into L1 while
                    # this flush ran: join, never replace.
                    self._cache.set_dirty(k, self._merge(k, cur, stored))
                else:
                    self._cache.set_dirty(k, stored)
            flushed = {id(v) for _, v in merged}
            for k, _ in merged:
                pd = self._payload_docs.get(k)
                if pd is not None and id(pd[0]) in flushed:   # not a newer put's
                    self._payload_docs.pop(k, None)
                self._merge_memo.pop(k, None)

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

        Returns key -> raw bytes or _MISS. Shared by the category batch
        lookup and the predicate-doc batch lookup (get_docs_batch)."""
        # Check cache, collect misses. The answer is assembled here as each
        # tier answers, never re-read from L1 at the end: a bounded L1 may
        # evict this batch's own entries when the batch is large.
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
            from src.storage.lmdb_node_cache import _MISS_SENTINEL
            lmdb_results = self._lmdb.batch_get(uncached)
            lmdb_hits = []
            still_uncached = []
            for key in uncached:
                val = lmdb_results.get(key)
                if val is not None:
                    if val == _MISS_SENTINEL:
                        lmdb_hits.append((key, _MISS))
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
                    self._cache[key] = val if val is not None else _MISS
                    snap[key] = val if val is not None else _MISS
                    # Backfill LMDB with KV results
                    if self._lmdb and self._lmdb.available:
                        from src.storage.lmdb_node_cache import _MISS_SENTINEL
                        lmdb_val = val if val is not None else _MISS_SENTINEL
                        kv_results_for_lmdb.append((key, lmdb_val))
            if kv_results_for_lmdb and self._lmdb and not self._lmdb._readonly:
                self._lmdb.batch_put(kv_results_for_lmdb)

        return {k: snap.get(k, _MISS) for k in all_keys}

    def get_categories_weighted_batch(
        self, queries: List[Tuple[str, Optional[str]]]
    ) -> Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]]:
        """Batch lookup with in-memory cache. Only fetches cache misses from KV."""
        # Flush any deferred writes first (from previous batch's auto-classify)
        self._flush_soon()

        if not queries:
            return {}

        all_keys: List[str] = []
        key_set: set = set()
        ont_keys: List[Optional[str]] = []

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

            if ont_keys[qi]:
                raw = cache_snap.get(ont_keys[qi], _MISS)
                if raw is not _MISS and raw is not None:
                    try:
                        doc = json.loads(raw.decode("utf-8"))
                        results[query_key] = (doc["category"], float(doc["confidence"]))
                        continue
                    except Exception:
                        pass

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

    def seed_batch(self, entries, version: Optional[str] = None):
        """Batch-seed category rows: one batch_get, write the missing ones and
        (with ``version``) the ones seeded under another version; learned
        rows (auto-classified, confidence 0.5) are never replaced. Every seed
        row's current value is relayed to LMDB.

        Args:
            entries: list of (lemma, category, confidence, ontology_type) tuples
        Returns:
            Number of entries written.
        """
        from .seed_version import needs_seed
        if not entries:
            return 0
        keys = [self._key(lemma, ont) for lemma, _, _, ont in entries]
        existing = self._kv.pipeline_get(keys)
        write, current = [], []
        for i, (lemma, cat, conf, ont) in enumerate(entries):
            raw = existing.get(keys[i], (None, None))[0]
            old = None
            if raw is not None:
                try:
                    old = json.loads(raw.decode("utf-8"))
                except Exception:
                    old = {}
            if raw is None or (version is not None and needs_seed(old, version)):
                doc = {"category": cat, "confidence": conf}
                if version is not None:
                    doc["seed_version"] = version
                payload = json.dumps(doc).encode("utf-8")
                write.append((keys[i], payload))
                current.append((keys[i], payload))
            else:
                current.append((keys[i], raw))
        if write:
            self._kv.pipeline_set(write)
            with self._cache_lock:
                for k, v in write:
                    self._cache.set_dirty(k, v)
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.extend(current)
        return len(write)

    # ---- predicate documents (affordance predicate mode) --------------------
    # Stored under the same vcat:: prefix and relayed to LMDB exactly like
    # category writes. Every doc carries "category"/"confidence" so a legacy
    # reader sees a well-formed entry.

    def get_docs_batch(self, names: List[str]) -> Dict[str, Optional[dict]]:
        """name -> doc (or None), in one batch round-trip for cache misses."""
        self._flush_soon()
        keys = [self._key(n) for n in names]
        snap = self._fetch_raw(list(dict.fromkeys(keys)))
        out: Dict[str, Optional[dict]] = {}
        for n, k in zip(names, keys):
            raw = snap.get(k, _MISS)
            doc = None
            if raw is not _MISS and raw is not None:
                doc = self._parse(k, raw)
            out[n] = doc
        return out

    # put_doc takes ``merged_from=(doc, merge_docs(doc, payload))`` from a
    # caller that already merged (predicate_store.StoreLexicon.commit).
    accepts_merge_hint = True

    def put_doc(self, name: str, doc: dict, merged_from=None) -> None:
        """Queue a doc write. Docs with counters merge (merge.py) in L1 now,
        in KV at flush (CAS) and in LMDB in affordance-lmdb-sync.

        ``merged_from=(base, merged)``: the caller's merge of ``doc`` into
        ``base``; used for L1 when L1 holds exactly ``base`` (the parsed doc
        object this store returned), instead of merging again."""
        from .merge import encode, is_mergeable, totals
        new_doc = totals(dict(doc))
        payload = encode(new_doc)
        key = self._key(name)
        with self._cache_lock:
            self._payload_docs[key] = (payload, new_doc)
            old = self._cache.get(key)
            old = old if isinstance(old, (bytes, bytearray)) else None
            base = self._parse(key, old) if old is not None else None
            if merged_from is not None and base is not None and base is merged_from[0] \
                    and isinstance(base, dict) and ("_g" in new_doc or "_go" in new_doc):
                m = merged_from[1]
                raw = encode(m)
                self._merge_memo[key] = (base, payload, raw, m)
                self._remember(key, raw, m)
                self._cache.set_dirty(key, raw)
            else:
                self._cache.set_dirty(key, self._merge(key, old, payload))
        with self._pending_lock:
            self._pending_writes.append((key, payload))
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.append((key, payload))

    def seed_docs(self, entries: List[Tuple[str, dict]]) -> int:
        """Seed lexicon docs (one batch_get + one batch_set).

        Written when absent or seeded under another seed version (counters
        kept, learned docs untouched). Every seed key's current value warms
        L1 and is RELAYED to LMDB (drain_pending_lmdb_writes ->
        affordance-lmdb-sync), so a reader that never seeded itself -- an
        unseeded pod with kv_on_cache_miss=0 -- sees the lexicon from LMDB."""
        from .merge import encode, merge_docs
        from .seed_version import needs_seed
        if not entries:
            return 0
        keys = [self._key(n) for n, _ in entries]
        existing = self._kv.pipeline_get(keys)
        write, current = [], []
        for k, (_, doc) in zip(keys, entries):
            raw = existing.get(k, (None, None))[0]
            old = None
            if raw is not None:
                try:
                    old = json.loads(raw.decode("utf-8"))
                except Exception:
                    old = None
            if needs_seed(old, doc.get("seed_version")):
                v = encode(merge_docs(old, doc, prefer_b=True))
                write.append((k, v))
                current.append((k, v))
            else:
                current.append((k, raw))
        if write:
            self._kv.pipeline_set(write)
        with self._cache_lock:
            for k, v in current:
                if v is not None:
                    self._cache.set_dirty(k, v)
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.extend(current)
        return len(write)

    def reset_session_state(self) -> Dict[str, int]:
        """Drop everything this process holds from the session bucket: the L1
        and the writes not yet sent or relayed. Called after the bucket is
        flushed (cold reset, src/storage/session_reset.py). A write made before
        the flush must not land in the emptied bucket. Returns the number
        dropped from each."""
        with self._cache_lock:
            l1 = len(self._cache)
            self._cache.clear()
            for memo in (self._parsed, self._parsed_kv, self._payload_docs, self._merge_memo):
                memo.clear()
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
        the affordance-lmdb-sync worker via NATS.
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
