"""
CouchbaseKV-backed entity store.

Uses the SESSION bucket (``NAM_SESSION_BUCKET``, default "session") with key
prefixes:
  ent::<entity_id>   -> JSON {surface_forms: [...], metadata: {...}}
  surf::<surface>    -> entity_id (plain text)
  cand::<key>        -> JSON [[entity_id, surface_form], ...]

The only construction path is storage/factory.py, which passes the session KV
(KVRegistry "ingest-session"). This matters for flush safety: session is
deliberately preserved as the warm head-store cache; a standard reset flushes
the derived buckets only (main, nam, graph, vector: the supervisor flush task's
default). These stores survive a standard reset.

Thread-safe via CAS on writes; reads are eventually consistent.
In-memory cache eliminates repeat KV lookups for common entities across batches.
"""
import json
import logging
import os
import queue
import threading
import time
from typing import Optional, Dict, Any, List, Tuple

from .base import EntityStore

logger = logging.getLogger(__name__)

_MISS = object()


class CouchbaseKVEntityStore(EntityStore):
    def __init__(self, kv, lmdb_cache=None, kv_on_cache_miss: bool = True):
        """
        Parameters
        ----------
        kv : CouchbaseKV
            Connected CouchbaseKV client pointing at the metrics bucket.
        lmdb_cache : LmdbEntityCache or None
            Optional node-level L2 cache (LMDB on tmpfs, read-only in workers).
        kv_on_cache_miss : bool
            If False, skip L3 KV lookups on L2 miss (only use L1 cache + L2 LMDB).
        """
        self._kv = kv
        self._lmdb = lmdb_cache
        self._kv_on_miss = kv_on_cache_miss  # If False, skip L3 KV on L2 miss
        # In-memory cache for surf:: keys (most frequently queried)
        self._surf_cache: Dict[str, object] = {}
        self._surf_cache_lock = threading.Lock()
        # In-memory cache for cand:: keys
        self._cand_cache: Dict[str, object] = {}
        self._cand_cache_lock = threading.Lock()
        # In-memory cache for redirect pointers (append-only, never invalidated).
        # Maps entity_id -> target_id (or _MISS sentinel for known non-redirects).
        self._redir_cache: Dict[str, object] = {}
        # Upper bound on each in-process read cache (0 = unbounded). Ingest's
        # key set is bounded by the corpus; a query pod reading through to KV
        # adds keys for every distinct query, so it sets a bound. Over the
        # bound the cache is cleared -- these are caches, a miss re-reads KV.
        self._cache_max = int(os.environ.get("NAM_ENTITY_STORE_CACHE_MAX", "0"))
        # Async write queue: (key, value) pairs buffered here, flushed by bg thread
        self._async_writes = os.environ.get("NAM_ENTITY_ASYNC_WRITES", "0").lower() in ("1", "true", "yes")
        self._write_q: queue.Queue = queue.Queue()
        if self._async_writes:
            self._write_thread = threading.Thread(target=self._flush_loop, daemon=True, name="entity-write-flush")
            self._write_thread.start()
            logger.info("CouchbaseKVEntityStore: async writes enabled")
        # Pending writes for external LMDB sync (drained by NATS worker)
        self._pending_lmdb_writes: List[Tuple[str, bytes]] = []
        self._pending_lmdb_lock = threading.Lock()

    def _flush_loop(self):
        """Background thread: batch-flush queued KV writes."""
        while True:
            batch: List[Tuple[str, bytes]] = []
            try:
                # Block until at least one item
                item = self._write_q.get(timeout=1.0)
                batch.append(item)
            except queue.Empty:
                continue
            # Drain up to 500 more without blocking
            for _ in range(500):
                try:
                    batch.append(self._write_q.get_nowait())
                except queue.Empty:
                    break
            if batch:
                try:
                    self._kv.pipeline_set(batch)
                except Exception as e:
                    logger.warning("entity async write flush failed: %s (%d items)", e, len(batch))

    def _trim_read_caches(self) -> None:
        if self._cache_max <= 0:
            return
        if len(self._surf_cache) > self._cache_max:
            with self._surf_cache_lock:
                self._surf_cache.clear()
        if len(self._cand_cache) > self._cache_max:
            with self._cand_cache_lock:
                self._cand_cache.clear()

    def resolve_surface_form(self, surface: str, entity_type: Optional[str] = None) -> Optional[str]:
        self._trim_read_caches()
        if entity_type:
            key = f"surf::{entity_type}::{surface}"
        else:
            key = f"surf::{surface}"
        # Check cache first
        with self._surf_cache_lock:
            cached = self._surf_cache.get(key)
        if cached is _MISS:
            return None
        if cached is not None:
            return cached.decode("utf-8")
        if self._kv_on_miss:
            raw, _ = self._kv.get(key)
            with self._surf_cache_lock:
                self._surf_cache[key] = raw if raw is not None else _MISS
            if raw:
                return raw.decode("utf-8")
        return None

    def resolve_surface_forms_batch(
        self,
        queries: List[Tuple[str, Optional[str]]],
    ) -> Dict[str, Optional[str]]:
        """Batch resolve via scoped keys only (L1 -> L2 -> L3).

        Ontology always provides entity_type, so unscoped fallback is unnecessary.
        """
        if not queries:
            return {}
        self._trim_read_caches()

        # Build scoped keys
        scoped_keys = []
        key_to_query: Dict[str, Tuple[str, Optional[str]]] = {}

        for surface, entity_type in queries:
            if entity_type:
                key = f"surf::{entity_type}::{surface}"
            else:
                key = f"surf::{surface}"
            scoped_keys.append(key)
            key_to_query[key] = (surface, entity_type)

        unique_keys = list(dict.fromkeys(scoped_keys))

        # L1: check in-memory cache
        uncached: List[str] = []
        with self._surf_cache_lock:
            for key in unique_keys:
                if key not in self._surf_cache:
                    uncached.append(key)

        # L2: check LMDB for L1 misses
        if uncached and self._lmdb and self._lmdb.available:
            lmdb_hits = self._lmdb.batch_get(uncached)
            if lmdb_hits:
                still_uncached = []
                with self._surf_cache_lock:
                    for key in uncached:
                        lval = lmdb_hits.get(key)
                        if lval is not None:
                            self._surf_cache[key] = lval if lval != b"__LMDB_MISS__" else _MISS
                        else:
                            still_uncached.append(key)
                uncached = still_uncached

        # L3: fetch remaining misses from KV (only if policy allows)
        if uncached and self._kv_on_miss:
            raw_results = self._kv.pipeline_get(uncached)
            with self._surf_cache_lock:
                for key in uncached:
                    val, _ = raw_results.get(key, (None, None))
                    self._surf_cache[key] = val if val is not None else _MISS

        # Build results
        with self._surf_cache_lock:
            result: Dict[Tuple[str, Optional[str]], Optional[str]] = {}
            for key in dict.fromkeys(scoped_keys):
                query = key_to_query[key]
                raw = self._surf_cache.get(key, _MISS)
                if raw is not _MISS and raw is not None:
                    result[query] = raw.decode("utf-8")
                else:
                    result[query] = None
        return result

    def resolve_fuzzy_batch(
        self,
        queries: List[tuple],
        max_candidates: int = 50,
    ) -> Dict[int, Optional[str]]:
        """Optimized batch fuzzy resolution.

        Collects all fuzzy keys from all queries, does a single batch
        cand:: lookup, then scores per-query candidates.
        """
        if not queries:
            return {}

        # Phase 1: collect all fuzzy keys with source tracking
        all_keys: List[str] = []
        key_to_queries: Dict[str, List[int]] = {}  # key -> list of query indices

        for qi, (canonical, entity_type, comparator) in enumerate(queries):
            qkeys = comparator.generate_keys(canonical)
            for k in qkeys:
                if k not in key_to_queries:
                    key_to_queries[k] = []
                    all_keys.append(k)
                key_to_queries[k].append(qi)

        if not all_keys:
            return {i: None for i in range(len(queries))}

        # Phase 2: single batch lookup
        by_key = self.lookup_candidate_keys_by_key(all_keys)

        # Phase 3: partition results per query and score
        per_query: List[List[tuple]] = [[] for _ in queries]
        for k, pairs in by_key.items():
            for qi in key_to_queries.get(k, []):
                per_query[qi].extend(pairs)

        result: Dict[int, Optional[str]] = {}
        for qi, (canonical, entity_type, comparator) in enumerate(queries):
            seen: set = set()
            unique = []
            for eid, sf in per_query[qi]:
                if eid not in seen:
                    seen.add(eid)
                    unique.append((eid, sf))
                if len(unique) >= max_candidates:
                    break
            best_id = None
            best_score = 0.0
            threshold = comparator.threshold()
            for eid, csf in unique:
                score = comparator.match(canonical, csf)
                # Equal scores resolve to the smallest id, not to whichever
                # candidate was written to cand:: first.
                if score >= threshold and (
                        score > best_score
                        or (score == best_score and best_id is not None and eid < best_id)):
                    best_score = score
                    best_id = eid
            result[qi] = best_id

        return result

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        key = f"ent::{entity_id}"
        raw, _ = self._kv.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def registered_in_kv(self, entity_ids_by_surface: Dict[str, str],
                         entity_type: Optional[str] = None) -> set:
        """Surfaces already registered IN THE KV, whatever the cache policy.

        A surface counts as registered when either ``ent::<entity_id>`` or its
        ``surf::`` key exists. One pipelined batch_get for the whole set.

        Deliberately bypasses ``kv_on_cache_miss``: that flag is a READ-PATH
        latency policy for ingest workers fronted by LMDB. Used as an existence
        check for seeding it answers "absent" for everything on a pod with a
        cold cache (nam-query), so every start re-wrote every seed.

        Raises DataServiceUnavailable-style RuntimeError when any read failed,
        because "could not read" must not be taken as "absent".
        """
        from src.storage.backends.couchbase_kv import record_read_failures

        if not entity_ids_by_surface:
            return set()
        pairs = {}
        for surface, entity_id in entity_ids_by_surface.items():
            surf_key = f"surf::{entity_type}::{surface}" if entity_type else f"surf::{surface}"
            pairs[surface] = (f"ent::{entity_id}", surf_key)
        keys = [k for pair in pairs.values() for k in pair]
        with record_read_failures() as rec:
            got = self._kv.pipeline_get(keys)
        if rec.failed:
            raise RuntimeError(
                "registered_in_kv: %d of %d reads failed; not treating them as absent"
                % (rec.count, len(keys)))
        present = set()
        for surface, (ent_key, surf_key) in pairs.items():
            if got.get(ent_key, (None, None))[0] or got.get(surf_key, (None, None))[0]:
                present.add(surface)
        return present

    def create_entity(
        self,
        entity_id: str,
        surface_forms: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        ent_key = f"ent::{entity_id}"
        doc = {"surface_forms": list(surface_forms), "metadata": dict(metadata), "created_at": time.time()}
        self._kv.set(ent_key, json.dumps(doc).encode("utf-8"))
        entity_type = metadata.get("entity_type")
        primary_form = surface_forms[0] if surface_forms else None

        # Batch-check existence of all surface form keys in one pipeline
        check_keys = []
        for sf in surface_forms:
            if entity_type:
                check_keys.append(f"surf::{entity_type}::{sf}")
            else:
                check_keys.append(f"surf::{sf}")

        if check_keys:
            existing = self._kv.pipeline_get(check_keys)
            entity_id_b = entity_id.encode("utf-8")
            writes = []
            for key in check_keys:
                val, _ = existing.get(key, (None, None))
                if not val:
                    writes.append((key, entity_id_b))
            if writes:
                self._kv.pipeline_set(writes)
                # Update surf cache for new writes
                with self._surf_cache_lock:
                    for key, val in writes:
                        self._surf_cache[key] = val

    def create_entities_batch(
        self,
        entities: List[Tuple[str, List[str], Dict[str, Any]]],
    ) -> None:
        """Batch create multiple entities.

        When async writes are enabled, updates in-memory cache immediately
        and queues KV writes for background flush. This eliminates blocking
        on KV round-trips in the hot path.
        """
        if not entities:
            return

        # Build all write payloads
        now = time.time()
        ent_writes = []
        surf_writes = []
        for entity_id, surface_forms, metadata in entities:
            ent_key = f"ent::{entity_id}"
            doc = {"surface_forms": list(surface_forms), "metadata": dict(metadata), "created_at": now}
            ent_writes.append((ent_key, json.dumps(doc).encode("utf-8")))

            entity_type = metadata.get("entity_type")
            entity_id_b = entity_id.encode("utf-8")
            for sf in surface_forms:
                if entity_type:
                    surf_writes.append((f"surf::{entity_type}::{sf}", entity_id_b))
                else:
                    surf_writes.append((f"surf::{sf}", entity_id_b))

        # Always update cache immediately (for same-process resolution)
        with self._surf_cache_lock:
            for key, val in surf_writes:
                # Only set if not already cached (first writer wins)
                if key not in self._surf_cache or self._surf_cache[key] is _MISS:
                    self._surf_cache[key] = val

        # Capture writes for external LMDB sync (published to NATS by worker)
        with self._pending_lmdb_lock:
            for key, val in surf_writes:
                self._pending_lmdb_writes.append((key, val))

        if self._async_writes:
            # Queue all writes for background flush
            for item in ent_writes:
                self._write_q.put_nowait(item)
            for item in surf_writes:
                self._write_q.put_nowait(item)
        else:
            # Synchronous path: write entity docs, check+write surf keys
            self._kv.pipeline_set(ent_writes)
            if surf_writes:
                # Deduplicate and check existence
                check_keys = list(dict(surf_writes).keys())
                existing = self._kv.pipeline_get(check_keys)
                new_writes = []
                surf_dict = dict(surf_writes)
                for key in check_keys:
                    val, _ = existing.get(key, (None, None))
                    if not val:
                        new_writes.append((key, surf_dict[key]))
                if new_writes:
                    self._kv.pipeline_set(new_writes)

    def lookup_candidate_keys_by_key(
        self,
        keys: List[str],
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Batch lookup with cand:: cache. Only fetches cache misses from KV."""
        if not keys:
            return {}
        self._trim_read_caches()
        cand_keys = [f"cand::{k}" for k in keys]
        unique_cand = list(dict.fromkeys(cand_keys))

        # Check L1 cache, collect misses
        uncached: List[str] = []
        with self._cand_cache_lock:
            for ck in unique_cand:
                if ck not in self._cand_cache:
                    uncached.append(ck)

        # L2: check LMDB for L1 misses
        if uncached and self._lmdb and self._lmdb.available:
            lmdb_hits = self._lmdb.batch_get(uncached)
            if lmdb_hits:
                still_uncached = []
                with self._cand_cache_lock:
                    for ck in uncached:
                        lval = lmdb_hits.get(ck)
                        if lval is not None:
                            self._cand_cache[ck] = lval if lval != b"__LMDB_MISS__" else _MISS
                        else:
                            still_uncached.append(ck)
                uncached = still_uncached

        # L3: fetch remaining from KV (only if policy allows)
        if uncached and self._kv_on_miss:
            raw_results = self._kv.pipeline_get(uncached)
            with self._cand_cache_lock:
                for ck in uncached:
                    val, _ = raw_results.get(ck, (None, None))
                    self._cand_cache[ck] = val if val is not None else _MISS
        elif uncached:
            # Mark all uncached keys as misses (no KV fallback)
            with self._cand_cache_lock:
                for ck in uncached:
                    self._cand_cache[ck] = _MISS

        with self._cand_cache_lock:
            cache_snap = {ck: self._cand_cache.get(ck, _MISS) for ck in unique_cand}

        result: Dict[str, List[Tuple[str, str]]] = {}
        for i, k in enumerate(keys):
            cand_key = cand_keys[i]
            if k in result:
                continue
            raw = cache_snap.get(cand_key, _MISS)
            pairs: List[Tuple[str, str]] = []
            if raw is not _MISS and raw is not None:
                try:
                    entries = json.loads(raw.decode("utf-8"))
                    for e in entries:
                        if isinstance(e, list) and len(e) == 2:
                            pairs.append((e[0], e[1]))
                except Exception:
                    pass
            result[k] = pairs
        return result

    def add_surface_form(
        self,
        entity_id: str,
        surface_form: str,
        max_forms: int,
    ) -> None:
        entity = self.get_entity(entity_id)
        if not entity:
            return
        forms = entity.get("surface_forms", [])
        meta = entity.get("metadata", {})
        entity_type = meta.get("entity_type")
        if surface_form not in forms:
            forms.append(surface_form)
            if len(forms) > max_forms:
                forms.pop(0)
        # Write the type-scoped key for the new surface form
        if entity_type:
            scoped_key = f"surf::{entity_type}::{surface_form}"
            existing, _ = self._kv.get(scoped_key)
            if not existing:
                entity_id_b = entity_id.encode("utf-8")
                self._kv.set(scoped_key, entity_id_b)
                with self._surf_cache_lock:
                    self._surf_cache[scoped_key] = entity_id_b
        self.create_entity(entity_id, forms, meta)

    def store_candidate_keys(
        self,
        entity_id: str,
        surface_form: str,
        keys: List[str],
    ) -> None:
        if not keys:
            return
        # Batch-read all candidate keys in one pipeline
        cand_keys = [f"cand::{k}" for k in keys]
        existing = self._kv.pipeline_get(cand_keys)

        pair = [entity_id, surface_form]
        writes = []
        for cand_key in cand_keys:
            raw, _ = existing.get(cand_key, (None, None))
            if raw:
                try:
                    entries = json.loads(raw.decode("utf-8"))
                except Exception:
                    entries = []
            else:
                entries = []
            if pair not in entries:
                entries.append(pair)
                writes.append((cand_key, json.dumps(entries).encode("utf-8")))
        if writes:
            self._kv.pipeline_set(writes)

    def store_candidate_keys_bulk(
        self,
        items: List[Tuple[str, str, List[str]]],
    ) -> None:
        """Batch store candidate keys for multiple (entity_id, surface_form, keys) tuples.

        Reads all candidate keys in one pipeline, merges, then writes all in one pipeline.
        Updates cand:: cache on write.
        """
        if not items:
            return
        # Collect all unique cand keys
        all_cand_keys = []
        key_to_pairs: dict = {}  # cand_key -> list of [entity_id, surface_form]
        for entity_id, surface_form, keys in items:
            pair = [entity_id, surface_form]
            for k in keys:
                cand_key = f"cand::{k}"
                all_cand_keys.append(cand_key)
                if cand_key not in key_to_pairs:
                    key_to_pairs[cand_key] = []
                key_to_pairs[cand_key].append(pair)

        if not all_cand_keys:
            return

        # Deduplicate keys for the batch read
        unique_keys = list(dict.fromkeys(all_cand_keys))
        existing = self._kv.pipeline_get(unique_keys)

        # Merge and collect writes
        writes = []
        for cand_key in unique_keys:
            raw, _ = existing.get(cand_key, (None, None))
            if raw:
                try:
                    entries = json.loads(raw.decode("utf-8"))
                except Exception:
                    entries = []
            else:
                entries = []
            changed = False
            for pair in key_to_pairs[cand_key]:
                if pair not in entries:
                    entries.append(pair)
                    changed = True
            if changed:
                writes.append((cand_key, json.dumps(entries).encode("utf-8")))

        if writes:
            self._kv.pipeline_set(writes)
            # Update L1 cand cache
            with self._cand_cache_lock:
                for key, val in writes:
                    self._cand_cache[key] = val
            # Capture writes for external LMDB sync
            with self._pending_lmdb_lock:
                for key, val in writes:
                    self._pending_lmdb_writes.append((key, val))

    def lookup_candidate_keys(
        self,
        keys: List[str],
    ) -> List[Tuple[str, str]]:
        if not keys:
            return []
        # Batch-read all candidate keys in one pipeline
        cand_keys = [f"cand::{k}" for k in keys]
        existing = self._kv.pipeline_get(cand_keys)

        results = []
        for cand_key in cand_keys:
            raw, _ = existing.get(cand_key, (None, None))
            if raw:
                try:
                    entries = json.loads(raw.decode("utf-8"))
                    for pair in entries:
                        if isinstance(pair, list) and len(pair) == 2:
                            results.append((pair[0], pair[1]))
                except Exception:
                    pass
        return results

    # --- Redirect support (entity ID migration) ---

    def write_redirect(self, old_entity_id: str, new_entity_id: str) -> None:
        if old_entity_id == new_entity_id:
            return

        redir_key = f"redir::{old_entity_id}"

        # Check cache first — idempotent skip
        cached = self._redir_cache.get(old_entity_id, _MISS)
        if cached is not _MISS and cached == new_entity_id:
            return

        # Check KV — idempotent skip
        raw, _ = self._kv.get(redir_key)
        if raw is not None and raw.decode("utf-8") == new_entity_id:
            self._redir_cache[old_entity_id] = new_entity_id
            return

        # Write forward pointer
        self._kv.set(redir_key, new_entity_id.encode("utf-8"))
        self._redir_cache[old_entity_id] = new_entity_id

        # Update reverse index
        rev_key = f"redir_rev::{new_entity_id}"
        raw_rev, _ = self._kv.get(rev_key)
        if raw_rev:
            try:
                rev_list = json.loads(raw_rev.decode("utf-8"))
            except Exception:
                rev_list = []
        else:
            rev_list = []

        if old_entity_id not in rev_list:
            rev_list.append(old_entity_id)
            self._kv.set(rev_key, json.dumps(rev_list).encode("utf-8"))

    def resolve_redirect(self, entity_id: str) -> str:
        visited = set()
        current = entity_id
        for _ in range(10):
            if current in visited:
                break
            visited.add(current)

            # Check cache
            cached = self._redir_cache.get(current, _MISS)
            if cached is _MISS:
                # Not in cache — check KV
                redir_key = f"redir::{current}"
                raw, _ = self._kv.get(redir_key)
                if raw is None:
                    self._redir_cache[current] = None
                    break
                next_id = raw.decode("utf-8")
                self._redir_cache[current] = next_id
                current = next_id
            elif cached is None:
                # Cached as "no redirect"
                break
            else:
                current = cached
        return current

    def reverse_redirects(self, canonical_id: str) -> List[str]:
        result = []
        visited = {canonical_id}
        queue = self._get_reverse_list(canonical_id)

        while queue:
            old_id = queue.pop(0)
            if old_id in visited:
                continue
            visited.add(old_id)
            result.append(old_id)
            queue.extend(self._get_reverse_list(old_id))

        return result

    def _get_reverse_list(self, entity_id: str) -> List[str]:
        rev_key = f"redir_rev::{entity_id}"
        raw, _ = self._kv.get(rev_key)
        if not raw:
            return []
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return []

    def get_raw_batch(self, keys: List[str]) -> Dict[str, bytes]:
        """L2 LMDB, then (when reading through) the session bucket."""
        if not keys:
            return {}
        out: Dict[str, bytes] = {}
        if self._lmdb is not None:
            for k, v in self._lmdb.batch_get(keys).items():
                if v is not None:
                    out[k] = v
        missing = [k for k in keys if k not in out]
        if missing and (self._lmdb is None or self._kv_on_miss):
            for k, (v, _cas) in self._kv.pipeline_get(missing).items():
                if v:
                    out[k] = v
        return out

    def put_raw_batch(self, items: List[Tuple[str, bytes]]) -> None:
        """Written to the session bucket and queued for LMDB sync."""
        if not items:
            return
        with self._pending_lmdb_lock:
            self._pending_lmdb_writes.extend(items)
        if self._async_writes:
            for item in items:
                self._write_q.put_nowait(item)
        else:
            self._kv.pipeline_set(list(items))

    def reset_session_state(self) -> Dict[str, int]:
        """Drop everything this process holds from the session bucket: the read
        caches, and the writes not yet relayed or sent. Called after the bucket
        is flushed (cold reset, src/storage/session_reset.py). A write made
        before the flush must not land in the emptied bucket. Returns the
        number dropped from each."""
        with self._surf_cache_lock:
            surf = len(self._surf_cache)
            self._surf_cache.clear()
        with self._cand_cache_lock:
            cand = len(self._cand_cache)
            self._cand_cache.clear()
        redir = len(self._redir_cache)
        self._redir_cache.clear()
        with self._pending_lmdb_lock:
            relay = len(self._pending_lmdb_writes)
            self._pending_lmdb_writes = []
        queued = 0
        while True:
            try:
                self._write_q.get_nowait()
                queued += 1
            except queue.Empty:
                break
        return {"surf": surf, "cand": cand, "redir": redir,
                "pending_relay": relay, "queued_writes": queued}

    def drain_pending_lmdb_writes(self) -> List[Tuple[str, bytes]]:
        """Return and clear accumulated writes for external LMDB sync.

        Called by NATS worker after encode() to publish writes to
        the lmdb-sync worker via NATS.
        """
        with self._pending_lmdb_lock:
            writes = self._pending_lmdb_writes
            self._pending_lmdb_writes = []
        return writes
