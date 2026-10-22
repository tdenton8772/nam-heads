"""LMDB-backed entity store: reads from shared LMDB, queues writes for async session bucket publish.

Entity resolution reads are sub-microsecond mmap lookups against the shared
LMDB volume (populated by the DaemonSet via DCP from the session bucket).

New entity creates are NOT written locally. Instead they are queued and
drained by the NATS worker after each batch, which writes them async to the
session bucket. The DCP -> DaemonSet -> LMDB path propagates them back for
future reads (eventual consistency, typically <1s).
"""

import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from .base import EntityStore
from .lmdb_cache import LmdbEntityCache

logger = logging.getLogger(__name__)

# Title prefixes for surface-form-based fuzzy resolution.
# Mirrored from EntityEncoder._TITLE_PREFIXES — keep in sync.
_TITLE_PREFIXES = frozenset({
    "president", "vice", "senator", "governor", "mayor", "king", "queen",
    "prince", "princess", "emperor", "empress", "pope", "cardinal",
    "bishop", "reverend", "rabbi", "imam", "sheikh",
    "general", "admiral", "colonel", "captain", "lieutenant", "sergeant",
    "commander", "marshal", "corporal", "private",
    "dr", "prof", "professor", "sir", "dame", "lord", "lady",
    "mr", "mrs", "ms", "miss", "master", "mx",
    "chief", "secretary", "minister", "chancellor", "ambassador",
    "representative", "speaker", "justice", "judge",
})


def _title_stripped(canonical: str) -> Optional[str]:
    """Strip leading title prefixes: 'president_abraham_lincoln' → 'abraham_lincoln'."""
    parts = canonical.split("_")
    if len(parts) < 2:
        return None
    i = 0
    while i < len(parts) - 1 and parts[i] in _TITLE_PREFIXES:
        i += 1
    if i > 0:
        return "_".join(parts[i:])
    return None


def _lookup_keys(canonical: str, entity_type: Optional[str]) -> List[str]:
    """surf:: keys tried for *canonical*, in precedence order.

    Exact canonical before its title-stripped form; within each form the
    typed key before the untyped key. The first key that hits wins.
    """
    forms = [canonical]
    stripped = _title_stripped(canonical)
    if stripped:
        forms.append(stripped)
    keys = []
    for form in forms:
        if entity_type:
            keys.append(f"surf::{entity_type}::{form}")
        keys.append(f"surf::{form}")
    return keys


class LmdbEntityStore(EntityStore):
    """Read from LMDB, queue writes for async session bucket publish."""

    def __init__(self, lmdb_cache: LmdbEntityCache):
        self._lmdb = lmdb_cache
        self._pending_writes: List[Tuple[str, bytes]] = []
        self._pending_lock = threading.Lock()

    def resolve_surface_form(self, surface: str, entity_type: Optional[str] = None) -> Optional[str]:
        if entity_type:
            key = f"surf::{entity_type}::{surface}"
        else:
            key = f"surf::{surface}"
        val = self._lmdb.get(key)
        if val is not None:
            return val.decode("utf-8")
        return None

    def resolve_surface_forms_batch(self, queries):
        keys = []
        key_to_query = {}
        for surface, entity_type in queries:
            if entity_type:
                key = f"surf::{entity_type}::{surface}"
            else:
                key = f"surf::{surface}"
            keys.append(key)
            key_to_query[key] = (surface, entity_type)

        hits = self._lmdb.batch_get(keys)
        result = {}
        for surface, entity_type in queries:
            if entity_type:
                key = f"surf::{entity_type}::{surface}"
            else:
                key = f"surf::{surface}"
            val = hits.get(key)
            result[(surface, entity_type)] = val.decode("utf-8") if val else None
        return result

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return None

    def create_entity(self, entity_id: str, surface_forms: List[str],
                      metadata: Dict[str, Any]) -> None:
        import json as _json
        import time as _time
        entity_type = metadata.get("entity_type")
        entity_id_b = entity_id.encode("utf-8")
        writes = []
        # Write ent:: doc so entity hash can be resolved back to name
        ent_doc = {"surface_forms": list(surface_forms), "metadata": dict(metadata),
                   "created_at": _time.time()}
        writes.append((f"ent::{entity_id}", _json.dumps(ent_doc).encode("utf-8")))
        primary_form = surface_forms[0] if surface_forms else None
        for sf in surface_forms:
            if entity_type:
                writes.append((f"surf::{entity_type}::{sf}", entity_id_b))
            # Untyped key for cross-type resolution: always for primary,
            # also for multi-token forms (specific enough to avoid collisions).
            if sf == primary_form or "_" in sf:
                writes.append((f"surf::{sf}", entity_id_b))
        with self._pending_lock:
            self._pending_writes.extend(writes)

    def create_entities_batch(self, entities):
        for entity_id, surface_forms, metadata in entities:
            self.create_entity(entity_id, surface_forms, metadata)

    def add_surface_form(self, entity_id: str, surface_form: str,
                         max_forms: int) -> None:
        pass

    def get_raw_batch(self, keys: List[str]) -> Dict[str, bytes]:
        if not keys:
            return {}
        return {k: v for k, v in self._lmdb.batch_get(keys).items() if v is not None}

    def put_raw_batch(self, items: List[Tuple[str, bytes]]) -> None:
        """Queued with the entity writes: published to lmdb-sync and the
        session bucket by the worker, back into LMDB over DCP."""
        if items:
            with self._pending_lock:
                self._pending_writes.extend(items)

    def reset_session_state(self) -> Dict[str, int]:
        """Drop the writes not yet published: the session bucket was flushed
        (cold reset, src/storage/session_reset.py). Reads hold nothing in
        process; LMDB itself is emptied by the lmdb-sync worker."""
        with self._pending_lock:
            pending = len(self._pending_writes)
            self._pending_writes = []
        return {"pending_relay": pending}

    def drain_pending_lmdb_writes(self) -> List[Tuple[str, bytes]]:
        """Return and clear pending writes for async session bucket publish."""
        with self._pending_lock:
            writes = self._pending_writes
            self._pending_writes = []
        return writes

    def resolve_fuzzy(self, surface, entity_type, comparator, max_candidates=50):
        """Surface-form resolution via LMDB surf:: lookups.

        Same keys and precedence as resolve_fuzzy_batch (_lookup_keys).
        """
        for key in _lookup_keys(surface, entity_type):
            val = self._lmdb.get(key)
            if val is not None:
                return val.decode("utf-8")
        return None

    def resolve_fuzzy_batch(self, queries, max_candidates=50):
        """Surface-form-based batch resolution via LMDB surf:: lookups.

        For titled entities like 'president_abraham_lincoln', checks if the
        title-stripped form 'abraham_lincoln' already has a surf:: entry.
        Also checks untyped surf:: keys to catch cross-type matches
        (e.g., 'full_name' vs 'titled_entity' for the same person).

        When several keys hit, the winner is fixed by _lookup_keys order
        (exact canonical before title-stripped, typed before untyped) --
        never by set or dict iteration order.
        """
        if not queries:
            return {}

        per_query = [_lookup_keys(canonical, entity_type)
                     for canonical, entity_type, _comp in queries]
        unique_keys = list(dict.fromkeys(k for keys in per_query for k in keys))
        if not unique_keys:
            return {}

        hits = self._lmdb.batch_get(unique_keys)

        result: Dict[int, str] = {}
        for qi, keys in enumerate(per_query):
            for key in keys:
                val = hits.get(key)
                if val is not None:
                    result[qi] = val.decode("utf-8")
                    break

        return result

    def store_candidate_keys(self, entity_id, surface_form, keys):
        pass  # cand:: keys not used — resolution via surf:: lookups

    def store_candidate_keys_bulk(self, items):
        pass  # cand:: keys not used — resolution via surf:: lookups

    def lookup_candidate_keys(self, keys):
        return []

    def lookup_candidate_keys_by_key(self, keys):
        return {}

    # Redirect stubs — LMDB store is read-only, redirects live in session KV
    def write_redirect(self, old_entity_id, new_entity_id):
        pass

    def resolve_redirect(self, entity_id):
        return entity_id

    def reverse_redirects(self, canonical_id):
        return []
