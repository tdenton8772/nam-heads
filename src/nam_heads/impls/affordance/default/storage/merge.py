"""Merge-on-write for predicate-lexicon docs (affordance store, vcat::).

A predicate doc's counters are G-counters: ``"_g": {field: {witness: n}}``
and ``"_go": {ontology: {witness: n}}``. Each writer (head pod) only ever
raises its OWN witness entry; two docs merge by taking, per field and
witness, the larger value. That merge is commutative, associative and
idempotent, so concurrent pods, replays of the ``affordance.writes`` stream
and every node's affordance-lmdb-sync converge on the same doc and no pod's
increments are lost. The readable totals (``count``, ``bare``, ``inflected``,
``by_ontology``) are recomputed from the counters on every merge.

Non-counter fields (kind, lemma, value, target, category, seed_version) are
descriptive: a seed doc's fields win over a learned doc's, and a newer seed
version over an older one.

Used by
* ``CouchbaseKVCategoryStore._flush_pending`` -- mergeable payloads go
  through ``batch_cas_merge_set`` (one batch_get, then pipeline_add /
  pipeline_set_cas; only lost races retry per key through ``cas_merge_set``);
* ``services/nats_worker.run_affordance_lmdb_sync`` -- ``merge_with_lmdb``
  folds each write into the node's LMDB copy before ``batch_put``, and the
  KV write-through uses ``batch_cas_merge_set``.
Plain category payloads ({category, confidence}) keep last-writer-wins.
"""

import json
import logging
import os
import socket
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ERR_KEY_ENOENT = 0x01
ERR_KEY_EEXISTS = 0x02
COUNTER_FIELDS = ("count", "bare", "inflected")


def witness_id() -> str:
    """This writer's counter identity: NAM_AFF_WITNESS, else the pod name."""
    return os.environ.get("NAM_AFF_WITNESS") or os.environ.get("HOSTNAME") or socket.gethostname() or "local"


def _load(payload) -> Optional[dict]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    try:
        return json.loads(payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload)
    except Exception:
        return None


def is_mergeable(payload) -> bool:
    d = _load(payload)
    return isinstance(d, dict) and ("_g" in d or "_go" in d)


def totals(doc: dict) -> dict:
    """Recompute the readable totals from the counters (in place)."""
    g = doc.get("_g") or {}
    for f in COUNTER_FIELDS:
        if f in g:
            doc[f] = sum(g[f].values())
    go = doc.get("_go") or {}
    if go:
        doc["by_ontology"] = {o: sum(w.values()) for o, w in sorted(go.items())}
    return doc


def _seed_rank(d: dict):
    return (1 if d.get("kind") == "seed" else 0, str(d.get("seed_version", "")))


def merge_docs(a: Optional[dict], b: Optional[dict], prefer_b: bool = False) -> dict:
    """Merge two docs: counters by per-witness max; descriptive fields from
    the seed doc (``prefer_b``: from ``b`` -- the seeding path, where the
    incoming seed replaces an older version's fields). Without ``prefer_b``
    the choice is deterministic and order-independent (seed over learned,
    then by seed_version string)."""
    if not a:
        return totals(dict(b or {}))
    if not b:
        return totals(dict(a))
    if prefer_b:
        hi, lo = b, a
    else:
        hi, lo = (a, b) if _seed_rank(a) >= _seed_rank(b) else (b, a)
    out = {k: v for k, v in lo.items() if k not in ("_g", "_go")}
    out.update({k: v for k, v in hi.items() if k not in ("_g", "_go")})
    for key in ("_g", "_go"):
        m: Dict[str, Dict[str, int]] = {}
        for src in (a.get(key) or {}, b.get(key) or {}):
            for f, ws in src.items():
                dst = m.setdefault(f, {})
                for w, n in ws.items():
                    dst[w] = max(dst.get(w, 0), int(n))
        if m:
            out[key] = {f: dict(sorted(ws.items())) for f, ws in sorted(m.items())}
    return totals(out)


def encode(doc: dict) -> bytes:
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode("utf-8")


def merge_payloads(old, new) -> bytes:
    """Merged payload; ``new`` unchanged when either side is not mergeable
    (a counter-free seed doc merges too: it carries descriptive fields)."""
    a, b = _load(old), _load(new)
    if not is_mergeable(new) and not is_mergeable(old):
        return new if isinstance(new, (bytes, bytearray)) else encode(b or {})
    if a is None or b is None:
        return new if isinstance(new, (bytes, bytearray)) else encode(b or {})
    return encode(merge_docs(a, b))


def merge_payload_docs(a: Optional[dict], b: Optional[dict]) -> Optional[dict]:
    """``merge_payloads`` on parsed docs: returns the merged doc, or ``b``
    itself when merge_payloads would return ``new`` unchanged (either side
    not mergeable / missing). ``a`` and ``b`` are not modified."""
    a_m = isinstance(a, dict) and ("_g" in a or "_go" in a)
    b_m = isinstance(b, dict) and ("_g" in b or "_go" in b)
    if not b_m and not a_m:
        return b
    if a is None or b is None:
        return b
    return merge_docs(a, b)


def merge_with_lmdb(lmdb, writes: List[Tuple]) -> List[Tuple]:
    """Fold each mergeable write into the value already in ``lmdb`` (and
    into earlier writes of the same key in this batch). Others pass through."""
    flags = [is_mergeable(v) for _, v in writes]
    if not any(flags):
        return list(writes)
    keys = [k.decode("utf-8") if isinstance(k, bytes) else k for k, _ in writes]
    wanted = list(dict.fromkeys(k for k, f in zip(keys, flags) if f))
    existing = lmdb.batch_get(wanted) if lmdb is not None else {}
    pending: Dict[str, bytes] = {}
    out = []
    for (k, v), ks, f in zip(writes, keys, flags):
        if f:
            v = merge_payloads(pending.get(ks, existing.get(ks)), v)
            pending[ks] = v
        out.append((k, v))
    return out


def apply_sync_batch(lmdb, kv, writes: List[Tuple]) -> Tuple[int, int]:
    """What affordance-lmdb-sync does with one ``affordance.writes`` message:
    merge counter docs into the LMDB copy, batch_put, then write KV --
    plain rows in one batch_set, counter docs by CAS merge. Returns
    (lmdb rows written, kv rows written)."""
    merged = merge_with_lmdb(lmdb, writes)
    written = lmdb.batch_put(merged) if lmdb is not None else 0
    kv_n = 0
    if kv is not None:
        rows = [(k.decode("utf-8") if isinstance(k, bytes) else k, v) for k, v in merged]
        plain = [(k, v) for k, v in rows if not is_mergeable(v)]
        if plain:
            kv.pipeline_set(plain)
            kv_n += len(plain)
        mergeable = [(k, v) for k, v in rows if is_mergeable(v)]
        if mergeable:
            stored = batch_cas_merge_set(kv, mergeable)
            kv_n += sum(1 for v in stored.values() if v is not None)
    return written, kv_n


def _default_merge(key: str, cur, payload) -> bytes:
    return merge_payloads(cur, payload)


def cas_merge_set(kv, key: str, payload: bytes, retries: int = 8,
                  merge=_default_merge) -> Optional[bytes]:
    """Write a mergeable payload to KV without losing a concurrent writer.
    Returns the stored (merged) payload, or None if it gave up.
    ``merge(key, current, payload)`` must equal ``merge_payloads(current,
    payload)`` (a caller may pass one that reuses parsed docs)."""
    for _ in range(retries):
        cur, cas = kv.get(key)
        merged = merge(key, cur, payload)
        if cur is None:
            if kv.add(key, merged):
                return merged
            continue
        if cur == merged:
            return merged
        status = kv.set(key, merged, cas=cas or 0)
        if status == 0:
            return merged
        if status not in (ERR_KEY_EEXISTS, ERR_KEY_ENOENT):
            logger.warning("vcat cas_merge_set key=%s status=%s", key, status)
            return None
    logger.warning("vcat cas_merge_set gave up after %d conflicts key=%s", retries, key)
    return None


def batch_cas_merge_set(kv, items: List[Tuple[str, bytes]],
                        merge=_default_merge) -> Dict[str, Optional[bytes]]:
    """cas_merge_set for many keys in a constant number of round trips.

    One batch_get (values + CAS) for every key, merge in memory, skip keys
    whose merged value equals what is stored, then one pipeline_add for absent
    keys and one pipeline_set_cas for changed ones. Only keys that lose a race
    (or hit a routing/connection error) fall back to the per-key
    cas_merge_set loop, so the stored result is the same as calling
    cas_merge_set on each key. A KV without the batch calls takes the
    per-key path for every key. ``merge`` as in ``cas_merge_set``.
    """
    out: Dict[str, Optional[bytes]] = {}
    if not items:
        return out
    # Several payloads for one key in a batch: fold them first (merge is
    # commutative and idempotent), so each key is written once.
    pending: Dict[str, bytes] = {}
    for k, v in items:
        pending[k] = merge(k, pending[k], v) if k in pending else v
    if not (hasattr(kv, "pipeline_get") and hasattr(kv, "pipeline_add") and hasattr(kv, "pipeline_set_cas")):
        for k, v in pending.items():
            out[k] = cas_merge_set(kv, k, v, merge=merge)
        return out

    keys = list(pending)
    current = kv.pipeline_get(keys)
    adds: List[Tuple[str, bytes]] = []
    sets: List[Tuple[str, bytes, int]] = []
    merged_for: Dict[str, bytes] = {}
    for k in keys:
        cur, cas = current.get(k, (None, None))
        merged = merge(k, cur, pending[k])
        merged_for[k] = merged
        if cur is None:
            adds.append((k, merged))
        elif cur == merged:
            out[k] = merged
        else:
            sets.append((k, merged, cas or 0))
    retry: List[str] = []
    for k, ok in _statuses(kv.pipeline_add(adds) if adds else []):
        if ok:
            out[k] = merged_for[k]
        else:
            retry.append(k)
    for k, ok in _statuses(kv.pipeline_set_cas(sets) if sets else []):
        if ok:
            out[k] = merged_for[k]
        else:
            retry.append(k)
    for k in retry:
        out[k] = cas_merge_set(kv, k, pending[k], merge=merge)
    return out


def _statuses(results) -> List[Tuple[str, bool]]:
    """(key, stored?) from a batch store result list of (key, status)."""
    return [(k, st == 0) for k, st in results]
