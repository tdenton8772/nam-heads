"""Merge-on-write for context-store values that are vote sets.

A context-store payload is ``{"canonical_value": str, "confidence": float}``.
Most values are plain strings and keep last-writer-wins. A value whose
``canonical_value`` is a JSON object of ``{kind: {"votes": {value: [witness,
...]}}}`` is MERGEABLE: two writes of it are combined by set union, so
concurrent writers (several head pods, and every node's context-lmdb-sync
replaying the same ``context.writes`` stream) cannot lose each other's
votes, and replaying a write is a no-op.

Used by
* ``CouchbaseKVContextStore._flush_pending`` -- KV writes of mergeable values
  go through ``batch_cas_merge_set`` (one batch_get, then pipeline_add /
  pipeline_set_cas; only lost races retry per key through ``cas_merge_set``);
* ``services/nats_worker.run_context_lmdb_sync`` -- ``merge_with_lmdb`` folds
  each incoming write into the node's LMDB copy before ``batch_put``, and
  ``batch_cas_merge_set`` for the KV write-through.
"""

import json
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_WITNESSES = 8
ERR_KEY_EEXISTS = 0x02
ERR_KEY_ENOENT = 0x01


def _votes_doc(payload: Optional[bytes]) -> Optional[dict]:
    if not payload:
        return None
    try:
        outer = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        doc = json.loads(outer["canonical_value"])
    except Exception:
        return None
    if not isinstance(doc, dict) or not doc:
        return None
    for v in doc.values():
        if not isinstance(v, dict) or not isinstance(v.get("votes"), dict):
            return None
    return doc


def is_mergeable(payload: Optional[bytes]) -> bool:
    return _votes_doc(payload) is not None


def merge_payloads(old: Optional[bytes], new: bytes) -> bytes:
    """Union of two vote-set payloads; ``new`` unchanged if either is not one."""
    a, b = _votes_doc(old), _votes_doc(new)
    if a is None or b is None:
        return new
    out = {}
    for kind in sorted(set(a) | set(b)):
        votes = {}
        for src in (a.get(kind, {}).get("votes", {}), b.get(kind, {}).get("votes", {})):
            for value, ws in src.items():
                votes.setdefault(value, set()).update(ws)
        out[kind] = {"votes": {v: sorted(w)[:MAX_WITNESSES] for v, w in sorted(votes.items())}}
    shares = [max(len(w) for w in e["votes"].values()) / max(1, sum(len(w) for w in e["votes"].values()))
              for e in out.values() if e["votes"]]
    return json.dumps({"canonical_value": json.dumps(out, sort_keys=True, separators=(",", ":")),
                       "confidence": round(min(shares), 3) if shares else 0.0}).encode("utf-8")


def merge_with_lmdb(lmdb, writes: List[Tuple]) -> List[Tuple]:
    """Fold each mergeable write into the value already in ``lmdb`` (and into
    earlier writes of the same key in this batch). Others pass through."""
    flags = [is_mergeable(v) for _, v in writes]
    if not any(flags):
        return list(writes)  # nothing to merge: no extra LMDB read
    pending = {}
    out = []
    keys = [k.decode("utf-8") if isinstance(k, bytes) else k for k, _ in writes]
    wanted = list(dict.fromkeys(k for k, f in zip(keys, flags) if f))
    existing = lmdb.batch_get(wanted) if lmdb is not None else {}
    for (k, v), ks, f in zip(writes, keys, flags):
        if f:
            base = pending.get(ks, existing.get(ks))
            v = merge_payloads(base, v)
            pending[ks] = v
        out.append((k, v))
    return out


def cas_merge_set(kv, key: str, payload: bytes, retries: int = 8) -> bool:
    """Write a mergeable payload to KV without losing a concurrent writer's votes."""
    for _ in range(retries):
        cur, cas = kv.get(key)
        merged = merge_payloads(cur, payload)
        if cur is None:
            if kv.add(key, merged):
                return True
            continue  # someone created it meanwhile: merge with theirs
        if cur == merged:
            return True
        status = kv.set(key, merged, cas=cas or 0)
        if status == 0:
            return True
        if status not in (ERR_KEY_EEXISTS, ERR_KEY_ENOENT):
            logger.warning("ctx cas_merge_set key=%s status=%s", key, status)
            return False
    logger.warning("ctx cas_merge_set gave up after %d conflicts key=%s", retries, key)
    return False


def batch_cas_merge_set(kv, items: List[Tuple[str, bytes]]):
    """``cas_merge_set`` for many vote-set payloads in a constant number of
    KV round trips (affordance/storage/merge.batch_cas_merge_set with this
    module's union merge): the stored result is the same as calling
    ``cas_merge_set`` on each key. Returns {key: stored payload or None}."""
    from src.storage.merges.counters import batch_cas_merge_set as _batch
    return _batch(kv, items, merge=lambda _k, cur, new: merge_payloads(cur, new))
