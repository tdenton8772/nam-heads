"""The merge rules a head store's values follow, shared by every process.

A store's key families each declare one of these (``IMPL_META["stores"]``).
The host applies the same function wherever a value meets an older one: the
head's in-process cache on ``put``, the node cache's writer, and the session
bucket's batched compare-and-set. So a merge runs on any implementation's data
in host processes, and that is why the set is closed: a new rule is a change to
this module, reviewed like the rest of the SDK.

Every function is ``merge(old: bytes | None, new: bytes) -> bytes``. ``old`` is
None when the key has no value yet.

- ``lww``           last writer wins (the new value)
- ``first_writer``  keep the existing value
- ``append_set``    a JSON list; the union of both, in first-seen order
- ``vote_set``      ``{"canonical_value": <json {kind: {"votes": {v: [w]}}}>,
                    "confidence"}``: per value, the union of witnesses (at most
                    8); confidence is the weakest kind's top share
- ``g_counter``     JSON doc with ``"_g": {field: {witness: n}}`` and
                    ``"_go": {ontology: {witness: n}}``: per witness, the max;
                    totals recomputed; descriptive fields from the seed doc

``vote_set`` and ``g_counter`` reproduce, byte for byte, NAM's context and
affordance store merges (tests/test_store.py checks this against NAM when
NAM_SRC is set). A value of another shape passes through as ``lww``, as there.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

Merge = Callable[[Optional[bytes], bytes], bytes]

MAX_WITNESSES = 8
COUNTER_FIELDS = ("count", "bare", "inflected")


def lww(old: Optional[bytes], new: bytes) -> bytes:
    return new


def first_writer(old: Optional[bytes], new: bytes) -> bytes:
    return new if old is None else old


def append_set(old: Optional[bytes], new: bytes) -> bytes:
    try:
        a = json.loads(old) if old else []
        b = json.loads(new)
    except ValueError:
        return new
    if not isinstance(a, list) or not isinstance(b, list):
        return new
    out, seen = [], set()
    for item in a + b:
        k = json.dumps(item, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return json.dumps(out, separators=(",", ":")).encode()


# ---- vote_set (context store) ---------------------------------------------------

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


def vote_set(old: Optional[bytes], new: bytes) -> bytes:
    a, b = _votes_doc(old), _votes_doc(new)
    if a is None or b is None:
        return new
    out = {}
    for kind in sorted(set(a) | set(b)):
        votes: Dict[str, set] = {}
        for src in (a.get(kind, {}).get("votes", {}), b.get(kind, {}).get("votes", {})):
            for value, ws in src.items():
                votes.setdefault(value, set()).update(ws)
        out[kind] = {"votes": {v: sorted(w)[:MAX_WITNESSES] for v, w in sorted(votes.items())}}
    shares = [max(len(w) for w in e["votes"].values()) / max(1, sum(len(w) for w in e["votes"].values()))
              for e in out.values() if e["votes"]]
    return json.dumps({"canonical_value": json.dumps(out, sort_keys=True, separators=(",", ":")),
                       "confidence": round(min(shares), 3) if shares else 0.0}).encode("utf-8")


# ---- g_counter (affordance predicate lexicon) -------------------------------------

def _load(payload) -> Optional[dict]:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    try:
        return json.loads(payload.decode("utf-8") if isinstance(payload, (bytes, bytearray)) else payload)
    except Exception:
        return None


def _counters(d) -> bool:
    return isinstance(d, dict) and ("_g" in d or "_go" in d)


def _totals(doc: dict) -> dict:
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


def _merge_docs(a: Optional[dict], b: Optional[dict]) -> dict:
    if not a:
        return _totals(dict(b or {}))
    if not b:
        return _totals(dict(a))
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
    return _totals(out)


def g_counter(old: Optional[bytes], new: bytes) -> bytes:
    a, b = _load(old), _load(new)
    if not _counters(b) and not _counters(a):
        return new if isinstance(new, (bytes, bytearray)) else json.dumps(b or {}).encode()
    if a is None or b is None:
        return new
    return json.dumps(_merge_docs(a, b), sort_keys=True, separators=(",", ":")).encode("utf-8")


MERGES: Dict[str, Merge] = {
    "lww": lww,
    "first_writer": first_writer,
    "append_set": append_set,
    "vote_set": vote_set,
    "g_counter": g_counter,
}


def get(name: str) -> Merge:
    try:
        return MERGES[name]
    except KeyError:
        raise KeyError(f"unknown merge {name!r}; known: {sorted(MERGES)}") from None
