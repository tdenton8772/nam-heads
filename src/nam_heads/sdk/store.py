"""Head stores: state a head keeps between records, provided by the host.

An implementation declares the stores it needs in ``IMPL_META["stores"]`` and
reads them with ``self.store(name)``. The host decides where a store lives (in
NAM: a key prefix in the session bucket, its own vbuckets, a node LMDB cache
filled by DCP) and gives the head a ``HeadStore`` handle. Without a host, the
SDK hands out a ``MemoryStore`` with the same semantics, so a stateful head is
testable on its own.

    IMPL_META = {..., "stores": [
        {"name": "lexicon",
         "scope": "slot",           # "slot": shared by every implementation of the slot
                                     # "impl": private to this implementation
         "families": [{"prefix": "",   "merge": "lww"},        # longest prefix wins
                      {"prefix": "v:", "merge": "vote_set"}],
         "miss_ttl_s": 300}]}       # a miss is cached this long (0: never)

Semantics every handle has (docs/design-head-stores.md in NAM):

- keys are relative to the store; values are bytes (``get_docs`` / ``put_doc``
  read and write JSON);
- ``get_many`` is the one read: a batch, one round trip per tier;
- ``put`` merges the value with what the key holds by its family's merge
  (``nam_heads.sdk.merge``), and the head reads its own write at once;
- a query-mode handle is read-only (``writable`` False): a write raises
  ``ReadOnlyStore`` in tests and is counted and dropped by the host.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import merge as M

STORE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
SCOPES = ("slot", "impl")


class ReadOnlyStore(RuntimeError):
    """A write to a store handle opened read-only (query mode)."""


class StoreSpecError(ValueError):
    """A store declaration in IMPL_META is malformed."""


@dataclass(frozen=True)
class Family:
    prefix: str
    merge: str


@dataclass(frozen=True)
class StoreSpec:
    name: str
    scope: str = "slot"
    families: Tuple[Family, ...] = (Family("", "lww"),)
    miss_ttl_s: float = 0.0

    def merge_for(self, key: str) -> M.Merge:
        """The merge of the longest family prefix ``key`` starts with."""
        best = None
        for f in self.families:
            if key.startswith(f.prefix) and (best is None or len(f.prefix) > len(best.prefix)):
                best = f
        return M.get(best.merge if best else "lww")


def parse_specs(raw: Any) -> List[StoreSpec]:
    """IMPL_META["stores"] -> [StoreSpec]. Raises StoreSpecError."""
    if raw is None:
        return []
    if not isinstance(raw, (list, tuple)):
        raise StoreSpecError("IMPL_META stores must be a list")
    out, names = [], set()
    for i, s in enumerate(raw):
        where = f"stores[{i}]"
        if not isinstance(s, dict):
            raise StoreSpecError(f"{where} must be an object")
        name = s.get("name")
        if not isinstance(name, str) or not STORE_NAME_RE.match(name):
            raise StoreSpecError(f"{where}: name must match {STORE_NAME_RE.pattern}")
        if name in names:
            raise StoreSpecError(f"{where}: store {name!r} declared twice")
        names.add(name)
        scope = s.get("scope", "slot")
        if scope not in SCOPES:
            raise StoreSpecError(f"{where}: scope must be one of {SCOPES}")
        fams = s.get("families", [{"prefix": "", "merge": "lww"}])
        if not isinstance(fams, (list, tuple)) or not fams:
            raise StoreSpecError(f"{where}: families must be a non-empty list")
        families, prefixes = [], set()
        for f in fams:
            if not isinstance(f, dict) or not isinstance(f.get("prefix", ""), str):
                raise StoreSpecError(f"{where}: each family is {{prefix, merge}}")
            if f.get("merge", "lww") not in M.MERGES:
                raise StoreSpecError(f"{where}: unknown merge {f.get('merge')!r}; known {sorted(M.MERGES)}")
            p = f.get("prefix", "")
            if p in prefixes:
                raise StoreSpecError(f"{where}: family prefix {p!r} declared twice")
            prefixes.add(p)
            families.append(Family(p, f.get("merge", "lww")))
        ttl = s.get("miss_ttl_s", 0)
        if not isinstance(ttl, (int, float)) or ttl < 0:
            raise StoreSpecError(f"{where}: miss_ttl_s must be >= 0")
        out.append(StoreSpec(name=name, scope=scope, families=tuple(families), miss_ttl_s=float(ttl)))
    return out


class HeadStore:
    """The interface a head uses. Hosts subclass it; ``MemoryStore`` is the SDK's."""

    spec: StoreSpec
    writable: bool = True

    def get_many(self, keys: Sequence[str]) -> Dict[str, Optional[bytes]]:
        raise NotImplementedError

    def put_many(self, items: Iterable[Tuple[str, bytes]]) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    # -- conveniences over the two primitives ---------------------------------
    def get(self, key: str) -> Optional[bytes]:
        return self.get_many([key]).get(key)

    def put(self, key: str, value: bytes) -> None:
        self.put_many([(key, value)])

    def get_docs(self, keys: Sequence[str]) -> Dict[str, Optional[dict]]:
        out = {}
        for k, v in self.get_many(keys).items():
            try:
                out[k] = json.loads(v) if v is not None else None
            except ValueError:
                out[k] = None
        return out

    def put_doc(self, key: str, doc: dict) -> None:
        self.put(key, json.dumps(doc, sort_keys=True, separators=(",", ":")).encode())


class MemoryStore(HeadStore):
    """A process-local store with the same semantics as a host's (merge on
    put, read-your-writes, read-only mode, miss TTL). The default when no host
    is registered; ``shared`` lets several handles see one backing dict, as
    two implementations of a slot share a slot-scoped store."""

    def __init__(self, spec: StoreSpec, writable: bool = True,
                 shared: Optional[Dict[str, bytes]] = None):
        self.spec = spec
        self.writable = writable
        self._data: Dict[str, bytes] = shared if shared is not None else {}
        self._lock = threading.Lock()
        self.reads = 0          # get_many calls (round trips, for tests)

    def get_many(self, keys: Sequence[str]) -> Dict[str, Optional[bytes]]:
        with self._lock:
            self.reads += 1
            return {k: self._data.get(k) for k in keys}

    def put_many(self, items: Iterable[Tuple[str, bytes]]) -> None:
        if not self.writable:
            raise ReadOnlyStore(f"store {self.spec.name!r} is read-only here (query mode)")
        with self._lock:
            for k, v in items:
                if not isinstance(v, (bytes, bytearray)):
                    raise TypeError(f"store values are bytes, got {type(v).__name__}")
                self._data[k] = self.spec.merge_for(k)(self._data.get(k), bytes(v))

    def delete(self, key: str) -> None:
        if not self.writable:
            raise ReadOnlyStore(f"store {self.spec.name!r} is read-only here (query mode)")
        with self._lock:
            self._data.pop(key, None)

    def items(self) -> Dict[str, bytes]:
        with self._lock:
            return dict(self._data)
