"""Reusable in-memory KV mock matching CouchbaseKV public interface."""
from typing import Dict, List, Optional, Tuple

ERR_SUCCESS = 0x00
ERR_KEY_ENOENT = 0x01
ERR_KEY_EEXISTS = 0x02


class FakeKV:
    """In-memory KV mock for testing KV-backed stores without a real data service.

    Mirrors CouchbaseKV: ``get`` returns (value, cas); ``set`` returns a
    status (0 on success) and honours a non-zero ``cas`` (KEY_EEXISTS on a
    mismatch, KEY_ENOENT for a missing key); ``add`` creates only if absent.
    """

    def __init__(self):
        self._data: Dict[str, bytes] = {}
        self._cas: Dict[str, int] = {}
        self._cas_counter = 0

    def _bump(self, key: str) -> int:
        self._cas_counter += 1
        self._cas[key] = self._cas_counter
        return self._cas_counter

    def get(self, key: str) -> Tuple[Optional[bytes], Optional[int]]:
        if key in self._data:
            return (self._data[key], self._cas[key])
        return (None, None)

    def set(self, key: str, value: bytes, cas=0, expiration=0) -> int:
        if cas:
            if key not in self._data:
                return ERR_KEY_ENOENT
            if self._cas.get(key) != cas:
                return ERR_KEY_EEXISTS
        self._data[key] = value
        self._bump(key)
        return ERR_SUCCESS

    def add(self, key: str, value: bytes, expiration=0) -> bool:
        if key in self._data:
            return False
        self._data[key] = value
        self._bump(key)
        return True

    def pipeline_get(self, keys: List[str]) -> Dict[str, Tuple[bytes, int]]:
        return {k: (self._data[k], self._cas[k]) for k in keys if k in self._data}

    def pipeline_add(self, items: List[Tuple[str, bytes]]) -> List[Tuple[str, int]]:
        """Mirrors CouchbaseKV.pipeline_add: (key, status); KEY_EEXISTS if present."""
        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        return [(k, ERR_SUCCESS if self.add(k, v) else ERR_KEY_EEXISTS) for k, v in items]

    def pipeline_set_cas(self, items: List[Tuple[str, bytes, int]]) -> List[Tuple[str, int]]:
        """Mirrors CouchbaseKV.pipeline_set_cas: (key, status) per CAS SET."""
        self.batch_calls = getattr(self, "batch_calls", 0) + 1
        return [(k, self.set(k, v, cas=c)) for k, v, c in items]

    def pipeline_set(self, items: List[Tuple[str, bytes]]) -> List[Tuple[str, int]]:
        results = []
        for k, v in items:
            self._data[k] = v
            results.append((k, self._bump(k)))
        return results
