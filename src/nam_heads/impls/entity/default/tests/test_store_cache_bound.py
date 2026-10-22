"""CouchbaseKVEntityStore read caches are bounded when NAM_ENTITY_STORE_CACHE_MAX is set.

The query pod reads through to KV (NAM_ENTITY_KV_ON_MISS=1), so every distinct
query adds surf::/cand:: keys; unbounded, those dicts grow for the life of the pod.
"""

from ..storage.couchbase_kv_store import CouchbaseKVEntityStore


class FakeKV:
    def __init__(self, data=None):
        self.data = data or {}
        self.gets = 0

    def get(self, key):
        self.gets += 1
        return self.data.get(key), None

    def pipeline_get(self, keys):
        self.gets += len(keys)
        return {k: (self.data.get(k), None) for k in keys}


def _store(monkeypatch, cap, data=None):
    monkeypatch.setenv("NAM_ENTITY_STORE_CACHE_MAX", str(cap))
    monkeypatch.setenv("NAM_ENTITY_ASYNC_WRITES", "0")
    return CouchbaseKVEntityStore(FakeKV(data), kv_on_cache_miss=True)


def test_unbounded_by_default(monkeypatch):
    monkeypatch.delenv("NAM_ENTITY_STORE_CACHE_MAX", raising=False)
    monkeypatch.setenv("NAM_ENTITY_ASYNC_WRITES", "0")
    st = CouchbaseKVEntityStore(FakeKV(), kv_on_cache_miss=True)
    for i in range(50):
        st.resolve_surface_form(f"s{i}")
    assert len(st._surf_cache) == 50


def test_surf_cache_bounded(monkeypatch):
    st = _store(monkeypatch, 10)
    for i in range(100):
        st.resolve_surface_form(f"s{i}")
        assert len(st._surf_cache) <= 11


def test_cand_cache_bounded(monkeypatch):
    st = _store(monkeypatch, 10)
    for i in range(100):
        st.lookup_candidate_keys_by_key([f"k{i}"])
        assert len(st._cand_cache) <= 11


def test_cleared_cache_rereads_kv(monkeypatch):
    st = _store(monkeypatch, 2, {"surf::a": b"eid_a"})
    assert st.resolve_surface_form("a") == "eid_a"
    for i in range(5):
        st.resolve_surface_form(f"x{i}")
    assert st.resolve_surface_form("a") == "eid_a"
    assert st._kv.gets >= 7


def test_kv_on_miss_reads_session_kv(monkeypatch):
    """The query-pod setting: a cold cache still finds the stored key."""
    st = _store(monkeypatch, 0, {"surf::compound_noun::polydor_records": b"63aa55edcc8d"})
    assert st.resolve_surface_form("polydor_records", "compound_noun") == "63aa55edcc8d"
    assert st.resolve_surface_form("polydor_records") is None
