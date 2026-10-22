"""After a session flush, the entity stores must not answer from process
memory or send writes queued before the flush (cold reset,
src/storage/session_reset.py)."""
from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore


def test_kv_store_answers_from_its_surf_cache_until_reset():
    kv = FakeKV()
    store = CouchbaseKVEntityStore(kv)
    kv.set("surf::person::albert_einstein", b"e1")
    assert store.resolve_surface_form("albert_einstein", "person") == "e1"
    kv._data.clear()                                                       # the session flush
    assert store.resolve_surface_form("albert_einstein", "person") == "e1"  # stale
    out = store.reset_session_state()
    assert out["surf"] == 1
    assert store.resolve_surface_form("albert_einstein", "person") is None


def test_kv_store_reset_drops_unsent_writes_and_every_cache():
    store = CouchbaseKVEntityStore(FakeKV())
    store._cand_cache["cand::x"] = b"e1"
    store._redir_cache["e1"] = "e2"
    store._pending_lmdb_writes.append(("surf::x", b"e1"))
    store._write_q.put(("ent::e1", b"{}"))
    out = store.reset_session_state()
    assert out == {"surf": 0, "cand": 1, "redir": 1, "pending_relay": 1, "queued_writes": 1}
    assert store.drain_pending_lmdb_writes() == [] and store._write_q.empty()
    assert not store._cand_cache and not store._redir_cache


class _Lmdb:
    def get(self, key):
        return None

    def batch_get(self, keys):
        return {}


def test_lmdb_store_reset_drops_unpublished_writes():
    store = LmdbEntityStore(_Lmdb())
    store.create_entity("e1", ["albert_einstein"], {"entity_type": "person"})
    out = store.reset_session_state()
    assert out["pending_relay"] >= 2   # ent:: and its surf:: keys
    assert store.drain_pending_lmdb_writes() == []
