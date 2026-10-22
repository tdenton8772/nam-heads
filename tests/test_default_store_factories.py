"""The axis heads' store factories on the deployed branch.

Ingest attribute/affordance/context containers and nam-query build their
store with backend ``couchbase_kv`` (configmap NAM_*_STORE_BACKEND), on the
shared "ingest-session" KV client, with the node LMDB in front on ingest
(NAM_<HEAD>_LMDB_PATH, NAM_<HEAD>_LMDB_READONLY=1, ingest-statefulset.yaml).
KVRegistry.get_or_create is faked: no data service is contacted."""
import importlib
import logging

import pytest

pytest.importorskip("lmdb")

from src.storage.lmdb_node_cache import LmdbAttributeCache

MB = 1024 * 1024

HEADS = {
    # head: (factory module, create function, KV store class, memory store class)
    "attribute": ("nam_heads.impls.attribute.default.storage.factory", "create_category_store",
                  "CouchbaseKVCategoryStore", "MemoryCategoryStore"),
    "affordance": ("nam_heads.impls.affordance.default.storage.factory", "create_category_store",
                   "CouchbaseKVCategoryStore", "MemoryCategoryStore"),
    "context": ("nam_heads.impls.context.default.storage.factory", "create_context_store",
                "CouchbaseKVContextStore", "MemoryContextStore"),
}


@pytest.fixture
def kv_calls(monkeypatch):
    from src.storage.backends import kv_registry
    calls = []
    client = type("FakeKV", (), {})()

    def get_or_create(cls, name, **kw):
        calls.append({"name": name, **kw})
        if getattr(get_or_create, "fail", None):
            raise get_or_create.fail
        return client

    monkeypatch.setattr(kv_registry.KVRegistry, "get_or_create", classmethod(get_or_create))
    for var in ("NAM_SESSION_BUCKET", "NAM_SESSION_VBUCKETS", "NAM_KV_POOL_INGEST_SESSION_WORKERS",
                "NAM_SEED_DEFER_ON_ERROR"):
        monkeypatch.delenv(var, raising=False)
    for head in HEADS:
        for suffix in ("LMDB_PATH", "LMDB_READONLY", "LMDB_MAP_SIZE_MB"):
            monkeypatch.delenv(f"NAM_{head.upper()}_{suffix}", raising=False)
    return calls, client, get_or_create


def _factory(head):
    mod_name, fn, kv_cls, mem_cls = HEADS[head]
    mod = importlib.import_module(mod_name)
    store_mod = importlib.import_module(mod_name.rsplit(".", 1)[0] + ".couchbase_kv_store")
    mem_mod = importlib.import_module(mod_name.rsplit(".", 1)[0] + ".memory_store")
    return mod, getattr(mod, fn), getattr(store_mod, kv_cls), getattr(mem_mod, mem_cls)


def _writer(path):
    w = LmdbAttributeCache(str(path), map_size=4 * MB)
    w.batch_put([("k::seed", b"v")])        # creates data.mdb for the readonly opener
    return w


@pytest.mark.parametrize("head", sorted(HEADS))
def test_ingest_branch_session_kv_with_the_node_lmdb_readonly(head, tmp_path, monkeypatch, kv_calls):
    calls, client, _ = kv_calls
    _mod, create, kv_cls, _ = _factory(head)
    w = _writer(tmp_path / head)
    monkeypatch.setenv(f"NAM_{head.upper()}_LMDB_PATH", str(tmp_path / head))
    monkeypatch.setenv(f"NAM_{head.upper()}_LMDB_READONLY", "1")
    monkeypatch.setenv("NAM_SESSION_VBUCKETS", "64")
    store = create("couchbase_kv", fallback=None, seed=False, kv_on_cache_miss=False)
    try:
        assert isinstance(store, kv_cls)
        assert store._kv is client and store._kv_on_miss is False
        assert isinstance(store._lmdb, LmdbAttributeCache)
        assert store._lmdb.available and store._lmdb._readonly
        assert store._lmdb.get("k::seed") == b"v"
        (call,) = calls
        assert call["name"] == "ingest-session" and call["pool_workers"] == 4
        assert call["bucket"] == "session" and call["default_vbuckets"] == 64
    finally:
        if store._lmdb is not None:
            store._lmdb.close()
        w.close()


@pytest.mark.parametrize("head", sorted(HEADS))
def test_query_branch_session_kv_without_lmdb(head, kv_calls):
    """nam-query sets no *_LMDB_PATH: the store reads the session KV directly."""
    calls, client, _ = kv_calls
    _mod, create, kv_cls, _ = _factory(head)
    store = create("couchbase", fallback=None, seed=False, kv_on_cache_miss=True)
    assert isinstance(store, kv_cls) and store._kv is client
    assert store._lmdb is None and store._kv_on_miss is True


@pytest.mark.parametrize("head", sorted(HEADS))
def test_an_unreachable_session_client_falls_back_to_memory_and_says_so(head, kv_calls, caplog):
    calls, _client, goc = kv_calls
    mod, create, _kv_cls, mem_cls = _factory(head)
    goc.fail = ConnectionError("data service down")
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        store = create("couchbase_kv", fallback="memory", seed=False)
    assert isinstance(store, mem_cls)
    assert any("couchbase_kv" in r.getMessage() and "data service down" in r.getMessage()
               for r in caplog.records), [r.getMessage() for r in caplog.records]
    with pytest.raises(ConnectionError):
        create("couchbase_kv", fallback=None, seed=False)


@pytest.mark.parametrize("head", sorted(HEADS))
def test_unknown_backend(head, kv_calls):
    _mod, create, _kv_cls, mem_cls = _factory(head)
    with pytest.raises(ValueError, match="Unknown"):
        create("sqlite", fallback=None, seed=False)
    assert isinstance(create("sqlite", fallback="memory", seed=False), mem_cls)


@pytest.mark.parametrize("head", sorted(HEADS))
def test_seeding_a_memory_store_is_idempotent(head, kv_calls):
    _mod, create, _kv_cls, _mem = _factory(head)
    store = create("memory", seed=True)
    n = store.count()
    assert n > 0
    mod = importlib.import_module(HEADS[head][0])
    mod._seed_store(store)
    assert store.count() == n


def test_context_lmdb_that_cannot_open_is_dropped_not_fatal(tmp_path, monkeypatch, kv_calls, caplog):
    """Context differs from attribute/affordance: an LMDB it cannot open is
    left out (warning) and the store reads the session KV alone."""
    from src.storage import lmdb_node_cache as lmdb_cache
    monkeypatch.setattr(lmdb_cache, "_READONLY_WAIT_TIMEOUT", 0)
    monkeypatch.setenv("NAM_CONTEXT_LMDB_PATH", str(tmp_path / "never-created"))
    monkeypatch.setenv("NAM_CONTEXT_LMDB_READONLY", "1")
    mod, create, kv_cls, _ = _factory("context")
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        store = create("couchbase_kv", fallback=None, seed=False)
    assert isinstance(store, kv_cls) and store._lmdb is None
    assert any("not available" in r.getMessage() for r in caplog.records)
