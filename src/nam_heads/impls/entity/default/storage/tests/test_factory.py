"""create_entity_store: the branch each deployment takes.

- ingest entity container: backend ``lmdb`` (ingest-statefulset.yaml sets
  NAM_ENTITY_STORE_BACKEND=lmdb and NAM_ENTITY_LMDB_PATH) -> a readonly
  LmdbEntityCache behind LmdbEntityStore;
- nam-query: backend ``couchbase_kv`` (configmap) -> CouchbaseKVEntityStore on
  the shared "ingest-session" KV client, KV read on miss;
- tests/dev: ``memory``.
The session KV is never contacted: KVRegistry.get_or_create is faked."""
import logging

import pytest

pytest.importorskip("lmdb")

from nam_heads.impls.entity.default.storage import factory as F
from nam_heads.impls.entity.default.storage.lmdb_cache import LmdbEntityCache
from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
from nam_heads.impls.entity.default.storage.memory_store import MemoryEntityStore

MB = 1024 * 1024


@pytest.fixture
def fake_kv(monkeypatch):
    """Record KVRegistry.get_or_create calls; hand back a sentinel client."""
    from src.storage.backends import kv_registry
    calls = []
    client = object()

    def get_or_create(name, *, pool_workers=8, **kwargs):
        calls.append({"name": name, "pool_workers": pool_workers, **kwargs})
        return client

    monkeypatch.setattr(kv_registry.KVRegistry, "get_or_create", classmethod(
        lambda cls, name, **kw: get_or_create(name, **kw)))
    for var in ("NAM_SESSION_BUCKET", "NAM_SESSION_VBUCKETS", "NAM_KV_POOL_INGEST_SESSION_WORKERS",
                "NAM_ENTITY_LMDB_PATH", "NAM_ENTITY_LMDB_READONLY", "NAM_ENTITY_LMDB_MAP_SIZE_MB",
                "NAM_ENTITY_ASYNC_WRITES"):
        monkeypatch.delenv(var, raising=False)
    return calls, client


def _seed_lmdb(path):
    w = LmdbEntityCache(str(path), map_size=4 * MB)
    w.put("surf::PERSON::ada lovelace", b"e-ada")
    w.close()


def test_memory_backend(fake_kv):
    assert isinstance(F.create_entity_store("memory", None), MemoryEntityStore)
    assert fake_kv[0] == []


def test_lmdb_backend_reads_the_node_cache_readonly(tmp_path, monkeypatch, fake_kv):
    _seed_lmdb(tmp_path / "entity")
    monkeypatch.setenv("NAM_ENTITY_LMDB_PATH", str(tmp_path / "entity"))
    store = F.create_entity_store("LMDB", "memory")     # backend name is case-insensitive
    assert isinstance(store, LmdbEntityStore)
    assert store.resolve_surface_form("ada lovelace", "PERSON") == "e-ada"
    cache = store._lmdb
    assert cache.available and cache._readonly
    assert cache.put("surf::x", b"y") is False
    assert fake_kv[0] == []                                    # no session KV on ingest


def test_lmdb_backend_without_a_path_raises_when_there_is_no_fallback(fake_kv):
    with pytest.raises(ValueError, match="NAM_ENTITY_LMDB_PATH"):
        F.create_entity_store("lmdb", None)


def test_a_fallback_to_memory_is_logged(fake_kv, caplog):
    """The entity head keeps running on a memory store (no shared aliases or
    redirects), so the warning is the only trace of the failed primary."""
    with caplog.at_level(logging.WARNING, logger=F.__name__):
        store = F.create_entity_store("lmdb", "memory")
    assert isinstance(store, MemoryEntityStore)
    msgs = [r.getMessage() for r in caplog.records if r.name == F.__name__]
    assert any("lmdb" in m and "NAM_ENTITY_LMDB_PATH" in m and "memory" in m for m in msgs), msgs


def test_unknown_backend(fake_kv, caplog):
    with pytest.raises(ValueError, match="Unknown db_backend"):
        F.create_entity_store("cassandra", None)
    with caplog.at_level(logging.WARNING, logger=F.__name__):
        assert isinstance(F.create_entity_store("cassandra", "memory"), MemoryEntityStore)
    assert caplog.records


def test_a_fallback_that_is_not_memory_reports_both(fake_kv):
    with pytest.raises(RuntimeError, match="Primary backend 'cassandra' failed.*fallback 'sqlite'"):
        F.create_entity_store("cassandra", "sqlite")


def test_couchbase_kv_uses_the_shared_session_client(fake_kv, monkeypatch):
    calls, client = fake_kv
    monkeypatch.setenv("NAM_SESSION_VBUCKETS", "64")
    monkeypatch.setenv("NAM_KV_POOL_INGEST_SESSION_WORKERS", "7")
    store = F.create_entity_store("couchbase_kv", "memory", kv_on_cache_miss=True)
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    assert isinstance(store, CouchbaseKVEntityStore)
    assert store._kv is client and store._lmdb is None and store._kv_on_miss is True
    (call,) = calls
    assert call["name"] == "ingest-session" and call["pool_workers"] == 7
    assert call["bucket"] == "session" and call["default_vbuckets"] == 64
    assert call["use_bucket_auth"] is True and call["password"] == ""


def test_couchbase_kv_puts_the_node_lmdb_in_front_when_configured(tmp_path, fake_kv, monkeypatch):
    _seed_lmdb(tmp_path / "entity")
    monkeypatch.setenv("NAM_ENTITY_LMDB_PATH", str(tmp_path / "entity"))
    monkeypatch.setenv("NAM_ENTITY_LMDB_READONLY", "true")
    store = F.create_entity_store("couchbase", None, kv_on_cache_miss=False)
    assert store._kv_on_miss is False
    assert isinstance(store._lmdb, LmdbEntityCache) and store._lmdb._readonly
    assert store._lmdb.get("surf::PERSON::ada lovelace") == b"e-ada"
