"""LmdbEntityCache: the node LMDB the ingest entity head reads surf::/cand::
keys from (entity container: NAM_ENTITY_STORE_BACKEND=lmdb, opened readonly
by storage/factory.py; the entity-lmdb-sync role is the writer)."""
import threading
import time

import pytest

pytest.importorskip("lmdb")

from nam_heads.impls.entity.default.storage import lmdb_cache as mod
from nam_heads.impls.entity.default.storage.lmdb_cache import LmdbEntityCache, _MISS_SENTINEL

MB = 1024 * 1024


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "entity-lmdb")


@pytest.fixture
def writer(path):
    c = LmdbEntityCache(path, map_size=4 * MB)
    yield c
    c.close()


def test_writer_put_get_and_miss(writer):
    assert writer.available
    assert writer.put("surf::PERSON::alice", b"ent-1") is True
    assert writer.get("surf::PERSON::alice") == b"ent-1"
    assert writer.get("surf::PERSON::bob") is None


def test_the_negative_cache_sentinel_comes_back_as_stored(writer):
    writer.put("cand::bob", _MISS_SENTINEL)
    assert writer.get("cand::bob") == _MISS_SENTINEL


def test_batch_get_returns_only_present_keys(writer):
    assert writer.batch_put([("surf::a", b"1"), ("surf::b", b"2")]) == 2
    assert writer.batch_get(["surf::a", "surf::b", "surf::c"]) == {"surf::a": b"1", "surf::b": b"2"}
    assert writer.batch_get([]) == {}
    assert writer.batch_put([]) == 0


def test_a_readonly_reader_sees_the_writer_and_cannot_write(path, writer):
    writer.batch_put([("surf::x", b"e1")])
    reader = LmdbEntityCache(path, map_size=4 * MB, readonly=True)
    try:
        assert reader.available
        assert reader.get("surf::x") == b"e1"
        assert reader.put("surf::y", b"e2") is False
        assert reader.batch_put([("surf::y", b"e2")]) == 0
        writer.put("surf::z", b"e3")              # committed after the reader opened
        assert reader.batch_get(["surf::z"]) == {"surf::z": b"e3"}
        assert writer.get("surf::y") is None
    finally:
        reader.close()


def test_a_readonly_opener_waits_for_the_writer_to_create_the_file(path, monkeypatch):
    """The entity container can start before entity-lmdb-sync has created
    data.mdb: the reader waits (backoff) instead of failing at once."""
    monkeypatch.setattr(mod, "_READONLY_WAIT_TIMEOUT", 20)

    def create_later():
        time.sleep(0.6)
        w = LmdbEntityCache(path, map_size=4 * MB)
        w.put("surf::late", b"e9")
        w.close()

    t = threading.Thread(target=create_later)
    t.start()
    t0 = time.monotonic()
    reader = LmdbEntityCache(path, map_size=4 * MB, readonly=True)
    waited = time.monotonic() - t0
    t.join()
    try:
        assert reader.available
        assert waited >= 0.5
        assert reader.get("surf::late") == b"e9"
    finally:
        reader.close()


def test_a_readonly_opener_gives_up_unavailable_and_answers_misses(path, monkeypatch):
    monkeypatch.setattr(mod, "_READONLY_WAIT_TIMEOUT", 1)
    t0 = time.monotonic()
    reader = LmdbEntityCache(path, map_size=4 * MB, readonly=True)
    assert time.monotonic() - t0 < 15
    assert reader.available is False
    assert reader.get("surf::x") is None
    assert reader.batch_get(["surf::x"]) == {}
    assert reader.put("surf::x", b"v") is False
    assert reader.batch_put([("surf::x", b"v")]) == 0
    assert reader.stat() == {}
    reader.close()                                 # no env: close is a no-op


def test_close_makes_it_unavailable_and_is_idempotent(writer):
    writer.put("surf::a", b"1")
    assert writer.stat()["entries"] == 1
    writer.close()
    assert writer.available is False
    assert writer.get("surf::a") is None
    assert writer.stat() == {}
    writer.close()


def test_a_key_lmdb_cannot_store_does_not_sink_the_batch(writer):
    written = writer.batch_put([("surf::a", b"1"), ("surf::" + "n" * 600, b"2"), ("surf::b", b"3")])
    assert written == 2
    assert writer.batch_get(["surf::a", "surf::b"]) == {"surf::a": b"1", "surf::b": b"3"}


def test_an_aborted_transaction_counts_zero(path):
    c = LmdbEntityCache(path, map_size=1 * MB)
    try:
        assert c.batch_put([("surf::small", b"v"), ("surf::big", b"z" * (4 * MB))]) == 0
        assert c.get("surf::small") is None
        assert c.put("surf::big", b"z" * (4 * MB)) is False
        assert c.put("surf::after", b"v") is True
    finally:
        c.close()


def test_an_unopenable_path_is_unavailable_not_raised(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    c = LmdbEntityCache(str(blocker / "lmdb"), map_size=MB)
    assert c.available is False
    assert c.get("surf::x") is None
