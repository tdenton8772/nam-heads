"""Tests for CouchbaseKV-backed context store."""
import json
import pytest
import sys
import os


from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore


@pytest.fixture
def kv():
    return FakeKV()


@pytest.fixture
def store(kv):
    return CouchbaseKVContextStore(kv)


def test_put_and_get(store):
    store.put("new york", "geographic", "new_york", confidence=1.0)
    assert store.get("new york", "geographic") == "new_york"


def test_get_miss(store):
    assert store.get("nonexistent", "temporal") is None


def test_deferred_writes_flushed_on_batch(store, kv):
    store.put("medieval", "temporal", "medieval_period", confidence=0.9)
    assert "ctx::medieval::temporal" not in kv._data

    results = store.get_batch([("medieval", "temporal")])
    assert results[("medieval", "temporal")] == "medieval_period"
    assert "ctx::medieval::temporal" in kv._data


def test_batch_lookup_uses_cache(store, kv):
    doc = json.dumps({"canonical_value": "usa", "confidence": 1.0}).encode()
    kv.set("ctx::america::geographic", doc)

    store.get_batch([("america", "geographic")])
    del kv._data["ctx::america::geographic"]
    results = store.get_batch([("america", "geographic")])
    assert results[("america", "geographic")] == "usa"


def test_batch_mixed_hits_and_misses(store, kv):
    doc = json.dumps({"canonical_value": "france", "confidence": 1.0}).encode()
    kv.set("ctx::paris::geographic", doc)

    results = store.get_batch([("paris", "geographic"), ("unknown", "temporal")])
    assert results[("paris", "geographic")] == "france"
    assert results[("unknown", "temporal")] is None


def test_has(store, kv):
    doc = json.dumps({"canonical_value": "uk", "confidence": 1.0}).encode()
    kv.set("ctx::london::geographic", doc)
    assert store.has("london", "geographic") is True
    assert store.has("nowhere", "geographic") is False


def test_count_returns_negative_one(store):
    assert store.count() == -1


def test_key_format(store, kv):
    """Verify ctx::<phrase>::<dimension> key structure."""
    store.put("renaissance", "temporal", "renaissance_period")
    # Trigger flush
    store.get_batch([("renaissance", "temporal")])
    assert "ctx::renaissance::temporal" in kv._data
    assert "ctx::temporal::renaissance" not in kv._data


def test_seed_batch_writes_missing(store, kv):
    doc = json.dumps({"canonical_value": "usa", "confidence": 1.0}).encode()
    kv.set("ctx::america::geographic", doc)

    entries = [
        ("america", "geographic", "usa", 1.0),           # already exists
        ("medieval", "temporal", "medieval_period", 1.0), # missing
        ("paris", "geographic", "france", 1.0),           # missing
    ]
    written = store.seed_batch(entries)
    assert written == 2
    assert "ctx::medieval::temporal" in kv._data
    assert "ctx::paris::geographic" in kv._data


def test_seed_batch_empty(store):
    assert store.seed_batch([]) == 0


# ---- LMDB drain tests ----

def test_drain_pending_lmdb_writes_empty(store):
    """Drain returns empty list when no writes pending."""
    assert store.drain_pending_lmdb_writes() == []


def test_drain_pending_lmdb_writes_after_put(store):
    """put() accumulates writes for LMDB relay."""
    store.put("medieval", "temporal", "medieval_period", confidence=0.9)
    store.put("paris", "geographic", "france", confidence=1.0)

    writes = store.drain_pending_lmdb_writes()
    assert len(writes) == 2
    keys = [k for k, v in writes]
    assert "ctx::medieval::temporal" in keys
    assert "ctx::paris::geographic" in keys

    # Each write value should be valid JSON with canonical_value
    for key, val in writes:
        doc = json.loads(val.decode("utf-8"))
        assert "canonical_value" in doc
        assert "confidence" in doc


def test_drain_clears_pending(store):
    """drain_pending_lmdb_writes clears the buffer."""
    store.put("x", "temporal", "y")
    first = store.drain_pending_lmdb_writes()
    assert len(first) == 1

    second = store.drain_pending_lmdb_writes()
    assert second == []


def test_drain_independent_of_kv_flush(store, kv):
    """LMDB drain is independent of KV deferred write flush."""
    store.put("test", "temporal", "test_val")
    # LMDB writes accumulate regardless of KV flush state
    writes = store.drain_pending_lmdb_writes()
    assert len(writes) == 1

    # KV writes should still be pending (not flushed yet)
    assert "ctx::test::temporal" not in kv._data

    # Now trigger KV flush via get_batch
    store.get_batch([("test", "temporal")])
    assert "ctx::test::temporal" in kv._data


def test_lmdb_cache_constructor(kv):
    """CouchbaseKVContextStore accepts lmdb_cache parameter."""
    store = CouchbaseKVContextStore(kv, lmdb_cache=None)
    assert store._lmdb is None

    class FakeLmdb:
        available = True
        _readonly = True
        def batch_get(self, keys):
            return {}
        def batch_put(self, items):
            return 0

    fake_lmdb = FakeLmdb()
    store_with_lmdb = CouchbaseKVContextStore(kv, lmdb_cache=fake_lmdb)
    assert store_with_lmdb._lmdb is fake_lmdb


# ---- kv_on_cache_miss tests ----

def test_kv_on_miss_default_true(kv):
    """Default kv_on_cache_miss is True."""
    store = CouchbaseKVContextStore(kv)
    assert store._kv_on_miss is True


def test_kv_on_miss_false_constructor(kv):
    """Constructor accepts kv_on_cache_miss=False."""
    store = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    assert store._kv_on_miss is False


def test_get_skips_kv_when_kv_on_miss_false(kv):
    """get() returns None for uncached keys when kv_on_cache_miss=False."""
    doc = json.dumps({"canonical_value": "usa", "confidence": 1.0}).encode()
    kv.set("ctx::america::geographic", doc)

    store = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    # Key exists in KV but store should not look it up
    assert store.get("america", "geographic") is None


def test_get_returns_cached_when_kv_on_miss_false(kv):
    """get() still returns L1-cached values when kv_on_cache_miss=False."""
    store = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    # Populate L1 cache via put()
    store.put("new york", "geographic", "new_york", confidence=1.0)
    assert store.get("new york", "geographic") == "new_york"


def test_get_batch_skips_kv_when_kv_on_miss_false(kv):
    """get_batch() skips L3 KV for uncached keys when kv_on_cache_miss=False."""
    doc = json.dumps({"canonical_value": "france", "confidence": 1.0}).encode()
    kv.set("ctx::paris::geographic", doc)

    store = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    results = store.get_batch([("paris", "geographic")])
    # Key exists in KV but should not be fetched
    assert results[("paris", "geographic")] is None


def test_get_batch_returns_cached_when_kv_on_miss_false(kv):
    """get_batch() still returns L1-cached values when kv_on_cache_miss=False."""
    store = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    store.put("medieval", "temporal", "medieval_period", confidence=0.9)
    # Trigger flush + read
    results = store.get_batch([("medieval", "temporal")])
    assert results[("medieval", "temporal")] == "medieval_period"
