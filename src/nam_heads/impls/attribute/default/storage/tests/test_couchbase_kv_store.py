"""Tests for CouchbaseKV-backed attribute category store."""
import json
import pytest
import sys
import os


from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.attribute.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


@pytest.fixture
def kv():
    return FakeKV()


@pytest.fixture
def store(kv):
    return CouchbaseKVCategoryStore(kv)


def test_put_and_get_category(store):
    store.put_category("bright", "quality", confidence=0.9)
    assert store.get_category("bright") == "quality"


def test_get_category_miss(store):
    assert store.get_category("nonexistent") is None


def test_get_category_weighted_global(store, kv):
    """Global key lookup (no ontology scope)."""
    doc = json.dumps({"category": "size", "confidence": 0.85}).encode()
    kv.set("acat::large", doc)
    result = store.get_category_weighted("large")
    assert result == ("size", 0.85)


def test_get_category_weighted_ontology_scoped(store, kv):
    """Ontology-scoped key takes priority over global."""
    global_doc = json.dumps({"category": "quality", "confidence": 0.7}).encode()
    scoped_doc = json.dumps({"category": "appearance", "confidence": 0.95}).encode()
    kv.set("acat::bright", global_doc)
    kv.set("acat::bright::person", scoped_doc)
    result = store.get_category_weighted("bright", ontology_type="person")
    assert result == ("appearance", 0.95)


def test_get_category_weighted_fallback_to_global(store, kv):
    """Ontology key missing, falls back to global."""
    global_doc = json.dumps({"category": "quality", "confidence": 0.7}).encode()
    kv.set("acat::bright", global_doc)
    result = store.get_category_weighted("bright", ontology_type="location")
    assert result == ("quality", 0.7)


def test_get_category_weighted_miss(store):
    """Neither scoped nor global exists."""
    assert store.get_category_weighted("unknown", ontology_type="person") is None


def test_deferred_writes_flushed_on_batch(store, kv):
    """put_category_weighted defers write; batch lookup flushes it."""
    store.put_category_weighted("fast", "speed", confidence=0.9, ontology_type=None)
    # Not yet in KV (deferred)
    assert "acat::fast" not in kv._data

    # Batch lookup triggers flush
    results = store.get_categories_weighted_batch([("fast", None)])
    assert ("fast", None) in results
    assert results[("fast", None)] == ("speed", 0.9)
    # Now should be in KV
    assert "acat::fast" in kv._data


def test_batch_lookup_uses_cache(store, kv):
    """Second batch call doesn't re-fetch cached keys."""
    doc = json.dumps({"category": "size", "confidence": 0.8}).encode()
    kv.set("acat::big", doc)

    # First batch populates cache
    store.get_categories_weighted_batch([("big", None)])

    # Remove from KV -- cache should still serve
    del kv._data["acat::big"]
    results = store.get_categories_weighted_batch([("big", None)])
    assert results[("big", None)] == ("size", 0.8)


def test_batch_lookup_mixed_hits_and_misses(store, kv):
    """Some cached, some not."""
    doc = json.dumps({"category": "size", "confidence": 0.8}).encode()
    kv.set("acat::big", doc)

    results = store.get_categories_weighted_batch([("big", None), ("unknown", None)])
    assert results[("big", None)] == ("size", 0.8)
    assert results[("unknown", None)] is None


def test_has_lemma(store, kv):
    doc = json.dumps({"category": "speed", "confidence": 1.0}).encode()
    kv.set("acat::fast", doc)
    assert store.has_lemma("fast") is True
    assert store.has_lemma("slow") is False


def test_count_returns_negative_one(store):
    """KV store can't scan, returns -1."""
    assert store.count() == -1


def test_seed_batch_writes_missing(store, kv):
    """seed_batch writes only entries not already in KV."""
    # Pre-populate one entry
    doc = json.dumps({"category": "size", "confidence": 1.0}).encode()
    kv.set("acat::large", doc)

    entries = [
        ("large", "size", 1.0, None),        # already exists
        ("bright", "quality", 0.9, None),     # missing
        ("bright", "appearance", 0.8, "person"),  # missing (scoped)
    ]
    written = store.seed_batch(entries)
    assert written == 2
    assert "acat::bright" in kv._data
    assert "acat::bright::person" in kv._data


def test_seed_batch_empty(store):
    assert store.seed_batch([]) == 0


def test_seed_batch_populates_cache(store, kv):
    """Seeded entries should be in cache for subsequent reads."""
    store.seed_batch([("fast", "speed", 0.9, None)])
    # Read should come from cache, not KV
    del kv._data["acat::fast"]
    result = store.get_category_weighted("fast")
    assert result == ("speed", 0.9)


# ---------------------------------------------------------------------------
# LMDB integration tests
# ---------------------------------------------------------------------------
lmdb_mod = pytest.importorskip("lmdb")


@pytest.fixture
def lmdb_cache(tmp_path):
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    c = LmdbAttributeCache(str(tmp_path / "attr-lmdb"), map_size=1 * 1024 * 1024, readonly=False)
    yield c
    c.close()


@pytest.fixture
def store_with_lmdb(kv, lmdb_cache):
    return CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb_cache)


def test_batch_lookup_uses_lmdb_l2(store_with_lmdb, kv, lmdb_cache):
    """LMDB L2 serves hits without going to KV."""
    payload = json.dumps({"category": "size", "confidence": 0.8}).encode()
    lmdb_cache.put("acat::large", payload)

    # KV does NOT have this key
    results = store_with_lmdb.get_categories_weighted_batch([("large", None)])
    assert results[("large", None)] == ("size", 0.8)


def test_batch_lookup_kv_backfills_lmdb(store_with_lmdb, kv, lmdb_cache):
    """KV results get backfilled into LMDB for future L2 hits."""
    payload = json.dumps({"category": "speed", "confidence": 0.95}).encode()
    kv.set("acat::fast", payload)

    # First lookup: miss in LMDB, hit in KV -> should backfill LMDB
    store_with_lmdb.get_categories_weighted_batch([("fast", None)])

    # Verify LMDB was backfilled
    val = lmdb_cache.get("acat::fast")
    assert val == payload


def test_batch_lookup_kv_miss_backfills_lmdb_sentinel(store_with_lmdb, kv, lmdb_cache):
    """KV misses should be backfilled as MISS_SENTINEL in LMDB."""
    from src.storage.lmdb_node_cache import _MISS_SENTINEL

    # Neither LMDB nor KV has this key
    results = store_with_lmdb.get_categories_weighted_batch([("nonexistent", None)])
    assert results[("nonexistent", None)] is None

    # LMDB should have sentinel
    val = lmdb_cache.get("acat::nonexistent")
    assert val == _MISS_SENTINEL


def test_drain_pending_lmdb_writes(store_with_lmdb):
    """put_category_weighted accumulates LMDB relay writes."""
    store_with_lmdb.put_category_weighted("hot", "temperature", confidence=0.9)
    store_with_lmdb.put_category_weighted("cold", "temperature", confidence=0.85)

    writes = store_with_lmdb.drain_pending_lmdb_writes()
    assert len(writes) == 2
    keys = [k for k, v in writes]
    assert "acat::hot" in keys
    assert "acat::cold" in keys

    # Second drain should be empty
    assert store_with_lmdb.drain_pending_lmdb_writes() == []


def test_drain_pending_lmdb_writes_no_lmdb(store):
    """drain_pending_lmdb_writes works even without LMDB cache."""
    store.put_category_weighted("hot", "temperature", confidence=0.9)
    writes = store.drain_pending_lmdb_writes()
    assert len(writes) == 1


def test_lmdb_l2_then_l1_cache(store_with_lmdb, kv, lmdb_cache):
    """After LMDB L2 hit, value is promoted to L1 in-memory cache."""
    payload = json.dumps({"category": "color", "confidence": 0.7}).encode()
    lmdb_cache.put("acat::red", payload)

    # First batch: L2 hit
    store_with_lmdb.get_categories_weighted_batch([("red", None)])

    # Close LMDB to prove L1 cache is serving
    lmdb_cache.close()
    results = store_with_lmdb.get_categories_weighted_batch([("red", None)])
    assert results[("red", None)] == ("color", 0.7)
