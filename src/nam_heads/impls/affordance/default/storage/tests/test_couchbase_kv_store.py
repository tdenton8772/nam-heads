"""Tests for CouchbaseKV-backed affordance verb category store."""
import json
import pytest
import sys
import os


from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


@pytest.fixture
def kv():
    return FakeKV()


@pytest.fixture
def store(kv):
    return CouchbaseKVCategoryStore(kv)


def test_put_and_get_category(store):
    store.put_category("run", "movement", confidence=0.9)
    assert store.get_category("run") == "movement"


def test_get_category_miss(store):
    assert store.get_category("nonexistent") is None


def test_get_category_weighted_global(store, kv):
    doc = json.dumps({"category": "physical", "confidence": 0.85}).encode()
    kv.set("vcat::push", doc)
    result = store.get_category_weighted("push")
    assert result == ("physical", 0.85)


def test_get_category_weighted_ontology_scoped(store, kv):
    """Ontology-scoped key takes priority over global."""
    global_doc = json.dumps({"category": "physical", "confidence": 0.7}).encode()
    scoped_doc = json.dumps({"category": "communicative", "confidence": 0.95}).encode()
    kv.set("vcat::address", global_doc)
    kv.set("vcat::address::person", scoped_doc)
    result = store.get_category_weighted("address", ontology_type="person")
    assert result == ("communicative", 0.95)


def test_get_category_weighted_fallback_to_global(store, kv):
    global_doc = json.dumps({"category": "movement", "confidence": 0.7}).encode()
    kv.set("vcat::run", global_doc)
    result = store.get_category_weighted("run", ontology_type="location")
    assert result == ("movement", 0.7)


def test_get_category_weighted_miss(store):
    assert store.get_category_weighted("unknown", ontology_type="person") is None


def test_deferred_writes_flushed_on_batch(store, kv):
    store.put_category_weighted("jump", "movement", confidence=0.9, ontology_type=None)
    assert "vcat::jump" not in kv._data

    results = store.get_categories_weighted_batch([("jump", None)])
    assert results[("jump", None)] == ("movement", 0.9)
    assert "vcat::jump" in kv._data


def test_batch_lookup_uses_cache(store, kv):
    doc = json.dumps({"category": "physical", "confidence": 0.8}).encode()
    kv.set("vcat::push", doc)

    store.get_categories_weighted_batch([("push", None)])
    del kv._data["vcat::push"]
    results = store.get_categories_weighted_batch([("push", None)])
    assert results[("push", None)] == ("physical", 0.8)


def test_batch_lookup_mixed_hits_and_misses(store, kv):
    doc = json.dumps({"category": "cognitive", "confidence": 0.8}).encode()
    kv.set("vcat::think", doc)

    results = store.get_categories_weighted_batch([("think", None), ("unknown", None)])
    assert results[("think", None)] == ("cognitive", 0.8)
    assert results[("unknown", None)] is None


def test_has_lemma(store, kv):
    doc = json.dumps({"category": "movement", "confidence": 1.0}).encode()
    kv.set("vcat::run", doc)
    assert store.has_lemma("run") is True
    assert store.has_lemma("fly") is False


def test_count_returns_negative_one(store):
    assert store.count() == -1


def test_key_prefix_is_vcat(store, kv):
    """Verify keys use vcat:: prefix (not acat::)."""
    store.put_category("walk", "movement")
    assert "vcat::walk" in kv._data
    assert "acat::walk" not in kv._data


def test_seed_batch_writes_missing(store, kv):
    doc = json.dumps({"category": "movement", "confidence": 1.0}).encode()
    kv.set("vcat::run", doc)

    entries = [
        ("run", "movement", 1.0, None),       # already exists
        ("think", "cognitive", 0.9, None),     # missing
        ("speak", "communicative", 0.8, "person"),  # missing (scoped)
    ]
    written = store.seed_batch(entries)
    assert written == 2
    assert "vcat::think" in kv._data
    assert "vcat::speak::person" in kv._data


def test_seed_batch_empty(store):
    assert store.seed_batch([]) == 0


# ---------------------------------------------------------------------------
# LMDB integration tests
# ---------------------------------------------------------------------------
lmdb_mod = pytest.importorskip("lmdb")


@pytest.fixture
def lmdb_cache(tmp_path):
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    c = LmdbAttributeCache(str(tmp_path / "aff-lmdb"), map_size=1 * 1024 * 1024, readonly=False)
    yield c
    c.close()


@pytest.fixture
def store_with_lmdb(kv, lmdb_cache):
    return CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb_cache)


def test_batch_lookup_uses_lmdb_l2(store_with_lmdb, kv, lmdb_cache):
    """LMDB L2 serves hits without going to KV."""
    payload = json.dumps({"category": "movement", "confidence": 0.8}).encode()
    lmdb_cache.put("vcat::run", payload)

    # KV does NOT have this key
    results = store_with_lmdb.get_categories_weighted_batch([("run", None)])
    assert results[("run", None)] == ("movement", 0.8)


def test_batch_lookup_kv_backfills_lmdb(store_with_lmdb, kv, lmdb_cache):
    """KV results get backfilled into LMDB for future L2 hits."""
    payload = json.dumps({"category": "cognitive", "confidence": 0.95}).encode()
    kv.set("vcat::think", payload)

    # First lookup: miss in LMDB, hit in KV -> should backfill LMDB
    store_with_lmdb.get_categories_weighted_batch([("think", None)])

    # Verify LMDB was backfilled
    val = lmdb_cache.get("vcat::think")
    assert val == payload


def test_batch_lookup_kv_miss_backfills_lmdb_sentinel(store_with_lmdb, kv, lmdb_cache):
    """KV misses should be backfilled as MISS_SENTINEL in LMDB."""
    from src.storage.lmdb_node_cache import _MISS_SENTINEL

    results = store_with_lmdb.get_categories_weighted_batch([("nonexistent", None)])
    assert results[("nonexistent", None)] is None

    val = lmdb_cache.get("vcat::nonexistent")
    assert val == _MISS_SENTINEL


def test_drain_pending_lmdb_writes(store_with_lmdb):
    """put_category_weighted accumulates LMDB relay writes."""
    store_with_lmdb.put_category_weighted("jump", "movement", confidence=0.9)
    store_with_lmdb.put_category_weighted("think", "cognitive", confidence=0.85)

    writes = store_with_lmdb.drain_pending_lmdb_writes()
    assert len(writes) == 2
    keys = [k for k, v in writes]
    assert "vcat::jump" in keys
    assert "vcat::think" in keys

    # Second drain should be empty
    assert store_with_lmdb.drain_pending_lmdb_writes() == []


def test_drain_pending_lmdb_writes_no_lmdb(store):
    """drain_pending_lmdb_writes works even without LMDB cache."""
    store.put_category_weighted("jump", "movement", confidence=0.9)
    writes = store.drain_pending_lmdb_writes()
    assert len(writes) == 1


def test_lmdb_l2_then_l1_cache(store_with_lmdb, kv, lmdb_cache):
    """After LMDB L2 hit, value is promoted to L1 in-memory cache."""
    payload = json.dumps({"category": "physical", "confidence": 0.7}).encode()
    lmdb_cache.put("vcat::push", payload)

    # First batch: L2 hit
    store_with_lmdb.get_categories_weighted_batch([("push", None)])

    # Close LMDB to prove L1 cache is serving
    lmdb_cache.close()
    results = store_with_lmdb.get_categories_weighted_batch([("push", None)])
    assert results[("push", None)] == ("physical", 0.7)
