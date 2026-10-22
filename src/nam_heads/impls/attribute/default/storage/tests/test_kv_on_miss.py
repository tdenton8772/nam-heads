"""Tests for kv_on_cache_miss flag in CouchbaseKVCategoryStore (attribute)."""
import json
from unittest.mock import MagicMock
import pytest

from nam_heads.impls.attribute.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


def _make_store(kv_on_miss=True, lmdb=None):
    kv = MagicMock()
    kv.get.return_value = (None, None)
    kv.pipeline_get.return_value = {}
    return CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb, kv_on_cache_miss=kv_on_miss), kv


class TestKvOnCacheMissDefault:
    def test_default_is_true(self):
        kv = MagicMock()
        store = CouchbaseKVCategoryStore(kv)
        assert store._kv_on_miss is True

    def test_false_constructor(self):
        store, _ = _make_store(kv_on_miss=False)
        assert store._kv_on_miss is False


class TestPerRecordLookup:
    def test_kv_on_miss_true_calls_kv(self):
        store, kv = _make_store(kv_on_miss=True)
        result = store.get_category_weighted("large")
        assert kv.get.called

    def test_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        result = store.get_category_weighted("large")
        kv.get.assert_not_called()
        assert result is None

    def test_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        # Pre-populate L1 cache
        doc = json.dumps({"category": "size", "confidence": 1.0}).encode()
        store._cache["acat::large"] = doc
        result = store.get_category_weighted("large")
        assert result == ("size", 1.0)
        kv.get.assert_not_called()

    def test_kv_on_miss_false_with_lmdb_hit(self):
        lmdb = MagicMock()
        lmdb.available = True
        doc = json.dumps({"category": "size", "confidence": 0.9}).encode()
        lmdb.get.return_value = doc
        store, kv = _make_store(kv_on_miss=False, lmdb=lmdb)
        result = store.get_category_weighted("large")
        assert result == ("size", 0.9)
        kv.get.assert_not_called()


class TestBatchLookup:
    def test_batch_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        queries = [("large", None), ("small", "weapon")]
        results = store.get_categories_weighted_batch(queries)
        kv.pipeline_get.assert_not_called()
        assert results[("large", None)] is None
        assert results[("small", "weapon")] is None

    def test_batch_kv_on_miss_true_calls_kv(self):
        store, kv = _make_store(kv_on_miss=True)
        kv.pipeline_get.return_value = {}
        queries = [("large", None)]
        store.get_categories_weighted_batch(queries)
        assert kv.pipeline_get.called

    def test_batch_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        doc = json.dumps({"category": "size", "confidence": 1.0}).encode()
        store._cache["acat::large"] = doc
        queries = [("large", None)]
        results = store.get_categories_weighted_batch(queries)
        assert results[("large", None)] == ("size", 1.0)
        kv.pipeline_get.assert_not_called()
