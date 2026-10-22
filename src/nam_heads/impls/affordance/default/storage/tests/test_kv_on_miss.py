"""Tests for kv_on_cache_miss flag in CouchbaseKVCategoryStore (affordance)."""
import json
from unittest.mock import MagicMock
import pytest

from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


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
    def test_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        result = store.get_category_weighted("repair")
        kv.get.assert_not_called()
        assert result is None

    def test_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        doc = json.dumps({"category": "physical", "confidence": 1.0}).encode()
        store._cache["vcat::repair"] = doc
        result = store.get_category_weighted("repair")
        assert result == ("physical", 1.0)
        kv.get.assert_not_called()


class TestBatchLookup:
    def test_batch_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        queries = [("repair", None), ("build", "tool")]
        results = store.get_categories_weighted_batch(queries)
        kv.pipeline_get.assert_not_called()
        assert results[("repair", None)] is None

    def test_batch_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        doc = json.dumps({"category": "physical", "confidence": 1.0}).encode()
        store._cache["vcat::repair"] = doc
        queries = [("repair", None)]
        results = store.get_categories_weighted_batch(queries)
        assert results[("repair", None)] == ("physical", 1.0)
        kv.pipeline_get.assert_not_called()
