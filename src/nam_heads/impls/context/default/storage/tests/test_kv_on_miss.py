"""Tests for kv_on_cache_miss flag in CouchbaseKVContextStore."""
import json
from unittest.mock import MagicMock
import pytest

from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore


def _make_store(kv_on_miss=True):
    kv = MagicMock()
    kv.get.return_value = (None, None)
    kv.pipeline_get.return_value = {}
    kv.pipeline_set.return_value = None
    return CouchbaseKVContextStore(kv, kv_on_cache_miss=kv_on_miss), kv


class TestKvOnCacheMissDefault:
    def test_default_is_true(self):
        kv = MagicMock()
        store = CouchbaseKVContextStore(kv)
        assert store._kv_on_miss is True

    def test_false_constructor(self):
        store, _ = _make_store(kv_on_miss=False)
        assert store._kv_on_miss is False


class TestSingleLookup:
    def test_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        result = store.get("hello world", "tense")
        kv.get.assert_not_called()
        assert result is None

    def test_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        doc = json.dumps({"canonical_value": "present", "confidence": 1.0}).encode()
        store._cache["ctx::hello world::tense"] = doc
        result = store.get("hello world", "tense")
        assert result == "present"
        kv.get.assert_not_called()

    def test_kv_on_miss_true_calls_kv(self):
        store, kv = _make_store(kv_on_miss=True)
        store.get("hello world", "tense")
        kv.get.assert_called_once()


class TestBatchLookup:
    def test_batch_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        queries = [("hello", "tense"), ("world", "mood")]
        results = store.get_batch(queries)
        kv.pipeline_get.assert_not_called()
        assert results[("hello", "tense")] is None

    def test_batch_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        doc = json.dumps({"canonical_value": "present", "confidence": 1.0}).encode()
        store._cache["ctx::hello::tense"] = doc
        queries = [("hello", "tense")]
        results = store.get_batch(queries)
        assert results[("hello", "tense")] == "present"
        kv.pipeline_get.assert_not_called()
