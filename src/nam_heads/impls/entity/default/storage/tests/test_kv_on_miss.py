"""Tests for kv_on_cache_miss flag in CouchbaseKVEntityStore."""
from unittest.mock import MagicMock
import pytest

from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore


def _make_store(kv_on_miss=True):
    kv = MagicMock()
    kv.get.return_value = (None, None)
    kv.pipeline_get.return_value = {}
    return CouchbaseKVEntityStore(kv, kv_on_cache_miss=kv_on_miss), kv


class TestKvOnCacheMissDefault:
    def test_default_is_true(self):
        kv = MagicMock()
        store = CouchbaseKVEntityStore(kv)
        assert store._kv_on_miss is True

    def test_false_constructor(self):
        store, _ = _make_store(kv_on_miss=False)
        assert store._kv_on_miss is False


class TestResolveSurfaceForm:
    def test_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        result = store.resolve_surface_form("Berlin")
        kv.get.assert_not_called()
        assert result is None

    def test_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        store._surf_cache["surf::Berlin"] = b"ent_123"
        result = store.resolve_surface_form("Berlin")
        assert result == "ent_123"
        kv.get.assert_not_called()

    def test_kv_on_miss_true_calls_kv(self):
        store, kv = _make_store(kv_on_miss=True)
        store.resolve_surface_form("Berlin")
        kv.get.assert_called_once()


class TestResolveBatch:
    def test_batch_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        queries = [("Berlin", "location"), ("hammer", "tool")]
        results = store.resolve_surface_forms_batch(queries)
        kv.pipeline_get.assert_not_called()
        assert results[("Berlin", "location")] is None

    def test_batch_kv_on_miss_false_returns_cached(self):
        store, kv = _make_store(kv_on_miss=False)
        store._surf_cache["surf::location::Berlin"] = b"ent_123"
        queries = [("Berlin", "location")]
        results = store.resolve_surface_forms_batch(queries)
        assert results[("Berlin", "location")] == "ent_123"
        kv.pipeline_get.assert_not_called()


class TestLookupCandidateKeys:
    def test_kv_on_miss_false_skips_kv(self):
        store, kv = _make_store(kv_on_miss=False)
        results = store.lookup_candidate_keys_by_key(["ngram_berlin"])
        kv.pipeline_get.assert_not_called()
        assert results["ngram_berlin"] == []

    def test_kv_on_miss_false_returns_cached(self):
        import json
        store, kv = _make_store(kv_on_miss=False)
        store._cand_cache["cand::ngram_berlin"] = json.dumps(
            [["ent_123", "Berlin"]]
        ).encode()
        results = store.lookup_candidate_keys_by_key(["ngram_berlin"])
        assert results["ngram_berlin"] == [("ent_123", "Berlin")]
        kv.pipeline_get.assert_not_called()

    def test_kv_on_miss_true_calls_kv(self):
        store, kv = _make_store(kv_on_miss=True)
        store.lookup_candidate_keys_by_key(["ngram_berlin"])
        kv.pipeline_get.assert_called_once()

    def test_kv_on_miss_false_marks_uncached_as_miss(self):
        """Uncached keys get _MISS sentinel so subsequent lookups don't retry."""
        store, kv = _make_store(kv_on_miss=False)
        store.lookup_candidate_keys_by_key(["ngram_a", "ngram_b"])
        # Both should now be cached as _MISS
        from nam_heads.impls.entity.default.storage.couchbase_kv_store import _MISS
        assert store._cand_cache["cand::ngram_a"] is _MISS
        assert store._cand_cache["cand::ngram_b"] is _MISS
        kv.pipeline_get.assert_not_called()
