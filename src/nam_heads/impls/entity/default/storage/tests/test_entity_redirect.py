"""Tests for entity redirect pointer support in EntityStore implementations."""
import pytest
from nam_heads.impls.entity.default.storage.memory_store import MemoryEntityStore


@pytest.fixture
def store():
    return MemoryEntityStore()


class TestWriteAndResolveRedirect:
    def test_basic_redirect(self, store):
        store.write_redirect("abc123", "def456")
        assert store.resolve_redirect("abc123") == "def456"

    def test_no_redirect_returns_self(self, store):
        assert store.resolve_redirect("abc123") == "abc123"

    def test_idempotent_write(self, store):
        store.write_redirect("abc123", "def456")
        store.write_redirect("abc123", "def456")
        assert store.resolve_redirect("abc123") == "def456"
        # Reverse index should not have duplicates
        assert store.reverse_redirects("def456") == ["abc123"]

    def test_chain_walking(self, store):
        """A -> B -> C resolves A to C."""
        store.write_redirect("aaa", "bbb")
        store.write_redirect("bbb", "ccc")
        assert store.resolve_redirect("aaa") == "ccc"
        assert store.resolve_redirect("bbb") == "ccc"
        assert store.resolve_redirect("ccc") == "ccc"

    def test_cycle_detection(self, store):
        """Cycle A -> B -> A stops without infinite loop."""
        store._redirects["aaa"] = "bbb"
        store._redirects["bbb"] = "aaa"
        # Should terminate, returning whichever it stops on
        result = store.resolve_redirect("aaa")
        assert result in ("aaa", "bbb")

    def test_max_depth(self, store):
        """Chain longer than 10 hops stops at hop 10."""
        ids = [f"id_{i:03d}" for i in range(15)]
        for i in range(14):
            store._redirects[ids[i]] = ids[i + 1]
        # From id_000, should follow at most 10 hops (reaching id_010)
        result = store.resolve_redirect(ids[0])
        assert result == ids[10]

    def test_same_id_no_op(self, store):
        """write_redirect(x, x) is a no-op."""
        store.write_redirect("abc", "abc")
        assert store.resolve_redirect("abc") == "abc"
        assert store.reverse_redirects("abc") == []


class TestReverseRedirects:
    def test_single_reverse(self, store):
        store.write_redirect("old1", "canonical")
        assert store.reverse_redirects("canonical") == ["old1"]

    def test_multiple_reverse(self, store):
        store.write_redirect("old1", "canonical")
        store.write_redirect("old2", "canonical")
        result = store.reverse_redirects("canonical")
        assert set(result) == {"old1", "old2"}

    def test_transitive_reverse(self, store):
        """A -> B -> C: reverse_redirects(C) should return [B, A]."""
        store.write_redirect("aaa", "bbb")
        store.write_redirect("bbb", "ccc")
        result = store.reverse_redirects("ccc")
        assert set(result) == {"aaa", "bbb"}

    def test_no_reverse(self, store):
        assert store.reverse_redirects("nonexistent") == []

    def test_reverse_does_not_include_self(self, store):
        store.write_redirect("old", "canonical")
        result = store.reverse_redirects("canonical")
        assert "canonical" not in result


class TestCouchbaseKVStoreRedirect:
    """Test redirect support on CouchbaseKVEntityStore using MockKV."""

    @pytest.fixture
    def cb_store(self):
        from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore

        class MockKV:
            def __init__(self):
                self._data = {}
                self._cas = 0

            def get(self, key):
                if key in self._data:
                    return self._data[key]
                return None, None

            def set(self, key, value, cas=0, expiration=0):
                self._cas += 1
                self._data[key] = (value, self._cas)
                return self._cas

            def pipeline_get(self, keys):
                return {k: self.get(k) for k in keys}

            def pipeline_set(self, items):
                return [(k, self.set(k, v)) for k, v in items]

        return CouchbaseKVEntityStore(MockKV())

    def test_write_and_resolve(self, cb_store):
        cb_store.write_redirect("abc123", "def456")
        assert cb_store.resolve_redirect("abc123") == "def456"

    def test_no_redirect(self, cb_store):
        assert cb_store.resolve_redirect("abc123") == "abc123"

    def test_idempotent(self, cb_store):
        cb_store.write_redirect("abc123", "def456")
        cb_store.write_redirect("abc123", "def456")
        assert cb_store.resolve_redirect("abc123") == "def456"
        rev = cb_store.reverse_redirects("def456")
        assert rev == ["abc123"]

    def test_chain(self, cb_store):
        cb_store.write_redirect("aaa", "bbb")
        cb_store.write_redirect("bbb", "ccc")
        assert cb_store.resolve_redirect("aaa") == "ccc"

    def test_reverse(self, cb_store):
        cb_store.write_redirect("old1", "canonical")
        cb_store.write_redirect("old2", "canonical")
        result = cb_store.reverse_redirects("canonical")
        assert set(result) == {"old1", "old2"}

    def test_transitive_reverse(self, cb_store):
        cb_store.write_redirect("aaa", "bbb")
        cb_store.write_redirect("bbb", "ccc")
        result = cb_store.reverse_redirects("ccc")
        assert set(result) == {"aaa", "bbb"}

    def test_cache_hit(self, cb_store):
        """Second resolve_redirect should hit L1 cache, not KV."""
        cb_store.write_redirect("abc123", "def456")
        assert cb_store.resolve_redirect("abc123") == "def456"
        # Clear KV data to prove cache is being used
        cb_store._kv._data.pop("redir::abc123", None)
        assert cb_store.resolve_redirect("abc123") == "def456"
