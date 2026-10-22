"""After a session flush, the affordance store must not answer from its L1 or send
writes queued before the flush (cold reset, src/storage/session_reset.py)."""
import json

from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


def _store_with_cached(lemma="bright"):
    kv = FakeKV()
    store = CouchbaseKVCategoryStore(kv)
    kv.set(store._key(lemma), json.dumps({"category": "quality", "confidence": 0.9}).encode())
    assert store.get_category(lemma) == "quality"
    return kv, store


def test_a_flushed_bucket_still_answers_from_l1_until_reset():
    kv, store = _store_with_cached()
    kv._data.clear()                                   # the session flush
    assert store.get_category("bright") == "quality"   # stale: the warm state
    out = store.reset_session_state()
    assert out["l1"] >= 1
    assert store.get_category("bright") is None


def test_reset_drops_writes_queued_before_the_flush():
    _kv, store = _store_with_cached()
    store._pending_writes.append(("acat::old", b"{}"))
    store._pending_lmdb_writes.append(("acat::old", b"{}"))
    out = store.reset_session_state()
    assert out["pending_writes"] == 1 and out["pending_relay"] == 1
    assert store.drain_pending_lmdb_writes() == []
    assert store._pending_writes == []


def test_reset_on_an_empty_store_is_a_no_op():
    store = CouchbaseKVCategoryStore(FakeKV())
    assert store.reset_session_state() == {"l1": 0, "pending_writes": 0, "pending_relay": 0}


def test_reset_clears_the_parsed_doc_memos():
    _kv, store = _store_with_cached()
    store._parsed["k"] = (b"x", {})
    store._parsed_kv["k"] = (b"x", {})
    store._payload_docs["k"] = (b"x", {})
    store._merge_memo["k"] = ()
    store.reset_session_state()
    assert not (store._parsed or store._parsed_kv or store._payload_docs or store._merge_memo)
