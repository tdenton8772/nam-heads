"""After a session flush, the context store must not answer from its L1 or
send writes queued before the flush (cold reset, src/storage/session_reset.py)."""
import json

from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore


def test_a_flushed_bucket_still_answers_from_l1_until_reset():
    kv = FakeKV()
    store = CouchbaseKVContextStore(kv)
    kv.set(store._key("hernando", "place"),
           json.dumps({"canonical_value": "NA/US/US-MS", "confidence": 1.0}).encode())
    assert store.get("hernando", "place") == "NA/US/US-MS"
    kv._data.clear()                                          # the session flush
    assert store.get("hernando", "place") == "NA/US/US-MS"   # stale: the warm state
    assert store.reset_session_state()["l1"] >= 1
    assert store.get("hernando", "place") is None


def test_reset_drops_writes_queued_before_the_flush():
    store = CouchbaseKVContextStore(FakeKV())
    store._pending_writes.append(("ctx::old", b"{}"))
    store._pending_lmdb_writes.append(("ctx::old", b"{}"))
    assert store.reset_session_state() == {"l1": 0, "pending_writes": 1, "pending_relay": 1}
    assert store.drain_pending_lmdb_writes() == []
