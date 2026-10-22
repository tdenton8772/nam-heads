"""KV category store: predicate documents (seed, one-round-trip read, write, LMDB relay)."""
import json

from nam_heads.impls._testing.fake_kv import FakeKV
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore


class CountingKV(FakeKV):
    def __init__(self):
        super().__init__()
        self.batch_gets = 0

    def pipeline_get(self, keys):
        self.batch_gets += 1
        return super().pipeline_get(keys)


def test_docs_seed_read_write_relay():
    kv = CountingKV()
    st = CouchbaseKVCategoryStore(kv)
    seed = {"category": "verb_base", "confidence": 1.0, "kind": "seed", "seed_version": "v1", "_g": {}}
    assert st.seed_docs([("b:found", seed)]) == 1
    assert st.seed_docs([("b:found", dict(seed))]) == 0          # same version: no write
    kv.batch_gets = 0
    st2 = CouchbaseKVCategoryStore(kv)            # cold cache: one KV round trip
    got = st2.get_docs_batch(["b:found", "p:found", "b:found"])
    assert kv.batch_gets == 1
    assert got["b:found"]["kind"] == "seed" and got["p:found"] is None
    st2.put_doc("p:found", {"category": "predicate", "confidence": 1.0, "count": 1})
    assert st2.get_docs_batch(["p:found"])["p:found"]["count"] == 1   # L1, then flushed
    assert json.loads(kv.get("vcat::p:found")[0])["count"] == 1
    assert "vcat::p:found" in dict(st2.drain_pending_lmdb_writes())


def test_seed_warms_l1_when_kv_on_miss_is_off():
    """kv_on_cache_miss=False, no LMDB: the seeded lexicon is still visible,
    with the counts already in KV."""
    kv = FakeKV()
    kv.set("vcat::b:found", json.dumps({"category": "verb_base", "confidence": 1.0,
                                        "kind": "seed", "count": 7}).encode())
    st = CouchbaseKVCategoryStore(kv, kv_on_cache_miss=False)
    st.seed_docs([("b:found", {"category": "verb_base", "confidence": 1.0, "kind": "seed"}),
                  ("b:join", {"category": "verb_base", "confidence": 1.0, "kind": "seed"})])
    got = st.get_docs_batch(["b:found", "b:join", "ph:lay:down"])
    assert got["b:found"]["count"] == 7 and got["b:join"]["kind"] == "seed"
    assert got["ph:lay:down"] is None             # not seeded, not fetched: miss


def test_legacy_category_reader_sees_well_formed_doc():
    kv = FakeKV()
    st = CouchbaseKVCategoryStore(kv)
    st.put_doc("p:found", {"category": "predicate", "confidence": 1.0, "count": 3})
    st.get_docs_batch([])                         # flush
    assert st.get_category_weighted("p:found") == ("predicate", 1.0)
