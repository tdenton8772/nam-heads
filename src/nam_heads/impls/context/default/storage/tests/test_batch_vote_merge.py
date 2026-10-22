"""Vote-set writes reach KV in a constant number of round trips with the
same stored result as the per-key CAS loop (head flush and
context-lmdb-sync's write-through)."""
import json
import random

from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore
from nam_heads.impls.context.default.storage.merge import batch_cas_merge_set, cas_merge_set
from nam_heads.impls._testing.fake_kv import FakeKV


class CountingKV(FakeKV):
    def __init__(self):
        super().__init__()
        self.gets = self.batch_gets = self.batch_calls = 0

    def get(self, key):
        self.gets += 1
        return super().get(key)

    def pipeline_get(self, keys):
        self.batch_gets += 1
        return super().pipeline_get(keys)


def _votes(rng, key):
    doc = {"place": {"votes": {rng.choice(["NA/US/A", "NA/US/B"]): [f"w{rng.randint(0, 20)}"]}}}
    return json.dumps({"canonical_value": json.dumps(doc, sort_keys=True), "confidence": 1.0}).encode()


def test_batch_matches_per_key_over_random_histories():
    rng = random.Random(11)
    keys = [f"ctx::cx:L:k{i}::context_axis" for i in range(12)]
    for _ in range(25):
        a, b = FakeKV(), FakeKV()
        for _ in range(5):
            items = [(k, _votes(rng, k)) for k in rng.choices(keys, k=rng.randint(1, 20))]
            batch_cas_merge_set(a, items)
            for k, v in items:
                cas_merge_set(b, k, v)
        assert {k: json.loads(v) for k, v in a._data.items()} == \
            {k: json.loads(v) for k, v in b._data.items()}


def test_head_flush_is_constant_round_trips():
    kv = CountingKV()
    st = CouchbaseKVContextStore(kv)
    rng = random.Random(2)
    for i in range(30):
        doc = {"place": {"votes": {"NA/US/A": [f"w{i}", "w99"]}}}
        st.put(f"cx:L:k{i}", "context_axis", json.dumps(doc), confidence=1.0)
    st.get_batch([])                       # flush
    assert kv.gets == 0                    # no per-key GET
    assert kv.batch_gets == 1
    assert len(kv._data) == 30


def test_sync_write_through_is_constant_round_trips(monkeypatch):
    import services.nats_worker as nw
    kv = CountingKV()
    items = []
    for i in range(20):
        doc = {"amb": {"votes": {"NA/US/A": [f"w{i}"]}}}
        items.append((f"ctx::cx:L:k{i}::context_axis",
                      json.dumps({"canonical_value": json.dumps(doc), "confidence": 1.0}).encode()))
    items.append(("ctx::cx:s:paris::context_axis", b'{"canonical_value": "x", "confidence": 1.0}'))
    nw.write_context_kv(kv, items)
    assert kv.gets == 0 and kv.batch_gets == 1
    assert len(kv._data) == 21
