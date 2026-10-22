"""The KV store's parsed-doc cache and merge memo change no stored value.

CouchbaseKVCategoryStore keeps parsed predicate docs beside L1 and memoizes
each key's last merge (put_doc into L1, then the KV flush), so a doc is
decoded and merged once per batch instead of three times. These tests replay
random multi-writer histories and compare every L1 value, KV value and LMDB
relay payload with the plain ``merge_payloads`` semantics.
"""
import json
import random

from nam_heads.impls.affordance.default.storage import merge as M
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
from nam_heads.impls._testing.fake_kv import FakeKV

KEYS = [f"p:v{i}" for i in range(6)] + ["b:found", "ph:grow:up"]
ONTS = ["cco:ont00001262", "cco:ont00001180", "BFO:0000015"]


def _doc(rng, witness):
    d = {"category": "predicate", "confidence": 1.0, "kind": rng.choice(["verb", "phrasal"]),
         "lemma": rng.choice(["found", "grow"]),
         "_g": {"count": {witness: rng.randint(1, 40)}}}
    if rng.random() < 0.7:
        d["_go"] = {rng.choice(ONTS): {witness: rng.randint(1, 9)}}
    if rng.random() < 0.15:
        d["kind"] = "seed"
        d["seed_version"] = rng.choice(["a", "b"])
    return d


def test_merge_payload_docs_matches_merge_payloads():
    rng = random.Random(3)
    cases = [None, {}, {"category": "x", "confidence": 1.0}]
    for _ in range(300):
        a = rng.choice(cases + [_doc(rng, rng.choice("pq"))])
        b = rng.choice(cases[1:] + [_doc(rng, rng.choice("pq"))])
        want = M.merge_payloads(None if a is None else M.encode(a), M.encode(b))
        got = M.merge_payload_docs(a, b)
        assert (M.encode(b) if got is b else M.encode(got)) == want


class PlainStore(CouchbaseKVCategoryStore):
    """The store without the parsed-doc cache, merge memo or merge hint:
    every read json-decodes, every put and flush merges with merge_payloads
    (the behaviour before the cache)."""

    def _parse(self, key, raw):
        return self._parse_free(raw)

    def _merge(self, key, cur, payload, kv=False):
        return M.merge_payloads(cur, payload)

    def _is_mergeable(self, key, payload):
        return M.is_mergeable(payload)

    def put_doc(self, name, doc, merged_from=None):
        return super().put_doc(name, doc, merged_from=None)


def test_random_histories_match_the_plain_store():
    """Same operations on both stores (each with its own KV, the same other
    writers): every read, L1 value, KV value and relay payload is equal."""
    for seed in range(40):
        rng = random.Random(seed)
        kv_a, kv_b = FakeKV(), FakeKV()
        a, b = CouchbaseKVCategoryStore(kv_a), PlainStore(kv_b)
        for _ in range(80):
            op = rng.random()
            names = rng.sample(KEYS, rng.randint(1, 3))
            if op < 0.5:
                ra, rb = a.get_docs_batch(names), b.get_docs_batch(names)
                assert ra == rb
                for n in names:
                    d = _doc(rng, "pod-0")
                    hint = (ra[n], M.merge_docs(ra[n], d)) if ra[n] is not None and rng.random() < 0.8 else None
                    a.put_doc(n, d, merged_from=hint)
                    b.put_doc(n, d)
            elif op < 0.8:
                # Another writer (a second pod, or affordance-lmdb-sync's
                # write-through) merges into KV directly.
                other = M.encode(_doc(rng, rng.choice(["pod-1", "pod-2"])))
                for kv in (kv_a, kv_b):
                    M.cas_merge_set(kv, "vcat::" + names[0], other)
            else:
                a._flush_pending()
                b._flush_pending()
            assert a._cache == b._cache
            assert kv_a._data == kv_b._data
        a._flush_pending()
        b._flush_pending()
        assert kv_a._data == kv_b._data and a._cache == b._cache
        assert a.drain_pending_lmdb_writes() == b.drain_pending_lmdb_writes()


def test_flush_reads_back_no_doc_it_already_knows():
    """After the first flush, a key whose KV value is the one L1 held is
    neither re-parsed nor re-merged (json.loads counted)."""
    kv = FakeKV()
    st = CouchbaseKVCategoryStore(kv)
    rng = random.Random(1)
    for _ in range(3):
        cur = st.get_docs_batch(["p:found"])["p:found"]
        d = _doc(rng, "pod-0")
        st.put_doc("p:found", d, merged_from=(cur, M.merge_docs(cur, d)) if cur else None)
    st._flush_pending()
    calls = {"n": 0}
    real = json.loads

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    json.loads = counting
    try:
        cur = st.get_docs_batch(["p:found"])["p:found"]
        d = _doc(rng, "pod-0")
        st.put_doc("p:found", d, merged_from=(cur, M.merge_docs(cur, d)))
        st._flush_pending()
    finally:
        json.loads = real
    assert calls["n"] == 0


def test_returned_docs_are_shared_and_not_mutated_by_the_store():
    kv = FakeKV()
    st = CouchbaseKVCategoryStore(kv)
    st.put_doc("p:found", {"category": "predicate", "confidence": 1.0, "_g": {"count": {"a": 1}}})
    d1 = st.get_docs_batch(["p:found"])["p:found"]
    snap = json.dumps(d1, sort_keys=True)
    st.put_doc("p:found", {"category": "predicate", "confidence": 1.0, "_g": {"count": {"a": 5}}})
    st._flush_pending()
    assert json.dumps(d1, sort_keys=True) == snap           # the old doc object is untouched
    assert st.get_docs_batch(["p:found"])["p:found"]["count"] == 5


def _counters(raw):
    d = json.loads(raw)
    return d.get("_g"), d.get("_go"), d.get("count")


def test_async_flush_stores_the_same_counters():
    """enable_async_flush (the worker's mode): the flush runs on its own
    thread; after _flush_pending KV holds what the synchronous store holds,
    and L1 holds the same counters (a flush racing a put joins, never drops)."""
    for seed in range(15):
        rng = random.Random(100 + seed)
        kv_a, kv_b = FakeKV(), FakeKV()
        a, b = CouchbaseKVCategoryStore(kv_a), PlainStore(kv_b)
        a.enable_async_flush()
        for _ in range(60):
            names = rng.sample(KEYS, rng.randint(1, 3))
            a.get_docs_batch(names)
            b.get_docs_batch(names)
            for n in names:
                d = _doc(rng, "pod-0")
                a.put_doc(n, d)
                b.put_doc(n, d)
        a._flush_pending()
        b._flush_pending()
        assert {k: _counters(v) for k, v in kv_a._data.items()} == \
            {k: _counters(v) for k, v in kv_b._data.items()}
        assert {k: _counters(v) for k, v in a._cache.items() if isinstance(v, bytes)} == \
            {k: _counters(v) for k, v in b._cache.items() if isinstance(v, bytes)}
        assert a.drain_pending_lmdb_writes() == b.drain_pending_lmdb_writes()


def test_async_flush_does_not_block_the_batch_read():
    import threading
    gate = threading.Event()

    class SlowKV(FakeKV):
        def pipeline_get(self, keys):
            gate.wait(5)
            return super().pipeline_get(keys)

    st = CouchbaseKVCategoryStore(SlowKV())
    st.enable_async_flush()
    st.put_doc("p:found", {"category": "predicate", "confidence": 1.0, "_g": {"count": {"a": 1}}})
    out = st.get_docs_batch(["p:found"])          # returns while the flush waits on KV
    assert out["p:found"]["count"] == 1
    assert st._flush_future is not None and not st._flush_future.done()
    gate.set()
    st._flush_pending()
    assert json.loads(st._kv.get("vcat::p:found")[0])["count"] == 1
