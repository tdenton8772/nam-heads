"""Store-layer concurrency fixes: vote-set merge on write (KV CAS, LMDB sync)
and expiring negative cache entries. Uses the real LMDB cache class."""
import json

import pytest

from nam_heads.impls.context.default.storage import couchbase_kv_store as cks
from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore
from nam_heads.impls.context.default.storage.merge import (
    ERR_KEY_EEXISTS, cas_merge_set, is_mergeable, merge_payloads, merge_with_lmdb)


def votes_payload(kind, value, *witnesses):
    doc = {kind: {"votes": {value: sorted(witnesses)}}}
    return json.dumps({"canonical_value": json.dumps(doc), "confidence": 1.0}).encode()


def votes_of(payload):
    return json.loads(json.loads(payload)["canonical_value"])


class FakeKV:
    """Dict KV with CAS. ``race`` injects one foreign write between a GET and
    the following SET, as a concurrent pod would."""

    def __init__(self):
        self.d, self.cas = {}, {}
        self.race = None

    def get(self, k):
        return self.d.get(k), self.cas.get(k, 0)

    def pipeline_get(self, keys):
        return {k: (self.d.get(k), self.cas.get(k)) for k in keys}

    def _write(self, k, v):
        self.d[k] = v
        self.cas[k] = self.cas.get(k, 0) + 1

    def add(self, k, v):
        if self.race:
            rk, rv = self.race
            self.race = None
            self._write(rk, rv)
        if k in self.d:
            return False
        self._write(k, v)
        return True

    def set(self, k, v, cas=0):
        if self.race:
            rk, rv = self.race
            self.race = None
            self._write(rk, rv)
        if cas and self.cas.get(k) != cas:
            return ERR_KEY_EEXISTS
        self._write(k, v)
        return 0

    def pipeline_set(self, items):
        for k, v in items:
            self._write(k, v)
        return [(k, 0) for k, _ in items]


# ---- merge -----------------------------------------------------------------
def test_merge_is_union_commutative_idempotent():
    a = votes_payload("amb", "X", "w1")
    b = votes_payload("amb", "X", "w2")
    c = votes_payload("amb", "Y", "w3")
    ab = merge_payloads(a, b)
    assert votes_of(ab)["amb"]["votes"]["X"] == ["w1", "w2"]
    assert votes_of(merge_payloads(merge_payloads(a, b), c)) == \
        votes_of(merge_payloads(merge_payloads(c, a), b))
    assert votes_of(merge_payloads(ab, a)) == votes_of(ab)  # replay is a no-op


def test_plain_values_stay_last_writer_wins():
    plain_old = json.dumps({"canonical_value": "united_states", "confidence": 1.0}).encode()
    plain_new = json.dumps({"canonical_value": "canada", "confidence": 1.0}).encode()
    assert not is_mergeable(plain_new)
    assert merge_payloads(plain_old, plain_new) == plain_new
    assert merge_payloads(votes_payload("amb", "X", "w"), plain_new) == plain_new


# ---- KV: CAS merge -----------------------------------------------------------
def test_cas_merge_survives_a_concurrent_writer():
    kv = FakeKV()
    k = "ctx::cx:L:richmond::context_axis"
    kv.pipeline_set([(k, votes_payload("amb", "VA", "pod-a-1"))])
    kv.race = (k, votes_payload("amb", "VA", "pod-b-1"))  # lands between our GET and SET
    assert cas_merge_set(kv, k, votes_payload("amb", "VA", "pod-c-1"))
    assert set(votes_of(kv.d[k])["amb"]["votes"]["VA"]) >= {"pod-b-1", "pod-c-1"}


def test_cas_merge_on_a_key_created_meanwhile():
    kv = FakeKV()
    k = "ctx::cx:L:nfl::context_axis"
    kv.race = (k, votes_payload("org:NFL", "national_football_league", "w1"))
    assert cas_merge_set(kv, k, votes_payload("org:NFL", "national_football_league", "w2"))
    assert votes_of(kv.d[k])["org:NFL"]["votes"]["national_football_league"] == ["w1", "w2"]


def test_store_flush_merges_two_pods_writes():
    kv = FakeKV()
    a = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    b = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    doc_a = json.dumps({"amb": {"votes": {"VA": ["wa"]}}})
    doc_b = json.dumps({"amb": {"votes": {"VA": ["wb"]}}})
    a.put("cx:L:richmond", "context_axis", doc_a)
    b.put("cx:L:richmond", "context_axis", doc_b)
    a._flush_pending()
    b._flush_pending()
    got = votes_of(kv.d["ctx::cx:L:richmond::context_axis"])
    assert got["amb"]["votes"]["VA"] == ["wa", "wb"]


def test_store_flush_plain_values_use_batch_set():
    kv = FakeKV()
    st = CouchbaseKVContextStore(kv, kv_on_cache_miss=False)
    st.put("ohio", "geographic", "united_states")
    st._flush_pending()
    assert json.loads(kv.d["ctx::ohio::geographic"])["canonical_value"] == "united_states"


# ---- negative cache TTL --------------------------------------------------------
def test_kv_miss_expires(monkeypatch):
    kv = FakeKV()
    st = CouchbaseKVContextStore(kv, kv_on_cache_miss=True)
    assert st.get_batch([("cx:L:x", "context_axis")])[("cx:L:x", "context_axis")] is None
    kv.pipeline_set([("ctx::cx:L:x::context_axis", votes_payload("place", "P", "w1"))])
    # Within the TTL the miss is still cached.
    assert st.get_batch([("cx:L:x", "context_axis")])[("cx:L:x", "context_axis")] is None
    # Expire it: the entry another pod wrote becomes visible.
    for v in st._cache.values():
        if isinstance(v, cks._Miss):
            v.expires = 0
    assert st.get_batch([("cx:L:x", "context_axis")])[("cx:L:x", "context_axis")] is not None
    assert st.get("cx:L:x", "context_axis") is not None


# ---- real LMDB: cold cache, KV_ON_MISS=0, sync of cx: keys ------------------------
@pytest.fixture
def lmdb_pair(tmp_path):
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    writer = LmdbAttributeCache(str(tmp_path), map_size=16 * 1024 * 1024, readonly=False)
    if not writer.available:
        pytest.skip("lmdb not available")
    writer.batch_put([("ctx::__init__::x", b"{}")])  # creates data.mdb for readers
    reader = LmdbAttributeCache(str(tmp_path), map_size=16 * 1024 * 1024, readonly=True)
    yield writer, reader
    reader.close()
    writer.close()


def test_cold_lmdb_kv_on_miss_off_then_sync_delivers(lmdb_pair):
    writer, reader = lmdb_pair
    kv = FakeKV()
    kv.pipeline_set([("ctx::cx:L:richmond::context_axis", votes_payload("amb", "VA", "w1"))])
    st = CouchbaseKVContextStore(kv, lmdb_cache=reader, kv_on_cache_miss=False)
    q = [("cx:L:richmond", "context_axis")]
    # In KV but not yet in LMDB: invisible (KV_ON_MISS=0), and NOT cached as a miss.
    assert st.get_batch(q)[q[0]] is None
    assert "ctx::cx:L:richmond::context_axis" not in st._cache
    # context-lmdb-sync delivers two pods' writes of the same cx:L: key and a
    # plain cx:s: key: vote sets are unioned, the plain value is written as is.
    from services.nats_worker import merge_context_writes, write_context_kv
    writes = [(b"ctx::cx:L:richmond::context_axis", votes_payload("amb", "VA", "w1")),
              (b"ctx::cx:L:richmond::context_axis", votes_payload("amb", "VA", "w2")),
              (b"ctx::cx:s:atlantis::context_axis",
               json.dumps({"canonical_value": '{"place":[["r-atlantis","region",0]]}',
                           "confidence": 1.0}).encode())]
    merged = merge_context_writes(writer, writes)
    writer.batch_put(merged)
    write_context_kv(kv, [(k.decode(), v) for k, v in merged])
    got = st.get_batch(q + [("cx:s:atlantis", "context_axis")])
    assert json.loads(got[q[0]])["amb"]["votes"]["VA"] == ["w1", "w2"]
    assert json.loads(got[("cx:s:atlantis", "context_axis")])["place"][0][0] == "r-atlantis"
    # A later sync message carrying only one pod's view does not shrink it.
    writer.batch_put(merge_context_writes(
        writer, [(b"ctx::cx:L:richmond::context_axis", votes_payload("amb", "VA", "w3"))]))
    raw = reader.get("ctx::cx:L:richmond::context_axis")
    assert votes_of(raw)["amb"]["votes"]["VA"] == ["w1", "w2", "w3"]
    assert votes_of(kv.d["ctx::cx:L:richmond::context_axis"])["amb"]["votes"]["VA"] == ["w1", "w2"]


def test_lmdb_batch_get_cost(lmdb_pair):
    """Report the L2 cost of one batch's context lookups (not an assertion on speed)."""
    import time
    writer, reader = lmdb_pair
    writer.batch_put([(f"ctx::cx:L:k{i}::context_axis", b"{}") for i in range(2000)])
    keys = [f"ctx::cx:L:k{i}::context_axis" for i in range(0, 20000, 2)]  # 10k keys, 10% hits
    t0 = time.perf_counter()
    reader.batch_get(keys)
    ms = (time.perf_counter() - t0) * 1000
    print(f"lmdb batch_get 10000 keys: {ms:.1f} ms")
    assert ms < 5000
