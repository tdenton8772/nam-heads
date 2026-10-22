"""Predicate lexicon on the production read path: real LMDB cache on a temp
dir, KV behind it, NAM_AFFORDANCE_KV_ON_MISS=0 (the chart default).

1. Seed alone (cold LMDB, no learned entries) gives the same output as the
   memory backend: seeding warms L1 with every seed key.
2. Every new key prefix (b:, s:, ph:, p:, and auto-classified categories)
   rides the LMDB relay exactly as affordance-lmdb-sync applies it
   (serialize -> deserialize -> LmdbAttributeCache.batch_put), and a fresh
   reader with KV_ON_MISS=0 sees them from LMDB.
"""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

pytest.importorskip("lmdb")

from src.storage.lmdb_node_cache import LmdbAttributeCache
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
from nam_heads.impls.affordance.default.storage.factory import _seed_store
from nam_heads.impls._testing.fake_kv import FakeKV
from src.serialization.converters import deserialize_lmdb_writes, serialize_lmdb_writes

TEXTS = [
    "John Smith (born 12 May 1950) is an American actor who starred in the film Jaws.",
    "The company was founded in 1998 by Larry Page and joined the index.",
    "The ship was laid down in 1910 and launched in 1911.",
    "Aspirin is used to treat pain.",
    "The pump is red.",
]


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def head_with(store, **cfg):
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "category_store_seed": False, **cfg})
    h.category_store = store
    return h


def values(h, nlp, texts, mode="ingest"):
    out = []
    for t in texts:
        ctx = {"spacy_doc": nlp.parse(t)}
        r = h.encode(t, ctx) if mode == "ingest" else h.encode_query(t, ctx)
        out.append([a.concept for a in r.addresses])
    return out


def test_seed_alone_on_cold_lmdb_matches_memory(tmp_path, nlp):
    kv = FakeKV()
    lmdb = LmdbAttributeCache(str(tmp_path / "aff"), map_size=64 * 1024 * 1024, readonly=False)
    st = CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb, kv_on_cache_miss=False)
    _seed_store(st)
    got = values(head_with(st), nlp, TEXTS, mode="query")
    mem = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    want = values(mem, nlp, TEXTS, mode="query")
    assert got == want
    # "The pump is red." has no value: copular values are off by default
    assert "establish" in got[1] and got[4] == ["__null__"]


def test_new_prefixes_ride_the_lmdb_relay(tmp_path, nlp):
    kv = FakeKV()
    path = str(tmp_path / "aff")
    writer_lmdb = LmdbAttributeCache(path, map_size=64 * 1024 * 1024, readonly=False)
    st = CouchbaseKVCategoryStore(kv, lmdb_cache=writer_lmdb, kv_on_cache_miss=False)
    _seed_store(st)
    st.drain_pending_lmdb_writes()                       # seed relay, not under test
    h = head_with(st)
    values(h, nlp, TEXTS * 3)                            # ingest: learn + auto-classify
    relayed = st.drain_pending_lmdb_writes()
    prefixes = {k.split("::", 1)[1].split(":", 1)[0] + ":" if ":" in k.split("::", 1)[1] else "category"
                for k, _ in relayed}
    assert {"p:", "b:", "ph:", "category"} <= prefixes
    # affordance-lmdb-sync: deserialize the NATS payload, merge, batch_put
    from nam_heads.impls.affordance.default.storage.merge import apply_sync_batch
    apply_sync_batch(writer_lmdb, None, deserialize_lmdb_writes(serialize_lmdb_writes(relayed)))
    # A fresh reader: empty L1, readonly LMDB, KV_ON_MISS=0, KV unreachable for misses
    reader = CouchbaseKVCategoryStore(FakeKV(), lmdb_cache=LmdbAttributeCache(path, readonly=True),
                                      kv_on_cache_miss=False)
    names = [k.split("::", 1)[1] for k, _ in relayed]
    docs = reader.get_docs_batch(names)
    assert all(docs[n] is not None for n in names)
    assert docs["p:establish"]["count"] >= 3
    assert docs["ph:lay:down"]["count"] >= 3
