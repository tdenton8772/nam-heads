"""Two pods, one session store: counters merge, seed reaches unseeded readers,
seed versions update seeded rows only."""
import json
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

pytest.importorskip("lmdb")

from src.storage.lmdb_node_cache import LmdbAttributeCache
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from nam_heads.impls.affordance.default.storage import merge as M
from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
from nam_heads.impls.affordance.default.storage.factory import _seed_store
from nam_heads.impls.affordance.default.storage.memory_store import MemoryCategoryStore
from nam_heads.impls._testing.fake_kv import FakeKV
from src.serialization.converters import deserialize_lmdb_writes, serialize_lmdb_writes

TEXTS = [
    "Smith founded the company and joined the board.",
    "The ship was laid down in 1910 and launched in 1911.",
    "Jones is an actor who starred in the film Jaws.",
    "The bridge was built in 1890 and rebuilt in 1950.",
    "She was born in Paris and studied at the Sorbonne.",
] * 4


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def pod(kv, lmdb, monkeypatch, name):
    st = CouchbaseKVCategoryStore(kv, lmdb_cache=lmdb, kv_on_cache_miss=False)
    _seed_store(st)
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "category_store_seed": False})
    h.category_store = st
    h._witness = name
    return h


def ingest(h, nlp, texts, monkeypatch, batch=2):
    monkeypatch.setenv("NAM_AFF_WITNESS", h._witness)
    for k in range(0, len(texts), batch):
        chunk = texts[k:k + batch]
        h.encode_batch_texts(chunk, [{"spacy_doc": nlp.parse(t)} for t in chunk])
    h.category_store.get_docs_batch([])          # flush deferred writes


def p_counts(kv):
    out = {}
    for k, v in kv._data.items():
        if k.startswith("vcat::p:"):
            out[k[6:]] = json.loads(v).get("count", 0)
    return out


# ---- 1. counters merge across pods ------------------------------------------

def test_two_pods_lose_no_counts(nlp, monkeypatch, tmp_path):
    """Two pods ingest halves of the same records into one KV; the merged
    counts equal a single pod ingesting everything."""
    single_kv = FakeKV()
    one = pod(single_kv, None, monkeypatch, "pod-0")
    ingest(one, nlp, TEXTS, monkeypatch)
    want = p_counts(single_kv)

    kv = FakeKV()
    a = pod(kv, None, monkeypatch, "pod-a")
    b = pod(kv, None, monkeypatch, "pod-b")
    # interleave batches: each pod's L1 is stale about the other's writes
    for k in range(0, len(TEXTS), 4):
        ingest(a, nlp, TEXTS[k:k + 2], monkeypatch)
        ingest(b, nlp, TEXTS[k + 2:k + 4], monkeypatch)
    got = p_counts(kv)
    assert got == want
    # 9 values x 4 copies ("is an actor" gives none: copular values are off)
    assert sum(got.values()) >= 36


def test_replayed_sync_messages_are_idempotent(tmp_path):
    lmdb = LmdbAttributeCache(str(tmp_path / "l"), map_size=16 * 1024 * 1024, readonly=False)
    kv = FakeKV()
    w = [("vcat::p:found", M.encode({"category": "predicate", "_g": {"count": {"pod-a": 3}}})),
         ("vcat::p:found", M.encode({"category": "predicate", "_g": {"count": {"pod-b": 2}}}))]
    msg = serialize_lmdb_writes(w)
    for _ in range(3):                                   # replayed three times
        M.apply_sync_batch(lmdb, kv, deserialize_lmdb_writes(msg))
    assert json.loads(lmdb.batch_get(["vcat::p:found"])["vcat::p:found"])["count"] == 5
    assert json.loads(kv.get("vcat::p:found")[0])["count"] == 5


def test_merge_is_commutative_and_keeps_seed_fields():
    seed = {"kind": "seed", "category": "verb_base", "seed_version": "v2", "_g": {}}
    learned = {"kind": "learned", "category": "verb_base", "_g": {"count": {"a": 4}}}
    ab, ba = M.merge_docs(seed, learned), M.merge_docs(learned, seed)
    assert ab == ba and ab["kind"] == "seed" and ab["count"] == 4


# ---- 2. seed reaches an unseeded reader through LMDB --------------------------

def test_cold_unseeded_reader_sees_seed_via_lmdb(nlp, tmp_path, monkeypatch):
    path = str(tmp_path / "aff")
    lmdb_w = LmdbAttributeCache(path, map_size=64 * 1024 * 1024, readonly=False)
    writer = CouchbaseKVCategoryStore(FakeKV(), lmdb_cache=lmdb_w, kv_on_cache_miss=False)
    _seed_store(writer)                                  # the ingest pod seeds
    relayed = writer.drain_pending_lmdb_writes()
    assert len(relayed) > 10000                          # every seed row relayed
    M.apply_sync_batch(lmdb_w, None, deserialize_lmdb_writes(serialize_lmdb_writes(relayed)))

    # nam-query-like reader: never seeds, KV_ON_MISS=0, empty KV, cold L1
    reader = CouchbaseKVCategoryStore(FakeKV(), lmdb_cache=LmdbAttributeCache(path, readonly=True),
                                      kv_on_cache_miss=False)
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "category_store_seed": False})
    h.category_store = reader
    t = "He joined the club, changed his name and founded a company."
    vals = [a.concept for a in h.encode_query(t, {"spacy_doc": nlp.parse(t)}).addresses]
    assert {"join", "change", "establish"} <= set(vals)
    assert not {"joine", "chang", "found"} & set(vals)


# ---- 5. seed versions ---------------------------------------------------------

def test_changed_seed_updates_seeded_rows_not_learned_ones(monkeypatch):
    from nam_heads.impls.affordance.default.storage import seed_version as SV
    kv = FakeKV()
    st = CouchbaseKVCategoryStore(kv)
    # an old seeded category row (unstamped, confidence 1.0), a learned one
    # (auto-classified, 0.5) and an old-version lexicon doc with counts
    kv.set("vcat::build", json.dumps({"category": "creative", "confidence": 1.0}).encode())
    kv.set("vcat::frobnicate", json.dumps({"category": "cognitive", "confidence": 0.5}).encode())
    kv.set("vcat::b:found", M.encode({"kind": "seed", "category": "verb_base", "seed_version": "old",
                                      "_g": {"count": {"pod-a": 9}}}))
    _seed_store(st)
    cur = SV.seed_version()
    build = json.loads(kv.get("vcat::build")[0])
    assert build["category"] == "physical" and build["seed_version"] == cur
    assert json.loads(kv.get("vcat::frobnicate")[0]) == {"category": "cognitive", "confidence": 0.5}
    found = json.loads(kv.get("vcat::b:found")[0])
    assert found["seed_version"] == cur and found["count"] == 9      # counters kept
    # same version again: nothing rewritten
    before = dict(kv._data)
    _seed_store(CouchbaseKVCategoryStore(kv))
    assert kv._data == before


def test_memory_store_seed_versions():
    st = MemoryCategoryStore()
    st.put_category_weighted("frobnicate", "cognitive", 0.5)
    assert st.seed_categories([("frobnicate", "physical", 1.0, None), ("build", "physical", 1.0, None)], "v1") == 1
    assert st.get_category_weighted("frobnicate") == ("cognitive", 0.5)
    assert st.seed_categories([("build", "creative", 1.0, None)], "v1") == 0
    assert st.seed_categories([("build", "creative", 1.0, None)], "v2") == 1
    assert st.get_category_weighted("build") == ("creative", 1.0)
