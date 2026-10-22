"""The attribute store grows from predications, seeds the person-noun filter,
relays every write to LMDB, is read by queries, and converges."""
import json
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from nam_heads.impls.attribute.default import learning as L
from nam_heads.impls.attribute.default.head import AttributeEncoder
from nam_heads.impls.attribute.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
from nam_heads.impls.attribute.default.storage.factory import _seed_store
from nam_heads.impls.attribute.default.storage.memory_store import MemoryCategoryStore
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.ontology import bindings as B


class FakeKV:
    def __init__(self):
        self.data = {}

    def get(self, k):
        return self.data.get(k), None

    def set(self, k, v):
        self.data[k] = v

    def pipeline_get(self, keys):
        return {k: (self.data.get(k), None) for k in keys}

    def pipeline_set(self, items):
        for k, v in items:
            self.data[k] = v


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def _ctx(nlp, text, sentence_ontology=B.PERSON):
    doc = nlp.parse(text)
    n = len(list(doc.sents)) if doc.sents else 1
    ont = EncodingResult("ontology", [SemanticAddress("ontology", sentence_ontology)],
                         {"sentence_ontologies": {i: sentence_ontology for i in range(n)}})
    return {"spacy_doc": doc, "ontology_result": ont}


def _encoder(store=None):
    # Learning shows in extraction through EVIDENCE_EXCLUSIONS (an unknown
    # predicate noun is not admitted for a Person until it is learned as a
    # role), which is off by default -- these tests turn it on.
    enc = AttributeEncoder(head_type="attribute",
                           config={"category_store_backend": "memory", "evidence_exclusions": True})
    if store is not None:
        enc.category_store = store
    return enc


def _ingest(nlp, enc, texts, ont=B.PERSON):
    return enc.encode_batch_texts(texts, [_ctx(nlp, t, ont) for t in texts])


def _concepts(res):
    return {a.concept for a in res.addresses}


VLOGGERS = ["John Smith (born 1990) is an English vlogger.",
            "Mary Jones (1950 - 2001) was a Canadian vlogger and actress.",
            "Anna Lee is a Korean vlogger who lives in Seoul.",
            "Tom Ford (born 1985) is an American vlogger."]


# ------------------------------------------------------------------ seed

def test_person_nouns_are_seeded_through_the_store():
    store = MemoryCategoryStore()
    _seed_store(store)
    assert L.is_role(store.get_doc("actor"))
    assert store.get_doc("actor")["role_source"] == "seed"
    # an adjective seed keeps its category and gains the role flag
    off = store.get_doc("official")
    assert off["category"] == "institutional" and L.is_role(off)
    assert not L.is_role(store.get_doc("large"))
    assert store.get_doc("vlogger") is None


def test_kv_seed_is_idempotent_and_merges_extras():
    kv = FakeKV()
    kv.data["acat::actor"] = json.dumps({"category": "physical", "confidence": 1.0}).encode()
    store = CouchbaseKVCategoryStore(kv)
    _seed_store(store)
    actor = json.loads(kv.data["acat::actor"])
    assert actor["category"] == "physical" and actor["role"] is True   # merged, not replaced
    snapshot = dict(kv.data)
    store2 = CouchbaseKVCategoryStore(kv)
    _seed_store(store2)
    assert kv.data == snapshot


# ------------------------------------------------------------------ writes

def test_every_settled_value_is_written_with_kind_class_and_relayed(nlp):
    kv = FakeKV()
    store = CouchbaseKVCategoryStore(kv)
    _seed_store(store)
    store.drain_pending_lmdb_writes()
    enc = _encoder(store)
    text = "Kenia Sinclair is a Jamaican athlete. She won a large medal."
    res = _ingest(nlp, enc, [text])[0]
    settled = _concepts(res) - {"__null__"}
    assert {"jamaican", "athlete", "large"} <= settled
    relayed = {k: json.loads(v) for k, v in store.drain_pending_lmdb_writes()}
    for v in settled:
        row = relayed[f"acat::{v}"]
        assert row["kinds"] and row["category"]
        assert row["ontologies"] == [B.PERSON] and row["classes"] == [B.PERSON]
    assert "noun" in relayed["acat::athlete"]["kinds"]
    assert "norp" in relayed["acat::jamaican"]["kinds"]
    # KV gets the same rows on the next batch read
    store.get_categories_weighted_batch([("x", None)])
    assert json.loads(kv.data["acat::athlete"])["classes"] == [B.PERSON]


def test_unchanged_rows_are_not_rewritten(nlp):
    kv = FakeKV()
    store = CouchbaseKVCategoryStore(kv)
    enc = _encoder(store)
    _ingest(nlp, enc, ["Kenia Sinclair is a Jamaican athlete."])
    assert store.drain_pending_lmdb_writes()
    _ingest(nlp, enc, ["Kenia Sinclair is a Jamaican athlete."])
    assert store.drain_pending_lmdb_writes() == []


def test_queries_never_write(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _encoder(store)
    before = store.count()
    enc.encode_query("German vlogger", _ctx(nlp, "German vlogger"))
    assert store.count() == before


# ------------------------------------------------------------------ learning

def test_role_is_learned_at_the_support_threshold_and_improves_extraction(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _encoder(store)
    # Under a Person, an unknown predicate noun is not admitted.
    first = _ingest(nlp, enc, VLOGGERS[:1])[0]
    assert "vlogger" not in _concepts(first)
    assert not L.is_role(store.get_doc("vlogger"))
    _ingest(nlp, enc, VLOGGERS[1:2])
    assert len(store.get_doc("vlogger")["person_support"]) == 2
    assert not L.is_role(store.get_doc("vlogger"))
    _ingest(nlp, enc, VLOGGERS[2:3])           # third distinct record
    row = store.get_doc("vlogger")
    assert L.is_role(row) and row["role_source"] == "learned" and row["category"] == "role"
    # now it is a role noun: admitted for a Person, and accepted in a title
    later = _ingest(nlp, enc, VLOGGERS[3:4] + ["The album features the vlogger Peter Brown."])
    assert "vlogger" in _concepts(later[0])
    assert "vlogger" in _concepts(later[1])


def test_repeating_one_record_does_not_count_as_support(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _encoder(store)
    for _ in range(5):
        _ingest(nlp, enc, VLOGGERS[:1])
    assert len(store.get_doc("vlogger")["person_support"]) == 1


def test_support_needs_a_surface_person_cue(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _encoder(store)
    _ingest(nlp, enc, ["John Smith is an English vlogger.", "Anna Lee is a Korean vlogger.",
                       "Tom Ford is an American vlogger."])
    assert not store.get_doc("vlogger").get("person_support")


def test_support_needs_a_person_sentence(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _encoder(store)
    _ingest(nlp, enc, VLOGGERS[:3], ont=B.MATERIAL_ARTIFACT)
    assert not L.is_role(store.get_doc("vlogger"))


def test_query_reads_the_same_store_and_agrees_with_ingest(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    ingest, query = _encoder(store), _encoder(store)
    q = lambda: _concepts(query.encode_query("Korean vlogger", _ctx(nlp, "Korean vlogger")))
    assert "vlogger" not in q()            # unknown noun, rejected under a Person
    _ingest(nlp, ingest, VLOGGERS[:3])
    ing = _concepts(_ingest(nlp, ingest, ["Anna Lee is a Korean vlogger."])[0])
    assert {"korean", "vlogger"} <= ing
    assert q() == ing


# ------------------------------------------------------------------ convergence

def _dump(store):
    lemmas = {k[0] for k in store._store}
    return {lemma: store.get_doc(lemma) for lemma in lemmas}


def test_two_runs_converge_regardless_of_order_and_batching(nlp):
    texts = VLOGGERS + ["The album features the vlogger Peter Brown.",
                        "Kenia Sinclair is a Jamaican athlete.",
                        "Anacondas are a group of large snakes."]
    a, b = MemoryCategoryStore(), MemoryCategoryStore()
    _seed_store(a)
    _seed_store(b)
    ea, eb = _encoder(a), _encoder(b)
    for _ in range(2):
        _ingest(nlp, ea, texts)                         # one batch, forward
    for _ in range(2):
        for t in reversed(texts):                       # one record at a time, reversed
            _ingest(nlp, eb, [t])
    assert _dump(a) == _dump(b)
    snap = _dump(a)
    _ingest(nlp, ea, texts)                             # fixed point
    assert _dump(a) == snap


def test_merge_is_commutative_and_idempotent():
    o1, o2 = L.Observation(), L.Observation()
    o1.kinds, o1.ontologies, o1.classes, o1.support = {"noun"}, {"person"}, {B.PERSON}, {"b", "d"}
    o2.kinds, o2.ontologies, o2.classes, o2.support = {"adj"}, {"object"}, {B.MATERIAL_ARTIFACT}, {"a", "c"}
    ab = L.merge(L.merge(None, o1), o2)
    ba = L.merge(L.merge(None, o2), o1)
    assert ab == ba
    assert L.merge(ab, o1) == ab
    assert ab["person_support"] == ["a", "b", "c"] and ab["role"] is True


# ------------------------------------------------- LMDB relay, production read path

def _sync(writer_lmdb, writes):
    """What run_attribute_lmdb_sync does with one relayed batch
    (nats_worker.py): the NATS protobuf hop, then batch_put."""
    from src.serialization.converters import serialize_lmdb_writes, deserialize_lmdb_writes
    writes = deserialize_lmdb_writes(serialize_lmdb_writes(writes))   # the NATS hop
    writer_lmdb.batch_put(writes)
    return writes


def _reader(path, kv):
    """A head/query process as the chart runs it: cold L1, read-only LMDB,
    NAM_ATTRIBUTE_KV_ON_MISS=0 (KV never consulted on a miss)."""
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    return CouchbaseKVCategoryStore(kv, lmdb_cache=LmdbAttributeCache(path, readonly=True),
                                    kv_on_cache_miss=False)


def test_seed_and_learned_rows_reach_kv_on_miss_off_readers_via_the_relay(nlp, tmp_path):
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    path = str(tmp_path / "attr-lmdb")
    sync_lmdb = LmdbAttributeCache(path, readonly=False)          # attribute-lmdb-sync
    kv = FakeKV()
    writer = CouchbaseKVCategoryStore(kv, lmdb_cache=LmdbAttributeCache(path, readonly=True),
                                      kv_on_cache_miss=False)
    _seed_store(writer)
    # nothing relayed yet: a KV_ON_MISS=0 reader cannot see the seed
    assert not L.is_role(_reader(path, kv).get_docs_batch(["actor"])["actor"])
    _sync(sync_lmdb, writer.drain_pending_lmdb_writes())
    assert L.is_role(_reader(path, kv).get_docs_batch(["actor"])["actor"])

    # learning on the writer reaches a fresh reader only through the relay
    enc = _encoder(writer)
    _ingest(nlp, enc, VLOGGERS[:3])
    assert not L.is_role(_reader(path, kv).get_docs_batch(["vlogger"])["vlogger"])
    _sync(sync_lmdb, writer.drain_pending_lmdb_writes())
    assert L.is_role(_reader(path, kv).get_docs_batch(["vlogger"])["vlogger"])

    # a second head started on a warm KV but a cold node re-relays the seed
    path2 = str(tmp_path / "other-node")
    sync2 = LmdbAttributeCache(path2, readonly=False)
    head2 = CouchbaseKVCategoryStore(kv, lmdb_cache=LmdbAttributeCache(path2, readonly=True),
                                     kv_on_cache_miss=False)
    _seed_store(head2)
    assert L.is_role(head2.get_docs_batch(["actor"])["actor"])     # its own L1
    _sync(sync2, head2.drain_pending_lmdb_writes())
    assert L.is_role(_reader(path2, kv).get_docs_batch(["actor"])["actor"])


def test_relayed_seed_alone_gives_correct_output_on_a_kv_on_miss_off_reader(nlp, tmp_path):
    """Production read path: cold L1, LMDB filled only by the relayed seed,
    NAM_ATTRIBUTE_KV_ON_MISS=0. The seed alone yields the person-noun
    values; without it the same reader loses them."""
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    path = str(tmp_path / "attr-lmdb")
    sync_lmdb = LmdbAttributeCache(path, readonly=False)
    kv = FakeKV()
    seeder = CouchbaseKVCategoryStore(kv, lmdb_cache=LmdbAttributeCache(path, readonly=True),
                                      kv_on_cache_miss=False)
    _seed_store(seeder)
    _sync(sync_lmdb, seeder.drain_pending_lmdb_writes())
    text = "The album features the guitarist Andy Lee. Mike Wilks is an English illustrator."
    got = _concepts(_encoder(_reader(path, kv)).encode_query(text, _ctx(nlp, text)))
    assert {"guitarist", "english", "illustrator"} <= got   # illustrator: copular, seed role noun
    empty = str(tmp_path / "empty-lmdb")
    LmdbAttributeCache(empty, readonly=False)
    bare = _concepts(_encoder(_reader(empty, kv)).encode_query(text, _ctx(nlp, text)))
    # without the seed, "illustrator" is an unknown predicate noun, excluded
    # for a Person ("guitarist" still comes from the legacy compound source)
    assert "illustrator" not in bare
