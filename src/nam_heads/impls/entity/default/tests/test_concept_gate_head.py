"""The entity head applies the concept gate: a lower-case phrase is an
entity only as a seed or learned concept, and a phrase that is not gets an
address (meaningful False) but no store entry."""

import pytest

from src.nlp import get_nlp_service
from ..head import EntityEncoder, _hash_entity_id
from ..storage.memory_store import MemoryEntityStore
from .. import concepts as C


@pytest.fixture(scope="module")
def nlp():
    return get_nlp_service(backend="rule")


def _encoder(store, k=3):
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.2
    enc.enable_fuzzy = True
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = store
    enc._concepts = C.ConceptLearner(k=k)
    return enc


def _batch(enc, nlp, texts):
    return enc.encode_batch_texts(texts, [{"spacy_doc": nlp.parse(t)} for t in texts])


def _addr(result, surface):
    hits = [a for a in result.addresses if a.metadata.get("surface_form") == surface]
    assert hits, [a.metadata.get("surface_form") for a in result.addresses]
    return hits[0]


DESCRIPTION = "Angola has powerful neighbors in the region."
TERM = "The basin holds sedimentary rock from the Jurassic."
LEARN = ["Engineers traced cancellation errors in the solver.",
         "Students learn about cancellation errors in numerics.",
         "The library avoids cancellation errors in sums."]


def test_fixture_phrases():
    assert not C.in_seed("powerful neighbors")
    assert C.in_seed("sedimentary rock")
    assert not C.in_seed("cancellation errors")


def test_a_description_has_an_address_but_no_entity(nlp):
    store = MemoryEntityStore()
    res = _batch(_encoder(store), nlp, [DESCRIPTION])[0]
    a = _addr(res, "powerful neighbors")
    assert a.metadata["meaningful"] is False
    assert a.concept == _hash_entity_id("", "powerful_neighbor")
    assert store.get_entity(a.concept) is None
    # The name in the same record is unaffected.
    assert _addr(res, "Angola").metadata["meaningful"] is True


def test_a_seed_term_is_an_entity(nlp):
    store = MemoryEntityStore()
    res = _batch(_encoder(store), nlp, [TERM])[0]
    a = _addr(res, "sedimentary rock")
    assert a.metadata["meaningful"] is True
    assert store.get_entity(a.concept) is not None
    assert not any(k.startswith("cpt::") for k in store.raw)       # a seed term is not counted


def test_a_term_is_learned_on_its_kth_record(nlp):
    store = MemoryEntityStore()
    enc = _encoder(store, k=3)
    first = _batch(enc, nlp, LEARN[:1])[0]
    second = _batch(enc, nlp, LEARN[1:2])[0]
    for res in (first, second):
        a = _addr(res, "cancellation errors")
        assert a.metadata["meaningful"] is False
        assert store.get_entity(a.concept) is None
    third = _addr(_batch(enc, nlp, LEARN[2:])[0], "cancellation errors")
    assert third.metadata["meaningful"] is True
    assert store.get_entity(third.concept) is not None


def test_the_same_record_never_learns(nlp):
    store = MemoryEntityStore()
    enc = _encoder(store, k=2)
    for _ in range(3):
        a = _addr(_batch(enc, nlp, LEARN[:1])[0], "cancellation errors")
        assert a.metadata["meaningful"] is False


def test_a_free_modifier_is_never_counted(nlp):
    assert "powerful" in C.free_modifiers()
    store = MemoryEntityStore()
    enc = _encoder(store, k=1)
    a = _addr(_batch(enc, nlp, [DESCRIPTION])[0], "powerful neighbors")
    assert a.metadata["meaningful"] is False
    assert not any(k.startswith("cpt::") for k in store.raw)


def test_single_text_ingest_path_agrees(nlp):
    store = MemoryEntityStore()
    enc = _encoder(store)
    res = enc._encode_internal(DESCRIPTION, mode="ingest", context={"spacy_doc": nlp.parse(DESCRIPTION)})
    a = _addr(res, "powerful neighbors")
    assert a.metadata["meaningful"] is False
    assert store.get_entity(a.concept) is None


def test_a_query_neither_reads_nor_counts(nlp):
    class Watching(MemoryEntityStore):
        reads = 0

        def get_raw_batch(self, keys):
            Watching.reads += 1
            return super().get_raw_batch(keys)
    store = Watching()
    _encoder(store, k=1).encode_query("cancellation errors in numerics")
    assert Watching.reads == 0 and store.raw == {}


def test_gate_off_restores_the_word_rules(nlp, monkeypatch):
    monkeypatch.setenv("NAM_ENTITY_CONCEPT_GATE", "0")
    store = MemoryEntityStore()
    a = _addr(_batch(_encoder(store), nlp, [DESCRIPTION])[0], "powerful neighbors")
    assert a.metadata["meaningful"] is True
    assert store.get_entity(a.concept) is not None
