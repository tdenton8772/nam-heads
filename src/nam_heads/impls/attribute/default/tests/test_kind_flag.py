"""The store's kind flag: seeded False for abstract / relational nouns,
learned True from named-subject predications, read by extraction through
kind_status; and the record title reaching the head through ctx["title"]."""
import json
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from nam_heads.impls.attribute.default import learning as L
from nam_heads.impls.attribute.default.head import AttributeEncoder
from nam_heads.impls.attribute.default.predication import abstract_nouns, person_nouns
from nam_heads.impls.attribute.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
from nam_heads.impls.attribute.default.storage.factory import _seed_store
from nam_heads.impls.attribute.default.storage.memory_store import MemoryCategoryStore
from nam_heads.impls.attribute.default.tests.test_store_learning import FakeKV


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def _enc(store=None):
    enc = AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})
    if store is not None:
        enc.category_store = store
    return enc


def _ingest(nlp, enc, texts, titles=None):
    titles = titles or [""] * len(texts)
    ctxs = [dict({"spacy_doc": nlp.parse(t)}, **({"title": ti} if ti else {})) for t, ti in zip(texts, titles)]
    return enc.encode_batch_texts(texts, ctxs)


def _anchors(nlp, text, res, concept):
    doc = nlp.parse(text)
    for a in res.addresses:
        if a.concept == concept:
            return sorted(" ".join(doc[i].text for i in range(s, e)) for s, e in a.metadata.get("anchors", []))
    return None


# ------------------------------------------------------------------ seed

def test_seed_file_is_abstract_nouns_only():
    seed = abstract_nouns()
    assert {"story", "impact", "legacy", "heart", "capital"} <= seed
    assert not seed & person_nouns()           # one KV seed batch never writes a key twice
    assert all(w.isalpha() and w == w.lower() for w in seed)


def test_abstract_nouns_are_seeded_kind_false():
    store = MemoryCategoryStore()
    _seed_store(store)
    row = store.get_doc("story")
    assert row["kind"] is False and row["kind_source"] == "seed"
    assert L.kind_status(row) is False
    assert L.kind_status(store.get_doc("actor")) is True        # a role noun is a kind
    assert L.kind_status(store.get_doc("airfield")) is None     # unknown


def test_kv_seed_carries_the_kind_flag_and_is_idempotent():
    kv = FakeKV()
    store = CouchbaseKVCategoryStore(kv)
    _seed_store(store)
    row = json.loads(kv.data["acat::legacy"])
    assert row["kind"] is False and row["kind_source"] == "seed"
    snap = dict(kv.data)
    _seed_store(CouchbaseKVCategoryStore(kv))
    assert kv.data == snap


# ------------------------------------------------------------------ learning

def test_kind_evidence_needs_a_named_subject_and_no_relational_complement():
    assert L.kind_evidence("copular", "head", "noun", named_subject=True)
    assert L.kind_evidence("appositive", "head", "noun", named_subject=True)
    assert not L.kind_evidence("copular", "head", "noun", named_subject=False)
    assert not L.kind_evidence("pronoun", "head", "noun", named_subject=True)
    assert not L.kind_evidence("copular", "mod", "noun", named_subject=True)
    assert not L.kind_evidence("copular", "head", "adj", named_subject=True)
    assert not L.kind_evidence("copular", "head", "noun", True, of_complement=True)   # "the capital of Y"
    assert not L.kind_evidence("copular", "head", "noun", True, relational=True)      # "a testament to X"


def test_kind_is_learned_at_the_threshold_and_overrides_the_seed():
    def obs(*fps):
        o = L.Observation()
        o.kinds = {"noun"}
        o.kind_support = set(fps)
        return o
    seeded = {"category": "uncategorized", "confidence": 0.5, "kind": False, "kind_source": "seed"}
    row = L.merge(seeded, obs("a", "b"))
    assert row["kind"] is False and row["kind_support"] == ["a", "b"]
    row = L.merge(row, obs("c"))
    assert row["kind"] is True and row["kind_source"] == "learned"
    # commutative, idempotent, bounded
    ab = L.merge(L.merge(None, obs("x", "y")), obs("z", "w"))
    ba = L.merge(L.merge(None, obs("z", "w")), obs("x", "y"))
    assert ab == ba and L.merge(ab, obs("x")) == ab
    assert ab["kind_support"] == ["w", "x", "y"]


VILLAGES = ["Kucura is a village in Serbia.", "Bir Ahmad is a village in Yemen.",
            "Vlierzele is a village in Belgium.", "Oordegem is a village in Belgium."]


def test_ingest_learns_kind_and_definite_anaphors_follow(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _enc(store)
    text = "Vrbas. The village hosts a fair."
    # cold: "the village" is not tied to Vrbas by the text, and the store
    # does not know "village" as a kind -> not a value of Vrbas
    cold = _ingest(nlp, enc, [text], ["Vrbas"])[0]
    assert "Vrbas" not in (_anchors(nlp, text, cold, "village") or [])
    _ingest(nlp, enc, VILLAGES[:3])
    row = store.get_doc("village")
    assert row["kind"] is True and row["kind_source"] == "learned" and len(row["kind_support"]) == 3
    warm = _ingest(nlp, enc, [text], ["Vrbas"])[0]
    assert "Vrbas" in _anchors(nlp, text, warm, "village")


def test_queries_read_the_kind_flag_but_never_write(nlp):
    store = MemoryCategoryStore()
    _seed_store(store)
    enc = _enc(store)
    before = {k: store.get_doc(k[0]) for k in list(store._store)}
    enc.encode_query("village", {"spacy_doc": nlp.parse("village")})
    assert {k: store.get_doc(k[0]) for k in list(store._store)} == before


# ------------------------------------------------------------------ ctx["title"]

def test_title_in_context_resolves_pronouns_in_both_encode_paths(nlp):
    text = "Svetlana Ishmouratova. You might wonder why. She is a Russian biathlete."
    ctx = {"spacy_doc": nlp.parse(text), "title": "Svetlana Ishmouratova"}
    enc = _enc()
    batch = enc.encode_batch_texts([text], [ctx])[0]
    single = enc.encode(text, ctx)
    assert "Svetlana Ishmouratova" in _anchors(nlp, text, batch, "russian")
    assert {a.concept: a.metadata.get("anchors") for a in batch.addresses} == \
        {a.concept: a.metadata.get("anchors") for a in single.addresses}
    untitled = enc.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    got = _anchors(nlp, text, untitled, "russian")
    assert "She" in got and "Svetlana Ishmouratova" not in got


def test_title_qualifier_is_read_from_the_store(nlp):
    """The qualifier ("company") is a store key of the batch read; the
    person guard asks the store whether it names a kind of person."""
    text = "Yves Rocher (company). He was a French botanist."
    enc = _enc()
    res = _ingest(nlp, enc, [text], ["Yves Rocher (company)"])[0]
    assert _anchors(nlp, text, res, "botanist") == ["He"]
    res = _ingest(nlp, enc, ["Yves Rocher (botanist). He was a French botanist."], ["Yves Rocher (botanist)"])[0]
    assert _anchors(nlp, "Yves Rocher (botanist). He was a French botanist.", res, "botanist") == ["Yves Rocher"]


def test_wordlist_loads_from_a_file(tmp_path):
    """A configured wordlist path is read (pathlib was not imported, so the
    loader's NameError was swallowed and a file wordlist never loaded)."""
    f = tmp_path / "words.txt"
    f.write_text("Crimson\nteal\n")
    enc = AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory", "wordlist": str(f)})
    assert enc.wordlist == {"crimson", "teal"}
