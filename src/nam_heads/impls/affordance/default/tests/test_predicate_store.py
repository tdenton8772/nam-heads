"""The predicate lexicon in the affordance store: seed, growth, use, agreement."""
import json
import os
import random

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.ontology import bindings as B
from nam_heads.impls.affordance.default import predicate_store as PS
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from nam_heads.impls.affordance.default.storage.memory_store import MemoryCategoryStore
from nam_heads.impls.affordance.default.storage.factory import _seed_store


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def head(store=None, **cfg):
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "emit_mode": "predicate", **cfg})
    if store is not None:
        h.category_store = store
    return h


def seeded():
    s = MemoryCategoryStore()
    _seed_store(s)
    return s


def run(h, nlp, texts, batch=4, mode="ingest"):
    docs = [nlp.parse(t) for t in texts]
    out = []
    for k in range(0, len(texts), batch):
        ctxs = [{"spacy_doc": d} for d in docs[k:k + batch]]
        if mode == "ingest":
            out += h.encode_batch_texts(texts[k:k + batch], ctxs)
        else:
            out += [h.encode_query(t, c) for t, c in zip(texts[k:k + batch], ctxs)]
    return [[a.concept for a in r.addresses] for r in out]


CORPUS = [
    "The ship was laid down in 1910 and launched in 1911.",
    "Her keel was laid down at Belfast.",
    "The cruiser was laid down in March.",
    "The battleship was laid down by Vickers.",
    "The destroyer was laid down in 1940.",
    "Smith founded the company and joined the board.",
    "The band teamed up with Jones.",
    "The firm was depicted in a film.",
]


def test_seed_goes_through_the_store_seeding_path():
    s = seeded()
    items = s.doc_items()
    assert items["b:found"]["kind"] == "seed"
    assert items["s:construct"]["target"] == "build"
    assert items["ph:know:as"]["value"] == "know_as"
    assert all("category" in d and "confidence" in d for d in items.values())
    n = len(items)
    _seed_store(s)                                # absent keys only
    assert len(s.doc_items()) == n


def test_seed_unfolds_a_fold_an_older_table_had(nlp):
    """A store seeded from an older table keeps that table's synonym docs
    (seeding writes the seed's keys only); the identity doc for each
    unfolded word replaces the old fold at the next seeding."""
    from nam_heads.impls.affordance.default import predicates as P
    assert "ensure" in P.seed_unfolded() and "ensure" not in P.seed_synonyms()
    s = MemoryCategoryStore()
    s.seed_docs([("s:ensure", {"kind": "seed", "seed_version": "old", "category": "synonym",
                               "confidence": 1.0, "target": "assure"})])
    h = head(s)
    t = "The Centre and the Society ensure that generations remember the war."
    assert "assure" in run(h, nlp, [t], mode="query")[0]
    _seed_store(s)
    assert s.doc_items()["s:ensure"]["target"] == "ensure"
    v = run(head(s), nlp, [t], mode="query")[0]
    assert "ensure" in v and "assure" not in v


def test_one_store_read_per_batch(nlp):
    s = seeded()
    h = head(s)
    s.doc_batches = 0
    run(h, nlp, CORPUS, batch=4)
    assert s.doc_batches == 2        # 8 records / batch 4, no extra reads here


def test_ingest_grows_the_store_with_value_docs(nlp):
    s = seeded()
    before = len(s.doc_items())
    run(head(s), nlp, CORPUS)
    items = s.doc_items()
    assert len(items) > before
    d = items["p:establish"]                       # "founded" -> found -> establish
    assert d["count"] == 1 and d["kind"] == "verb" and d["lemma"] == "found"
    assert items["b:found"]["kind"] == "seed" and items["b:found"]["count"] >= 1
    # legacy category growth: the auto-classified category of a new value
    assert s.get_category_weighted("establish") is not None


def test_query_reads_but_never_writes(nlp):
    s = seeded()
    snap = PS.store_fingerprint(s)
    run(head(s), nlp, CORPUS, mode="query")
    assert PS.store_fingerprint(s) == snap


def test_seed_doc_survives_a_synonym_target_write(nlp):
    """depict -> portray touches b:portray, which the batch did not prefetch."""
    s = seeded()
    run(head(s), nlp, ["The firm was depicted in a film."])
    assert s.doc_items()["b:portray"]["kind"] == "seed"
    vals = run(head(s), nlp, ["She is known for portraying Madison."])
    assert "portray" in vals[0]


def test_learned_phrasal_used_past_threshold_and_agrees_at_query(nlp):
    s = seeded()
    h = head(s)
    first = run(h, nlp, CORPUS[:5], batch=1)
    assert "lay_down" not in first[0]               # not learned yet
    assert s.doc_items()["ph:lay:down"]["count"] >= PS.LEARN_PHRASAL_MIN
    later = run(h, nlp, ["The frigate was laid down in 1801."])
    assert "lay_down" in later[0]
    q = run(h, nlp, ["The frigate was laid down in 1801."], mode="query")
    assert q == later                               # query reads the same store


def test_use_learned_off_ignores_learned_entries(nlp):
    s = seeded()
    run(head(s), nlp, CORPUS[:5], batch=1)
    off = run(head(s, use_learned_predicates=False), nlp, ["The frigate was laid down in 1801."], mode="query")
    assert "lay_down" not in off[0]


def test_learned_base_needs_bare_and_inflected_forms():
    s = seeded()
    lex = PS.StoreLexicon(s)
    lex.docs = {}
    lex.note_verb("vertebrate", "VB", "vertebrate", False, "")   # tagger noun-as-VB
    lex.commit()
    lex.docs = s.doc_items()
    assert not lex.is_base("vertebrate")
    lex.note_verb("livestream", "VB", "livestream", False, "")
    lex.note_verb("livestreamed", "VBD", "livestream", False, "")
    lex.commit()
    lex.docs = s.doc_items()
    assert lex.is_base("livestream")


def test_two_runs_converge_to_the_same_store(nlp):
    a, b = seeded(), seeded()
    run(head(a), nlp, CORPUS * 2, batch=3)
    run(head(b), nlp, CORPUS * 2, batch=3)
    assert PS.store_fingerprint(a) == PS.store_fingerprint(b)


def test_record_order_learns_the_same_usable_entries(nlp):
    def usable(store):
        lex = PS.StoreLexicon(store, learn=False)
        lex.docs = store.doc_items()
        return sorted(n for n, d in lex.docs.items()
                      if (n.startswith("ph:") and d.get("kind") == "learned" and lex.phrasal(*n[3:].split(":", 1)))
                      or (n.startswith("b:") and d.get("kind") == "learned" and lex.is_base(n[2:])))
    a, b = seeded(), seeded()
    texts = CORPUS * 2
    shuffled = texts[:]
    random.Random(3).shuffle(shuffled)
    run(head(a), nlp, texts, batch=2)
    run(head(b), nlp, shuffled, batch=5)
    assert usable(a) == usable(b) and "ph:lay:down" in usable(a)


def test_ont_freq_reaches_metadata_past_threshold(nlp, monkeypatch):
    monkeypatch.setattr(PS, "ONT_FREQ_MIN", 2)
    s = seeded()
    from src.encoders.registry import EncodingResult, SemanticAddress
    h = head(s)
    t = "Smith wrote the book."
    d = nlp.parse(t)
    ont = EncodingResult("ontology", [SemanticAddress("ontology", B.PERSON, 0.9)], {})
    for _ in range(3):
        r = h.encode_batch_texts([t], [{"spacy_doc": d, "ontology_result": ont}])[0]
    meta = {a.concept: a.metadata for a in r.addresses}["write"]
    assert meta["ont_freq"] == {"cco:ont00001262": 2}


def test_store_without_docs_falls_back_to_seed_lexicon(nlp):
    from nam_heads.impls.affordance.default.storage.base import CategoryStore

    class Legacy(MemoryCategoryStore):        # a backend without the doc API
        get_docs_batch = CategoryStore.get_docs_batch
    h = head(Legacy())
    t = "The company was founded in 1998."
    r = h.encode_batch_texts([t], [{"spacy_doc": nlp.parse(t)}])[0]
    assert r.metadata["lexicon"] == "seed"
    assert [a.concept for a in r.addresses] == ["establish"]
