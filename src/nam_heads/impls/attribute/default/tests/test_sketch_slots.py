"""Attribute head on the resolved sentence sketch: every value carries the
chains of the subjects it was predicated of (NLP2's chain of the subject's
own mention); an unresolved pronoun falls back to the head's main-subject
resolution; what the store learns does not depend on the sketch."""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")
# Memory stores, set before any head module reads its config from the
# environment (a head config imported without them points at the session KV).
for _k in ("NAM_ATTRIBUTE_STORE_BACKEND", "NAM_AFFORDANCE_STORE_BACKEND", "NAM_CONTEXT_STORE_BACKEND",
           "NAM_ENTITY_DB_BACKEND", "NAM_ENTITY_STORE_BACKEND"):
    os.environ.setdefault(_k, "memory")
os.environ.setdefault("NAM_KV_ON_MISS", "0")

from nam_heads.impls.attribute.default.head import AttributeEncoder
from src.nlp.sketch import EntityRef, resolve_sketch
from src.nlp.sketch.model import RESOLVED_CONTRACT


@pytest.fixture(scope="module")
def nlp():
    from nam_heads.impls.nlp.rule import RuleNLPService
    return RuleNLPService()


def head():
    return AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})


def encode(nlp, title, body, surfaces=(), sketch=True, h=None):
    text = f"{title}. {body}" if title else body
    doc = nlp.parse(text, title=title)
    if sketch:
        doc.sketch = resolve_sketch(doc.skeleton, [EntityRef(f"e:{s.lower()}", s) for s in surfaces])
    ctx = {"spacy_doc": doc, **({"title": title} if title else {})}
    return (h or head()).encode_batch_texts([text], [ctx])[0], doc


def values(res):
    return {a.concept: a.metadata for a in res.addresses if a.concept != "__null__"}


def chain(doc, name):
    return next(c.chain_id for c in doc.sketch.chains if c.canonical == name)


def test_copular_complement_is_the_subjects(nlp):
    res, doc = encode(nlp, "Richard Feynman", "Richard Feynman was an American physicist.",
                      ["Richard Feynman"])
    assert res.metadata["sketch"] == RESOLVED_CONTRACT
    v = values(res)
    assert v["physicist"]["chains"] == [chain(doc, "Richard Feynman")]


def test_pronoun_subject_takes_its_nlp2_chain(nlp):
    res, doc = encode(nlp, "Richard Feynman",
                      "Richard Feynman worked at Caltech. He was a gifted teacher.",
                      ["Richard Feynman", "Caltech"])
    assert values(res)["teacher"]["chains"] == [chain(doc, "Richard Feynman")]


def test_unresolved_pronoun_falls_back_to_the_main_subject(nlp):
    """NLP2 leaves this "he" unresolved (a nearer subject outside every chain
    blocks it); the head's title-subject resolution binds it."""
    res, doc = encode(nlp, "António de Sousa Franco",
                      "If you're wondering who António de Sousa Franco was, he was a notable "
                      "Portuguese economist and politician born in 1942.",
                      ["António de Sousa Franco"])
    he = next(t.i for t in doc.tokens if t.text == "he")
    assert next(m for m in doc.sketch.mentions if m.start == he).chain == -1
    v = values(res)
    assert v["economist"]["chains"] == [chain(doc, "António de Sousa Franco")]


def test_value_whose_subject_has_no_chain_is_unbound(nlp):
    """"The old capital" is no mention NLP2 chains, and the head does not
    resolve it either (a definite the text does not tie to Utah): its values
    reach no bundle."""
    res, doc = encode(nlp, "", "Utah is a state. The old capital was a small fort near the lake.", ["Utah"])
    v = values(res)
    assert v["state"]["chains"] == [chain(doc, "Utah")]
    assert v["fort"]["chains"] == []


def test_learning_does_not_depend_on_the_sketch(nlp):
    title = "Richard Feynman"
    body = "Richard Feynman was an American physicist. He was a gifted teacher. The scientist died in 1988."
    rows = []
    for sk in (False, True):
        h = head()
        encode(nlp, title, body, ["Richard Feynman"], sketch=sk, h=h)
        st = h.category_store                     # MemoryCategoryStore
        rows.append((dict(st._store), {k: dict(v) for k, v in st._extras.items()}))
    assert rows[0] == rows[1]
    assert rows[0][1]                             # it did learn something


def test_without_sketch_no_chains(nlp, monkeypatch):
    plain, _d = encode(nlp, "Richard Feynman", "Richard Feynman was an American physicist.", sketch=False)
    assert "sketch" not in plain.metadata
    assert all("chains" not in m for m in values(plain).values())
    monkeypatch.setenv("NAM_SKETCH_SLOTS", "0")
    off, _d = encode(nlp, "Richard Feynman", "Richard Feynman was an American physicist.", ["Richard Feynman"])
    assert [(a.concept, a.metadata) for a in off.addresses] == [(a.concept, a.metadata) for a in plain.addresses]
