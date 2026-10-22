"""Affordance head on the resolved sentence sketch: each value is a clause's
action, linked to the chains of its actor and patient; a predicate no clause
has as its action is not emitted (empty over wrong) but still learned from;
without a sketch the head is unchanged."""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")
# Memory stores, set before any head module reads its config from the
# environment (a head config imported without them points at the session KV).
for _k in ("NAM_ATTRIBUTE_STORE_BACKEND", "NAM_AFFORDANCE_STORE_BACKEND", "NAM_CONTEXT_STORE_BACKEND",
           "NAM_ENTITY_DB_BACKEND", "NAM_ENTITY_STORE_BACKEND"):
    os.environ.setdefault(_k, "memory")
os.environ.setdefault("NAM_KV_ON_MISS", "0")

from nam_heads.impls.affordance.default.head import AffordanceEncoder
from src.nlp.sketch import EntityRef, resolve_sketch
from src.nlp.sketch.model import RESOLVED_CONTRACT


@pytest.fixture(scope="module")
def nlp():
    from nam_heads.impls.nlp.rule import RuleNLPService
    return RuleNLPService()


def head():
    return AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})


def encode(nlp, text, title="", surfaces=(), sketch=True, h=None):
    doc = nlp.parse(text, title=title)
    if sketch:
        doc.sketch = resolve_sketch(doc.skeleton, [EntityRef(f"e:{s.lower()}", s) for s in surfaces])
    res = (h or head()).encode_batch_texts([text], [{"spacy_doc": doc}])[0]
    return res, doc


def chain(doc, name):
    return next(c.chain_id for c in doc.sketch.chains if c.canonical == name)


def test_values_link_actor_and_patient_chains(nlp):
    text = "Brigham Young was a leader. He founded Salt Lake City in 1847."
    res, doc = encode(nlp, text, "Brigham Young", ["Brigham Young", "Salt Lake City"])
    assert res.metadata["sketch"] == RESOLVED_CONTRACT
    found = next(a for a in res.addresses if a.concept in ("found", "establish"))
    tok = next(t.i for t in doc.tokens if t.text == "founded")
    assert found.metadata["links"] == [[chain(doc, "Brigham Young"), tok], [chain(doc, "Salt Lake City"), tok]]


def test_passive_voice_comes_from_the_clause(nlp, monkeypatch):
    text = "Salt Lake City was founded by Brigham Young."
    res, doc = encode(nlp, text, "", ["Salt Lake City", "Brigham Young"])
    found = next(a for a in res.addresses if a.concept in ("found", "establish"))
    tok = next(t.i for t in doc.tokens if t.text == "founded")
    # actor = the by-agent, patient = the grammatical subject
    assert found.metadata["links"] == [[chain(doc, "Brigham Young"), tok], [chain(doc, "Salt Lake City"), tok]]
    # the score's passive penalty follows the clause's voice, not the token morphology
    import nam_heads.impls.affordance.default.head as H
    real = H.action_slots
    monkeypatch.setattr(H, "action_slots", lambda sk: {t: ("active", c) for t, (_v, c) in real(sk).items()})
    as_active, _d = encode(nlp, text, "", ["Salt Lake City", "Brigham Young"])
    act = next(a for a in as_active.addresses if a.concept == found.concept)
    assert found.confidence < act.confidence


def test_predicate_without_a_clause_is_not_emitted_but_learned(nlp, monkeypatch):
    """A predicate the extractor reads where the sketch has no clause action
    is dropped from the output; the store still records it."""
    import nam_heads.impls.affordance.default.head as H
    text = "Brigham Young was a leader. He founded Salt Lake City."
    real = H.action_slots

    def without_found(sk):
        return {t: v for t, v in real(sk).items()
                if sk.doc is None or sk.doc.tokens[t].text != "founded"}
    monkeypatch.setattr(H, "action_slots", without_found)
    h = head()
    seen = []
    from nam_heads.impls.affordance.default import predicate_store as PS
    orig = PS.StoreLexicon.record_value

    def spy(self, p, cls):
        seen.append(p.value)
        return orig(self, p, cls)
    monkeypatch.setattr(PS.StoreLexicon, "record_value", spy)
    res, _doc = encode(nlp, text, "Brigham Young", ["Brigham Young", "Salt Lake City"], h=h)
    assert not {a.concept for a in res.addresses} & {"found", "establish"}
    assert set(seen) & {"found", "establish"}


def test_without_sketch_the_head_is_unchanged(nlp, monkeypatch):
    text = "Brigham Young was a leader. He founded Salt Lake City."
    plain, _d = encode(nlp, text, "Brigham Young", sketch=False)
    assert "sketch" not in plain.metadata
    assert all("links" not in (a.metadata or {}) for a in plain.addresses)
    monkeypatch.setenv("NAM_SKETCH_SLOTS", "0")
    off, _d = encode(nlp, text, "Brigham Young", ["Brigham Young", "Salt Lake City"])
    assert [(a.concept, a.confidence, a.metadata) for a in off.addresses] == \
        [(a.concept, a.confidence, a.metadata) for a in plain.addresses]
