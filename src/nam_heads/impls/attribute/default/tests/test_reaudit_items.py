"""Scoped category rows, min_len, per-record bound, explicit evidence
exclusions, and abstain (bfo:entity) handling on the predication path."""
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from nam_heads.impls.attribute.default.head import AttributeEncoder
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.ontology import bindings as B


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def _ont(concept, sentence_ontologies=None):
    meta = {"sentence_ontologies": sentence_ontologies} if sentence_ontologies else {}
    return EncodingResult("ontology", [SemanticAddress("ontology", concept)], meta)


def _enc(**cfg):
    return AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory", **cfg})


def _pred(enc, nlp, text, ont=None, sent=None):
    ctx = {"spacy_doc": nlp.parse(text)}
    if ont:
        ctx["ontology_result"] = _ont(ont, sent)
    res = enc._settle_predications([text], [ctx], mode="query")[0]
    return {a.concept: a for a in res.addresses if a.concept != "__null__"}


def test_scoped_category_rows_are_read(nlp):
    """acat::lemma::ontology (ONTOLOGY_CATEGORY_OVERRIDES) weighs the score
    under that ontology, as on the legacy path."""
    enc = _enc()
    text = "Bob Smith is a tall carpenter."
    base = _pred(enc, nlp, text, B.PERSON)["tall"].confidence
    enc.category_store.put_category_weighted("tall", "emotional", 1.0, ontology_type=B.PERSON)
    assert _pred(enc, nlp, text, B.PERSON)["tall"].confidence > base   # emotional 1.3 vs physical 0.9


def test_configured_min_len_is_honoured(nlp):
    text = "Bob Smith is a tall carpenter."
    assert "tall" in _pred(_enc(min_len=2), nlp, text)
    assert "tall" not in _pred(_enc(min_len=5), nlp, text)


def test_per_record_predication_bound(nlp):
    text = ("Joel Greenblatt is an American academic, manager, investor and writer. "
            "Mary Lee is a Canadian singer and painter.")
    assert len(_pred(_enc(), nlp, text)) > 3
    assert len(_pred(_enc(max_predicated_per_record=3), nlp, text)) == 3


def test_evidence_exclusion_is_explicit_and_switchable(nlp, monkeypatch):
    text = "Mad Love is a film."
    P = B.PERSON
    monkeypatch.delenv("NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS", raising=False)
    # OFF by default (it removed more correct values than junk at current
    # ontology quality); config and env both turn it on.
    assert "film" in _pred(_enc(), nlp, text, P, {0: P})
    assert "film" not in _pred(_enc(evidence_exclusions=True), nlp, text, P, {0: P})
    assert "film" in _pred(_enc(evidence_exclusions=False), nlp, text, P, {0: P})
    monkeypatch.setenv("NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS", "1")
    assert "film" not in _pred(_enc(), nlp, text, P, {0: P})
    monkeypatch.setenv("NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS", "0")
    assert "film" in _pred(_enc(), nlp, text, P, {0: P})


def test_abstaining_sentence_ontology_falls_back_to_the_record(nlp):
    text = "Mad Love is a period drama film."
    enc = _enc(evidence_exclusions=True)
    # sentence abstains -> record ontology (object) steers: kind noun admitted
    assert "film" in _pred(enc, nlp, text, B.MATERIAL_ARTIFACT, {0: B.ENTITY})
    assert "film" in _pred(enc, nlp, text, B.MATERIAL_ARTIFACT)     # sentence absent
    # sentence abstains, record person -> person steers: kind noun excluded
    assert "film" not in _pred(enc, nlp, text, B.PERSON, {0: B.ENTITY})
    # record abstains, no sentence ontology -> neutral
    assert "film" in _pred(enc, nlp, text, B.ENTITY)
