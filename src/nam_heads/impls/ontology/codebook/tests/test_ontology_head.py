"""The CCO ontology head (contract the NATS stage, bundler and query rely on)."""

import json

import pytest

from nam_heads.sdk import host

# These tests run the codebook against real parses and its parser fingerprint,
# so they need the host's parser (NAM_SRC=<nam checkout>/nam, see conftest.py).
# tests/test_conformance.py covers the head without one.
pytestmark = pytest.mark.skipif(not host.has_parser(), reason="needs a host parser")

from nam_heads.impls.ontology.codebook.codebook import UNCLASSIFIED, CodebookError
from nam_heads.impls.ontology.codebook.head import OntologyEncoder
from nam_heads.sdk import EncodingResult
from nam_heads.sdk.testing import parser_or_skip
from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import get_registry


@pytest.fixture(scope="module")
def enc():
    return OntologyEncoder(head_type="ontology")


@pytest.fixture(scope="module")
def nlp():
    return parser_or_skip()


def test_encode_returns_sorted_certified_classes(enc, nlp):
    text = "Albert_Einstein. Albert Einstein was a German-born theoretical physicist. He received the Nobel Prize."
    res = enc.encode(text, {"spacy_doc": nlp.parse(text)})
    assert isinstance(res, EncodingResult)
    reg = get_registry()
    assert all(reg.is_class(a.concept) for a in res.addresses)
    confs = [a.confidence for a in res.addresses[1:]]
    assert confs == sorted(confs, reverse=True)
    assert res.addresses[0].concept == B.PERSON
    json.dumps(res.metadata)                       # crosses NATS as JSON


def test_sentence_classes_published_for_the_bundler(enc, nlp):
    text = "Albert_Einstein. Albert Einstein was a German-born theoretical physicist. He received the Nobel Prize."
    meta = enc.encode(text, {"spacy_doc": nlp.parse(text)}).metadata
    assert meta["sentence_ontologies"].get(1) == B.PERSON
    assert 0 not in meta["sentence_ontologies"]     # the bare "Title." fragment does not vote
    assert set(meta["sentence_confidences"]) == set(meta["sentence_ontologies"])


def test_classify_nearest_matches_sentence_vocabulary(enc, nlp):
    doc = nlp.parse("The Beatles were an English rock band formed in Liverpool in 1960.")
    cls = enc.classify_nearest("", spacy_doc=next(doc.sents))
    assert cls is None or get_registry().is_class(cls)
    assert enc.classify_nearest("x", spacy_doc=None) is None


def test_no_parse_abstains_and_says_so(enc):
    res = enc.encode("Paris is the capital of France.", {})
    assert res.addresses[0].concept == UNCLASSIFIED
    assert res.metadata["abstained"] == "no_parse"


def test_skip_nlp_flag_does_not_change_the_head(enc, nlp, monkeypatch):
    monkeypatch.setenv("NAM_SKIP_HEAD_NLP", "1")
    text = "The Beatles were an English rock band formed in Liverpool in 1960."
    a = enc.encode(text, {"spacy_doc": nlp.parse(text)})
    assert a.addresses[0].concept != UNCLASSIFIED or a.metadata.get("abstained") is not None


def test_query_uses_the_same_classifier(enc, nlp):
    q = "The Beatles were an English rock band."
    ing = enc.encode(q, {"spacy_doc": nlp.parse(q)})
    qry = enc.encode_query(q, {"spacy_doc": nlp.parse(q)})
    assert [a.concept for a in ing.addresses] == [a.concept for a in qry.addresses]
    assert qry.metadata["mode"] == "query"


def test_emittable_classes_cover_backoff_and_root(enc):
    reg = get_registry()
    em = set(enc.emittable_classes())
    assert UNCLASSIFIED in em
    for c in enc.codebook.classes:
        assert reg.ancestors(c, inclusive=True) <= em


def test_refuses_to_start_without_codebook(tmp_path):
    with pytest.raises(CodebookError):
        OntologyEncoder(head_type="ontology", config={"codebook_dir": str(tmp_path)})
