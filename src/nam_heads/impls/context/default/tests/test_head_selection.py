"""heads/context/head.py: the particular-evidence extractor by default, the
legacy NER extraction behind NAM_CONTEXT_HEAD=legacy, one head contract."""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.head import ContextEncoder
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.ontology import bindings as B

TEXT = "Lincoln was born in Kentucky in 1809. He worked for the Illinois Central Railroad."


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def _enc(monkeypatch, mode=None, **cfg):
    if mode is None:
        monkeypatch.delenv("NAM_CONTEXT_HEAD", raising=False)
    else:
        monkeypatch.setenv("NAM_CONTEXT_HEAD", mode)
    return ContextEncoder(head_type="context", config={"context_store_backend": "memory", **cfg})


def _ont(cls):
    return EncodingResult("ontology", [SemanticAddress("ontology", cls, 0.9)], {})


def test_default_is_particular(monkeypatch, nlp):
    enc = _enc(monkeypatch)
    assert enc.head_mode == "particular"
    got = [a.concept for a in enc.encode(TEXT, {"spacy_doc": nlp.parse(TEXT)}).addresses]
    assert "t:C19/D1800/Y1809" in got and "g:NA/US/US-KY" in got


def test_legacy_is_selectable(monkeypatch, nlp):
    enc = _enc(monkeypatch, "legacy")
    assert enc.head_mode == "legacy" and enc._impl is None
    got = [a.concept for a in enc.encode(TEXT, {"spacy_doc": nlp.parse(TEXT)}).addresses]
    assert not any(c.startswith(("t:", "g:")) for c in got)
    assert ContextEncoder(head_type="context", config={"context_store_backend": "memory",
                                                       "head": "legacy"}).head_mode == "legacy"


def test_unknown_mode_refused(monkeypatch):
    with pytest.raises(ValueError):
        _enc(monkeypatch, "bogus")


def test_contract_store_metrics_config_and_paths_agree(monkeypatch, nlp):
    enc = _enc(monkeypatch)
    assert enc.store is enc._impl.store and hasattr(enc.store, "get_batch")
    d = nlp.parse(TEXT)
    one = [a.concept for a in enc.encode(TEXT, {"spacy_doc": d}).addresses]
    batch = [a.concept for a in enc.encode_batch_texts([TEXT], [{"spacy_doc": d}])[0].addresses]
    query = [a.concept for a in enc.encode_query(TEXT, {"spacy_doc": d}).addresses]
    assert one == batch == query
    m = enc.get_metrics()
    assert "context_store_size" in m and m["context_head"] == "particular"
    assert m["head_type"] == "context"
    c = enc.config_summary()
    assert c["head"] == "particular" and "context_store_backend" in c


def test_no_type_derived_domain_or_commercial_default(monkeypatch, nlp):
    """Owner rejected values derived from the record's type (DOMAIN_SEEDS) or
    the 'commercial' ORG default; the default head emits neither."""
    from nam_heads.impls.context.default.definitions import DOMAIN_SEEDS, INSTITUTIONAL_DEFAULT
    enc = _enc(monkeypatch)
    text = "Acme Holdings sold tools to farmers."
    for cls in (B.ORGANIZATION, B.PERSON, B.GEOSPATIAL_REGION):
        got = {a.concept for a in enc.encode(text, {"spacy_doc": nlp.parse(text),
                                                    "ontology_result": _ont(cls)}).addresses}
        assert INSTITUTIONAL_DEFAULT not in got
        assert not got & set(DOMAIN_SEEDS.values())


def test_discovery_finds_one_head_class():
    from src.encoders.registry import EncoderRegistry
    found = EncoderRegistry().discover_encoders()
    assert found["context"] is ContextEncoder
