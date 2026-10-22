"""Every case of test_attribute_head.py, run through the DEFAULT path
(predication_mode on) with the doc supplied by the CALLER, as the NLP stage
supplies it: the rule-parsed ParsedDocument is put in the context as
``spacy_doc`` before encode()/encode_query() sees it. test_attribute_head.py
itself encodes most texts with no doc, so there the head parses them
(record_parse); both paths must give the cases' answers.

The cases are imported, not copied: a change to test_attribute_head.py is
exercised here too.
"""
import inspect
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.attribute.default import head as head_mod
from nam_heads.impls.attribute.default.tests import test_attribute_head as cases

CASES = sorted(n for n, f in inspect.getmembers(cases, inspect.isfunction) if n.startswith("test_"))


@pytest.fixture
def default_path(monkeypatch):
    from src.nlp import get_nlp_service
    nlp = get_nlp_service()
    orig_encode = head_mod.AttributeEncoder.encode
    orig_query = head_mod.AttributeEncoder.encode_query
    seen = {"docs": 0}

    def with_doc(fn):
        def wrapped(self, text, context=None):
            ctx = dict(context or {})
            if ctx.get("spacy_doc") is None:
                ctx["spacy_doc"] = nlp.parse(text)
            seen["docs"] += 1
            assert self.predication_mode
            return fn(self, text, ctx)
        return wrapped

    monkeypatch.setattr(head_mod.AttributeEncoder, "encode", with_doc(orig_encode))
    monkeypatch.setattr(head_mod.AttributeEncoder, "encode_query", with_doc(orig_query))
    return seen


@pytest.mark.parametrize("name", CASES)
def test_case_on_default_path(name, default_path):
    fn = getattr(cases, name)
    params = inspect.signature(fn).parameters
    kwargs = {}
    if "encoder" in params:
        kwargs["encoder"] = head_mod.AttributeEncoder(
            head_type="attribute", config={"category_store_backend": "memory"})
    fn(**kwargs)
    assert default_path["docs"] > 0 or name in NO_ENCODE


# Cases that never call encode() (store / extractor-only checks).
NO_ENCODE = {n for n in CASES if ".encode(" not in inspect.getsource(getattr(cases, n))}
