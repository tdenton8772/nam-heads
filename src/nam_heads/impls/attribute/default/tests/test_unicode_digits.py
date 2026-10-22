"""Non-ASCII digits in the attribute and affordance heads.

A year in a lifespan parenthetical is a run of decimal digits
(``str.isdecimal``): "(1905 - 1981)" and its Arabic-Indic spelling are a
lifespan; superscript digits "(¹⁹⁰⁵ - ¹⁹⁸¹)" are not years (str.isdigit() was
True for them). Neither head raises on '²', '⁵', '⑦', '½' or '٣٤'.
"""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.attribute.default.predication import _lifespan_between

TEXTS = [
    "It was founded in ² BC and rebuilt in AD ⑦.",
    "Its area is 5 km² and it dates from ⁵ AD.",
    "The rite lasted ½ BC in some accounts.",
    "He ruled from ٣٤ BC to AD ٣٤ and was a famous red king.",
    "Albert (¹⁹⁰⁵ - ¹⁹⁸¹) was a Belgian painter.",
]


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def _span(doc):
    toks = [t.text for t in doc]
    return toks.index("("), toks.index(")") + 1


@pytest.mark.parametrize("text,expected", [
    ("Albert (1905 - 1981) was a painter.", True),
    ("Albert (١٩٠٥ - ١٩٨١) was a painter.", True),
    ("Albert (¹⁹⁰⁵ - ¹⁹⁸¹) was a painter.", False),
    ("Albert (²²² - ⑦⑦⑦) was a painter.", False),
])
def test_lifespan_years_are_decimal_digits(nlp, text, expected):
    doc = nlp.parse(text)
    a, b = _span(doc)
    assert _lifespan_between(doc, a, b) is expected


def _ctx(nlp, texts):
    return [{"spacy_doc": nlp.parse(t)} for t in texts]


def test_attribute_head_batch_survives(nlp):
    from nam_heads.impls.attribute.default.head import AttributeEncoder
    enc = AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})
    out = enc.encode_batch_texts(TEXTS, _ctx(nlp, TEXTS))
    assert len(out) == len(TEXTS)


def test_affordance_head_batch_survives(nlp):
    from nam_heads.impls.affordance.default.head import AffordanceEncoder
    enc = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    out = enc.encode_batch_texts(TEXTS, _ctx(nlp, TEXTS))
    assert len(out) == len(TEXTS)
