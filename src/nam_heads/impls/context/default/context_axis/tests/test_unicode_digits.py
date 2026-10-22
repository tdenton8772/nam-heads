"""Non-ASCII digits never fail the context head.

Production (2026-10-04): context.encode_batch_failed
"invalid literal for int() with base 10: '²'" (also '⑦', '⁵'). str.isdigit()
is True for superscripts and circled digits, but int() reads only decimal
digits (Unicode category Nd), so one such token failed the whole batch of 32.

Decisions (temporal.extract; a number is ``str.isdecimal()``):
  '²' '⁵' '⑦'  superscript / circled: not a number -- no value, no error
  '½'          a vulgar fraction: not a number (it never passed isdigit)
  '٣٤'         Arabic-Indic decimal digits: the number 34, as int() and the
               regexes' \\d already read it -- "٣٤ BC" == "34 BC"
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.context_axis import temporal
from nam_heads.impls.context.default.context_axis.encoder import ParticularContextEncoder
from nam_heads.impls.context.default.context_axis.seed import seed_store
from nam_heads.impls.context.default.storage.memory_store import MemoryContextStore


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


@pytest.fixture(scope="module")
def store():
    st = MemoryContextStore()
    seed_store(st)
    return st


def _toks(*words):
    return [SimpleNamespace(text=w, i=k) for k, w in enumerate(words)]


LEX = SimpleNamespace(periods={}, era_words={})


@pytest.mark.parametrize("words", [
    ("²", "BC"), ("⁵", "AD"), ("⑦", "BC"), ("AD", "²"), ("AD", "⑦"), ("AD", "⁵"),
    ("½", "BC"), ("AD", "½"),
])
def test_superscript_circled_and_fraction_are_not_numbers(words):
    assert temporal.extract(_toks(*words), [], LEX) == []


def test_arabic_indic_digits_read_as_their_number():
    assert temporal.extract(_toks("٣٤", "BC"), [], LEX) == \
        temporal.extract(_toks("34", "BC"), [], LEX) == [(0, 2, "C-1/D-30/Y-34", "era_year")]
    assert temporal.extract(_toks("AD", "٣٤"), [], LEX) == \
        temporal.extract(_toks("AD", "34"), [], LEX) == [(0, 2, "C1/D30/Y34", "era_year")]


def test_ascii_years_unchanged():
    assert temporal.extract(_toks("44", "BC"), [], LEX) == [(0, 2, "C-1/D-40/Y-44", "era_year")]
    assert temporal.extract(_toks("AD", "800"), [], LEX) == [(0, 2, "C9/D800/Y800", "era_year")]


BAD = [
    "It was founded in ² BC and rebuilt in AD ⑦.",
    "Its area is 5 km² and it dates from ⁵ AD.",
    "The rite lasted ½ BC in some accounts.",
]


def _texts_and_ctx(nlp, texts):
    return texts, [{"spacy_doc": nlp.parse(t)} for t in texts]


def test_rule_nlp_hands_the_bad_tokens_to_the_head(nlp):
    """The tests below exercise the crash only if the tokens survive parsing."""
    toks = {t.text for text in BAD for t in nlp.parse(text)}
    assert {"²", "⑦", "⁵", "½"} <= toks


def test_a_batch_survives_one_bad_token(nlp, store):
    good = ["Abraham Lincoln was born in 1809 in Kentucky.",
            "The Roman general fought in 44 BC near Rome."]
    enc = ParticularContextEncoder(head_type="context", config={"store": store, "learn": False})
    texts, ctx = _texts_and_ctx(nlp, good[:1] + BAD + good[1:])
    out = enc.encode_batch_texts(texts, ctx)            # must not raise
    assert len(out) == len(texts)
    # The good records get exactly what they get in a batch without the bad ones.
    alone = enc.encode_batch_texts(*_texts_and_ctx(nlp, good))
    assert [a.concept for a in out[0].addresses] == [a.concept for a in alone[0].addresses]
    assert [a.concept for a in out[-1].addresses] == [a.concept for a in alone[1].addresses]
    assert "t:C-1/D-40/Y-44" in {a.concept for a in out[-1].addresses}
    # The bad tokens yield no temporal value.
    for r in out[1:-1]:
        assert not [a.concept for a in r.addresses if a.concept.startswith("t:")]


def test_arabic_indic_record_matches_ascii_record(nlp, store):
    enc = ParticularContextEncoder(head_type="context", config={"store": store, "learn": False})
    a = enc.encode_batch_texts(*_texts_and_ctx(nlp, ["He ruled from ٣٤ BC to AD ٣٤."]))[0]
    b = enc.encode_batch_texts(*_texts_and_ctx(nlp, ["He ruled from 34 BC to AD 34."]))[0]
    ta = sorted(x.concept for x in a.addresses if x.concept.startswith("t:"))
    tb = sorted(x.concept for x in b.addresses if x.concept.startswith("t:"))
    assert ta == tb == ["t:C-1/D-30/Y-34", "t:C1/D30/Y34"]


def test_pipeline_context_head_survives_the_batch(nlp, store):
    """The class nats_worker runs (heads/context/head.py, particular mode)."""
    from nam_heads.impls.context.default.head import ContextEncoder
    enc = ContextEncoder(head_type="context", config={"store": store, "learn": False})
    texts, ctx = _texts_and_ctx(nlp, BAD + ["He ruled from ٣٤ BC."])
    out = enc.encode_batch_texts(texts, ctx)
    assert len(out) == len(texts)
