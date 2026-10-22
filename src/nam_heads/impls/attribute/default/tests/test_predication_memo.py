"""extract_predications is computed once per doc per batch (memoized by
identity); the memoized result equals a fresh extraction and never leaks to
another doc."""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.attribute.default import predication as PR

TEXTS = [
    "Albert Brooks (born July 22, 1947) is an American actor and director.",
    "The red fox is a small, agile mammal.",
    "Python is a large, nonvenomous snake.",
]


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def test_same_doc_same_list_equal_to_fresh(nlp):
    doc = nlp.parse(TEXTS[0])
    a = PR.extract_predications(doc)
    assert PR.extract_predications(doc) is a
    assert a == PR._extract_predications(doc)


def test_different_docs_do_not_share(nlp):
    docs = [nlp.parse(t) for t in TEXTS]
    outs = [PR.extract_predications(d) for d in docs]
    for d, o in zip(docs, outs):
        assert o == PR._extract_predications(d)
    # The same text parsed again is a new doc object: a fresh entry.
    again = nlp.parse(TEXTS[1])
    assert PR.extract_predications(again) == outs[1]


def test_memo_is_bounded(nlp, monkeypatch):
    monkeypatch.setattr(PR, "_PRED_MEMO_MAX", 4)
    PR._PRED_MEMO.clear()
    for k in range(10):
        PR.extract_predications(nlp.parse(f"The {k} red fox is a small mammal."))
        assert len(PR._PRED_MEMO) <= 4


def test_three_callers_one_extraction(nlp, monkeypatch):
    """candidate_values, attribute_values and legacy_anchors on one doc run
    the extraction once -- per title: the title decides the main subject, so
    it is part of the memo key (the head passes the same title to all three)."""
    calls = {"n": 0}
    real = PR._extract_predications

    def counting(doc, title=""):
        calls["n"] += 1
        return real(doc, title)
    monkeypatch.setattr(PR, "_extract_predications", counting)
    doc = nlp.parse(TEXTS[2])
    PR.candidate_values(doc)
    PR.attribute_values(doc)
    PR.legacy_anchors(doc, ["large"])
    assert calls["n"] == 1
    PR.candidate_values(doc, title="Python")
    PR.attribute_values(doc, title="Python")
    PR.legacy_anchors(doc, ["large"], title="Python")
    assert calls["n"] == 2


def test_memo_keys_on_the_title(nlp):
    doc = nlp.parse("Python. It is a large snake.")
    plain = PR.extract_predications(doc)
    titled = PR.extract_predications(doc, "Python")
    assert plain is not titled
    assert plain.main is None and titled.main == (0, 1)
    assert PR.extract_predications(doc, "Python") is titled
