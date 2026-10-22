"""The predicate-mode prefetch asks only for keys a rule can consult.

It asked for ~1,000 keys per record on wiki text (every candidate value x
every sentence ontology as a scoped category key, a phrasal doc for every
next word); each is an L1/LMDB lookup on every batch. Phrasal docs/values
can exist only for a seed pair or a true particle, and a scoped category row
only for definitions.ONTOLOGY_CATEGORY_OVERRIDES -- the prefetch now asks for
those alone, and extraction still never reads a key outside the prefetch.
"""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.affordance.default import predicate_store as PS
from nam_heads.impls.affordance.default import predicates as P
from nam_heads.impls.affordance.default.definitions import ONTOLOGY_CATEGORY_OVERRIDES
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from nam_heads.impls.affordance.default.storage.factory import _seed_store
from nam_heads.impls.affordance.default.storage.memory_store import MemoryCategoryStore
from nam_heads.ontology import bindings as B

TEXTS = [
    "The ship was laid down in 1910 and launched in 1911 by the navy.",
    "Smith founded the company in 1920 and later signed a contract with Jones.",
    "She grew up in Ohio, was known as a painter, and gave up teaching.",
    "The band teamed up with the label, which was set up to record their songs.",
    "The device is used to measure pressure and was designed by engineers.",
    "He took part in the war and passed away in 1950.",
]


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


class RecordingStore(MemoryCategoryStore):
    def __init__(self):
        super().__init__()
        self.requested = []

    def get_docs_batch(self, names):
        self.requested.append(list(names))
        return super().get_docs_batch(names)


def _ont(cls):
    class R:
        addresses = [type("A", (), {"concept": cls})()]
        metadata = {"sentence_ontologies": {0: cls, 1: B.PERSON, 2: B.ORGANIZATION}}
    return R()


def _head(store):
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "emit_mode": "predicate"})
    h.category_store = store
    return h


def test_prefetch_keys_are_bounded_and_cover_extraction(nlp):
    s = RecordingStore()
    _seed_store(s)
    h = _head(s)
    ctx = [{"spacy_doc": nlp.parse(t), "ontology_result": _ont(B.ORGANIZATION)} for t in TEXTS]
    h.encode_batch_texts(TEXTS, ctx)
    # One read for the batch (the prefetch covered every key consulted).
    assert len(s.requested) == 1
    names = s.requested[0]
    assert len(names) / len(TEXTS) < 120
    for n in names:
        if n.startswith(PS.PHR):
            v, part = n[len(PS.PHR):].split(":")
            assert part in PS.TRUE_PARTICLES or (v, part) in P.PHRASAL, n
        elif "::" in n:
            v, ont = n.split("::", 1)
            assert v in ONTOLOGY_CATEGORY_OVERRIDES.get(ont, {}), n


def test_scoped_override_row_still_reaches_the_category(nlp):
    """'found' under Organization: the seeded override (institutional) is
    prefetched and read for the predicate's category metadata."""
    s = RecordingStore()
    _seed_store(s)
    h = _head(s)
    text = "Smith founded the company in 1920."
    doc = nlp.parse(text)

    class Ont:
        addresses = [type("A", (), {"concept": B.ORGANIZATION})()]
        metadata = {"sentence_ontologies": {0: B.ORGANIZATION}}
    r = h.encode_batch_texts([text], [{"spacy_doc": doc, "ontology_result": Ont()}])[0]
    assert f"found::{B.ORGANIZATION}" in s.requested[0]
    cats = {a.concept: a.metadata.get("category") for a in r.addresses}
    assert ONTOLOGY_CATEGORY_OVERRIDES[B.ORGANIZATION]["found"] == "institutional"
    assert any(c == "institutional" for c in cats.values()), cats


def test_token_keys_match_the_unrestricted_rule_where_it_matters():
    """Every phrasal key the old rule asked for that _token_keys drops is one
    ``StoreLexicon.phrasal`` can never accept (not seeded, not a true particle)."""
    for text, nxt in [("laid", ("down", "in")), ("founded", ("the", "company")),
                      ("known", ("as", "a")), ("grew", ("up", "in")), ("set", ("up", "to"))]:
        keys, values = PS._token_keys(text, "", nxt)
        cands = P.surface_candidates(text)
        cands += [P.seed_synonyms()[c] for c in list(cands) if c in P.seed_synonyms()]
        for c in cands:
            for w in nxt:
                k = f"{PS.PHR}{c}:{w}"
                if k not in keys:
                    assert w not in PS.TRUE_PARTICLES and (c, w) not in P.PHRASAL
        assert all(PS.BASE + c in keys and PS.SYN + c in keys for c in cands)
