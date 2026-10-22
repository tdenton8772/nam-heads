"""Tests for the rule-based dependency parser."""

import pytest
from ..tokenizer import Tokenizer
from ..pos_tagger import POSTagger
from ..lemmatizer import Lemmatizer
from ..ner import NERRecognizer
from ..dep_parser import DependencyParser
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def pipeline():
    tokenizer = Tokenizer()
    tagger = POSTagger(DATA_DIR)
    lemmatizer = Lemmatizer(DATA_DIR)
    ner = NERRecognizer(DATA_DIR)
    dep_parser = DependencyParser()
    return tokenizer, tagger, lemmatizer, ner, dep_parser


def _parse(pipeline, text):
    tokenizer, tagger, lemmatizer, ner, dep_parser = pipeline
    tokens, sents = tokenizer.tokenize(text)
    tagger.tag(tokens, sents)
    lemmatizer.lemmatize(tokens)
    ner.recognize(tokens, sents)
    dep_parser.parse(tokens, sents)
    return tokens, sents


class TestROOT:
    def test_verb_is_root(self, pipeline):
        tokens, _ = _parse(pipeline, "The cat sat.")
        root = [t for t in tokens if t.dep == "ROOT"]
        assert len(root) == 1
        assert root[0].pos == "VERB"

    def test_root_heads_self(self, pipeline):
        tokens, _ = _parse(pipeline, "The cat sat.")
        root = [t for t in tokens if t.dep == "ROOT"][0]
        assert root.head_idx == root.idx


class TestSubjects:
    def test_nsubj(self, pipeline):
        tokens, _ = _parse(pipeline, "The cat sat.")
        subj = [t for t in tokens if t.dep == "nsubj"]
        assert len(subj) == 1
        assert subj[0].text == "cat"

    def test_nsubjpass(self, pipeline):
        tokens, _ = _parse(pipeline, "The pump was repaired.")
        subj = [t for t in tokens if t.dep == "nsubjpass"]
        assert len(subj) == 1
        assert subj[0].text == "pump"


class TestObjects:
    def test_dobj(self, pipeline):
        tokens, _ = _parse(pipeline, "He repaired the pump.")
        obj = [t for t in tokens if t.dep == "dobj"]
        assert len(obj) == 1
        assert obj[0].text == "pump"


class TestModifiers:
    def test_amod(self, pipeline):
        tokens, _ = _parse(pipeline, "The big dog barked.")
        amod = [t for t in tokens if t.dep == "amod"]
        assert len(amod) >= 1
        assert amod[0].text == "big"

    def test_advmod(self, pipeline):
        tokens, _ = _parse(pipeline, "He ran quickly.")
        advmod = [t for t in tokens if t.dep == "advmod"]
        assert len(advmod) >= 1

    def test_det(self, pipeline):
        tokens, _ = _parse(pipeline, "The cat sat.")
        det = [t for t in tokens if t.dep == "det"]
        assert len(det) >= 1
        assert det[0].text == "The"

    def test_compound(self, pipeline):
        tokens, _ = _parse(pipeline, "The fire truck arrived.")
        compound = [t for t in tokens if t.dep == "compound"]
        assert len(compound) >= 1


class TestCompoundPROPN:
    """Consecutive PROPN tokens without a verb should form a compound."""

    def test_two_propn_no_verb(self, pipeline):
        """'Albert Einstein' — Albert should be compound, Einstein ROOT."""
        tokens, _ = _parse(pipeline, "Albert Einstein")
        albert = [t for t in tokens if t.text == "Albert"][0]
        einstein = [t for t in tokens if t.text == "Einstein"][0]
        assert albert.dep == "compound"
        assert albert.head_idx == einstein.idx
        assert einstein.dep == "ROOT"
        assert einstein.head_idx == einstein.idx

    def test_three_propn_no_verb(self, pipeline):
        """'Martin Luther King' — ROOT cascades to rightmost PROPN."""
        tokens, _ = _parse(pipeline, "Martin Luther King")
        martin = [t for t in tokens if t.text == "Martin"][0]
        luther = [t for t in tokens if t.text == "Luther"][0]
        king = [t for t in tokens if t.text == "King"][0]
        assert martin.dep == "compound"
        assert luther.dep == "compound"
        assert king.dep == "ROOT"
        assert king.head_idx == king.idx

    def test_propn_compound_with_verb(self, pipeline):
        """'Albert Einstein discovered relativity' — verb is ROOT, not PROPN."""
        tokens, _ = _parse(pipeline, "Albert Einstein discovered relativity.")
        albert = [t for t in tokens if t.text == "Albert"][0]
        einstein = [t for t in tokens if t.text == "Einstein"][0]
        root = [t for t in tokens if t.dep == "ROOT"]
        assert albert.dep == "compound"
        assert albert.head_idx == einstein.idx
        assert len(root) == 1
        assert root[0].text == "discovered"

    def test_compound_noun_no_verb(self, pipeline):
        """'fire truck' — compound NOUN pair, ROOT transfers to rightmost."""
        tokens, _ = _parse(pipeline, "fire truck")
        fire = [t for t in tokens if t.text == "fire"][0]
        truck = [t for t in tokens if t.text == "truck"][0]
        assert fire.dep == "compound"
        assert fire.head_idx == truck.idx
        assert truck.dep == "ROOT"


class TestPrepositions:
    def test_prep(self, pipeline):
        tokens, _ = _parse(pipeline, "He sat on the chair.")
        prep = [t for t in tokens if t.dep == "prep"]
        assert len(prep) >= 1
        assert prep[0].text == "on"

    def test_pobj(self, pipeline):
        tokens, _ = _parse(pipeline, "He sat on the chair.")
        pobj = [t for t in tokens if t.dep == "pobj"]
        assert len(pobj) >= 1
        assert pobj[0].text == "chair"


class TestPassiveVoice:
    def test_passive_morph(self, pipeline):
        tokens, _ = _parse(pipeline, "The pump was repaired.")
        repaired = [t for t in tokens if t.text == "repaired"][0]
        assert repaired.morph_data.get("Voice") == ["Pass"]


class TestCoordination:
    def test_conj(self, pipeline):
        tokens, _ = _parse(pipeline, "Cats and dogs are friends.")
        conj = [t for t in tokens if t.dep == "conj"]
        assert len(conj) >= 1

    def test_cc(self, pipeline):
        tokens, _ = _parse(pipeline, "Cats and dogs are friends.")
        cc = [t for t in tokens if t.dep == "cc"]
        assert len(cc) >= 1
        assert cc[0].text == "and"


class TestSubordination:
    def test_advcl(self, pipeline):
        tokens, _ = _parse(pipeline, "He left because she arrived.")
        advcl = [t for t in tokens if t.dep == "advcl"]
        assert len(advcl) >= 1

    def test_mark(self, pipeline):
        tokens, _ = _parse(pipeline, "He left because she arrived.")
        mark = [t for t in tokens if t.dep == "mark"]
        assert len(mark) >= 1
        assert mark[0].text == "because"


class TestXcomp:
    def test_xcomp(self, pipeline):
        tokens, _ = _parse(pipeline, "He wants to run.")
        xcomp = [t for t in tokens if t.dep == "xcomp"]
        assert len(xcomp) >= 1
        assert xcomp[0].text == "run"


class TestVBNAsAmod:
    def test_participial_modifier(self, pipeline):
        tokens, _ = _parse(pipeline, "The broken window was found.")
        broken = [t for t in tokens if t.text == "broken"][0]
        assert broken.tag == "VBN"
        assert broken.dep == "amod"


class TestCorrectnessGuarantees:
    def test_every_token_has_dep(self, pipeline):
        tokens, _ = _parse(pipeline, "The angry technician repaired the pump in Detroit.")
        for t in tokens:
            assert t.dep, f"Token '{t.text}' has no dep"

    def test_every_token_has_valid_head(self, pipeline):
        tokens, _ = _parse(pipeline, "The angry technician repaired the pump in Detroit.")
        idxs = {t.idx for t in tokens}
        for t in tokens:
            assert t.head_idx in idxs, f"Token '{t.text}' has invalid head_idx {t.head_idx}"

    def test_root_heads_self(self, pipeline):
        tokens, _ = _parse(pipeline, "The angry technician repaired the pump in Detroit.")
        roots = [t for t in tokens if t.dep == "ROOT"]
        assert len(roots) == 1
        assert roots[0].head_idx == roots[0].idx

    def test_no_cycles(self, pipeline):
        """Every token should reach ROOT by following head links."""
        tokens, _ = _parse(pipeline, "The angry technician repaired the pump in Detroit.")
        token_map = {t.idx: t for t in tokens}
        for t in tokens:
            visited = set()
            current = t
            while current.head_idx != current.idx:
                assert current.idx not in visited, f"Cycle detected at '{current.text}'"
                visited.add(current.idx)
                current = token_map[current.head_idx]
            assert current.dep == "ROOT", f"Token '{t.text}' doesn't reach ROOT"
