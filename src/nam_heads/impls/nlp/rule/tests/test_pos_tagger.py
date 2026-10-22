"""Tests for the rule-based POS tagger."""

import pytest
from ..tokenizer import Tokenizer
from ..pos_tagger import POSTagger
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def pipeline():
    tokenizer = Tokenizer()
    tagger = POSTagger(DATA_DIR)
    return tokenizer, tagger


def _tag(pipeline, text):
    tokenizer, tagger = pipeline
    tokens, sents = tokenizer.tokenize(text)
    tagger.tag(tokens, sents)
    return tokens


class TestBasicPOS:
    def test_simple_sentence(self, pipeline):
        tokens = _tag(pipeline, "The cat sat.")
        pos = [(t.text, t.pos) for t in tokens]
        assert pos[0] == ("The", "DET")
        assert pos[1][1] == "NOUN"
        assert pos[2][1] == "VERB"
        assert pos[3] == (".", "PUNCT")

    def test_adjective(self, pipeline):
        tokens = _tag(pipeline, "The big dog ran.")
        big = [t for t in tokens if t.text == "big"][0]
        assert big.pos == "ADJ"

    def test_adverb(self, pipeline):
        tokens = _tag(pipeline, "He ran quickly.")
        quickly = [t for t in tokens if t.text == "quickly"][0]
        assert quickly.pos == "ADV"

    def test_proper_noun(self, pipeline):
        tokens = _tag(pipeline, "I saw Detroit yesterday.")
        detroit = [t for t in tokens if t.text == "Detroit"][0]
        assert detroit.pos == "PROPN"

    def test_number(self, pipeline):
        tokens = _tag(pipeline, "There are 42 apples.")
        num = [t for t in tokens if t.text == "42"][0]
        assert num.pos == "NUM"

    def test_preposition(self, pipeline):
        tokens = _tag(pipeline, "The cat sat on the mat.")
        on = [t for t in tokens if t.text == "on"][0]
        assert on.pos == "ADP"

    def test_conjunction(self, pipeline):
        tokens = _tag(pipeline, "The cat and the dog.")
        and_tok = [t for t in tokens if t.text == "and"][0]
        assert and_tok.pos == "CCONJ"


class TestFineTags:
    def test_vbn_after_be(self, pipeline):
        tokens = _tag(pipeline, "The pump was repaired.")
        repaired = [t for t in tokens if t.text == "repaired"][0]
        assert repaired.tag == "VBN"

    def test_vbg(self, pipeline):
        tokens = _tag(pipeline, "He is running.")
        running = [t for t in tokens if t.text == "running"][0]
        assert running.tag == "VBG"

    def test_vbz(self, pipeline):
        tokens = _tag(pipeline, "She walks home.")
        walks = [t for t in tokens if t.text == "walks"][0]
        assert walks.tag == "VBZ"

    def test_jj(self, pipeline):
        tokens = _tag(pipeline, "The big dog.")
        big = [t for t in tokens if t.text == "big"][0]
        assert big.tag == "JJ"

    def test_nn(self, pipeline):
        tokens = _tag(pipeline, "The dog barked.")
        dog = [t for t in tokens if t.text == "dog"][0]
        assert dog.tag == "NN"


class TestContextDisambiguation:
    def test_verb_after_det_becomes_noun(self, pipeline):
        """After DET, 'run' should become NOUN (e.g. 'the run')."""
        tokens = _tag(pipeline, "The run was long.")
        run = [t for t in tokens if t.text == "run"][0]
        assert run.pos == "NOUN"

    def test_participle_after_det_stays_verb(self, pipeline):
        """After DET, 'broken' (VBN) should stay VERB."""
        tokens = _tag(pipeline, "The broken glass.")
        broken = [t for t in tokens if t.text == "broken"][0]
        assert broken.pos == "VERB"
        assert broken.tag == "VBN"


class TestStopwords:
    def test_common_stopwords(self, pipeline):
        tokens = _tag(pipeline, "The cat is on the mat.")
        the_tok = [t for t in tokens if t.text == "The"][0]
        assert the_tok.is_stop
        is_tok = [t for t in tokens if t.text == "is"][0]
        assert is_tok.is_stop

    def test_content_words_not_stop(self, pipeline):
        tokens = _tag(pipeline, "The cat sat.")
        cat = [t for t in tokens if t.text == "cat"][0]
        assert not cat.is_stop
