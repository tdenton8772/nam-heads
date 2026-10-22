"""Tests for the rule-based tokenizer."""

import pytest
from ..tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


class TestBasicTokenization:
    def test_simple_sentence(self, tokenizer):
        tokens, sents = tokenizer.tokenize("The cat sat.")
        texts = [t.text for t in tokens]
        assert texts == ["The", "cat", "sat", "."]

    def test_punctuation_flags(self, tokenizer):
        tokens, _ = tokenizer.tokenize("Hello, world!")
        assert not tokens[0].is_punct
        assert tokens[1].is_punct  # comma
        assert not tokens[2].is_punct
        assert tokens[3].is_punct  # exclamation

    def test_number_detection(self, tokenizer):
        tokens, _ = tokenizer.tokenize("I have 42 apples and three oranges.")
        assert tokens[2].like_num  # 42
        assert not tokens[3].like_num  # apples
        assert tokens[5].like_num  # three

    def test_decimal_numbers(self, tokenizer):
        tokens, _ = tokenizer.tokenize("The price is 3.14 dollars.")
        # "3.14" gets split by regex since . is punctuation
        # The number "3" should be like_num
        nums = [t for t in tokens if t.like_num]
        assert len(nums) >= 1

    def test_hyphenated_words(self, tokenizer):
        tokens, _ = tokenizer.tokenize("The well-known scientist arrived.")
        texts = [t.text for t in tokens]
        assert "well-known" in texts


class TestContractions:
    def test_dont(self, tokenizer):
        tokens, _ = tokenizer.tokenize("I don't know.")
        texts = [t.text for t in tokens]
        assert "do" in texts
        assert "n't" in texts

    def test_theyre(self, tokenizer):
        tokens, _ = tokenizer.tokenize("They're coming.")
        texts = [t.text for t in tokens]
        assert "They" in texts
        assert "'re" in texts

    def test_cannot(self, tokenizer):
        tokens, _ = tokenizer.tokenize("I cannot go.")
        texts = [t.text for t in tokens]
        assert "can" in texts
        assert "not" in texts

    def test_ill(self, tokenizer):
        tokens, _ = tokenizer.tokenize("I'll be there.")
        texts = [t.text for t in tokens]
        assert "I" in texts
        assert "'ll" in texts

    def test_possessive_s(self, tokenizer):
        tokens, _ = tokenizer.tokenize("John's book is here.")
        texts = [t.text for t in tokens]
        assert "John" in texts
        assert "'s" in texts


class TestSentenceSplitting:
    def test_two_sentences(self, tokenizer):
        tokens, sents = tokenizer.tokenize("I ran. He walked.")
        assert len(sents) == 2
        assert sents[0]["start"] == 0
        assert sents[1]["text"].startswith("He")

    def test_abbreviation_guard(self, tokenizer):
        tokens, sents = tokenizer.tokenize("Dr. Smith went home.")
        assert len(sents) == 1

    def test_question_mark(self, tokenizer):
        tokens, sents = tokenizer.tokenize("Is it done? Yes it is.")
        assert len(sents) == 2

    def test_exclamation(self, tokenizer):
        tokens, sents = tokenizer.tokenize("Stop! Don't move.")
        # "Stop!" -> "Don" starts with capital -> sentence boundary
        assert len(sents) == 2

    def test_single_sentence(self, tokenizer):
        tokens, sents = tokenizer.tokenize("The cat sat on the mat.")
        assert len(sents) == 1
        assert sents[0]["start"] == 0
        assert sents[0]["end"] == len(tokens)

    def test_empty(self, tokenizer):
        tokens, sents = tokenizer.tokenize("")
        assert tokens == []
        assert sents == []


class TestTokenIndices:
    def test_indices_sequential(self, tokenizer):
        tokens, _ = tokenizer.tokenize("The quick brown fox.")
        for i, t in enumerate(tokens):
            assert t.idx == i

    def test_head_idx_defaults_to_self(self, tokenizer):
        tokens, _ = tokenizer.tokenize("Hello world.")
        for t in tokens:
            assert t.head_idx == t.idx

    def test_ent_iob_defaults_o(self, tokenizer):
        tokens, _ = tokenizer.tokenize("Hello world.")
        for t in tokens:
            assert t.ent_iob == "O"


def test_apostrophe_shortcut_keeps_contraction_splits():
    from nam_heads.impls.nlp.rule import tokenizer as tk
    assert tk._CONTRACTIONS_NEED_APOSTROPHE
    t = tk.Tokenizer()
    assert t._split_raw("We don't know it's Bob's cannot plain") == [
        "We", "do", "n't", "know", "it", "'s", "Bob", "'s", "can", "not", "plain"]
