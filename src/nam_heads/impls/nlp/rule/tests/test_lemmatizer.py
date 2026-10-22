"""Tests for the rule-based lemmatizer."""

import pytest
from ..tokenizer import Tokenizer, RawToken
from ..pos_tagger import POSTagger
from ..lemmatizer import Lemmatizer
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def pipeline():
    tokenizer = Tokenizer()
    tagger = POSTagger(DATA_DIR)
    lemmatizer = Lemmatizer(DATA_DIR)
    return tokenizer, tagger, lemmatizer


def _lemmatize(pipeline, text):
    tokenizer, tagger, lemmatizer = pipeline
    tokens, sents = tokenizer.tokenize(text)
    tagger.tag(tokens, sents)
    lemmatizer.lemmatize(tokens)
    return tokens


class TestIrregularVerbs:
    def test_was(self, pipeline):
        tokens = _lemmatize(pipeline, "He was there.")
        was = [t for t in tokens if t.text == "was"][0]
        assert was.lemma == "be"

    def test_went(self, pipeline):
        tokens = _lemmatize(pipeline, "She went home.")
        went = [t for t in tokens if t.text == "went"][0]
        assert went.lemma == "go"

    def test_took(self, pipeline):
        tokens = _lemmatize(pipeline, "He took it.")
        took = [t for t in tokens if t.text == "took"][0]
        assert took.lemma == "take"


class TestVerbSuffixStripping:
    def test_running(self, pipeline):
        tokens = _lemmatize(pipeline, "He is running fast.")
        running = [t for t in tokens if t.text == "running"][0]
        assert running.lemma == "run"

    def test_walked(self, pipeline):
        tokens = _lemmatize(pipeline, "She walked home.")
        walked = [t for t in tokens if t.text == "walked"][0]
        assert walked.lemma == "walk"

    def test_makes(self, pipeline):
        tokens = _lemmatize(pipeline, "He makes things.")
        makes = [t for t in tokens if t.text == "makes"][0]
        assert makes.lemma == "make"


class TestIrregularNouns:
    def test_men(self, pipeline):
        tokens = _lemmatize(pipeline, "The men arrived.")
        men = [t for t in tokens if t.text == "men"][0]
        assert men.lemma == "man"

    def test_children(self, pipeline):
        tokens = _lemmatize(pipeline, "The children played.")
        children = [t for t in tokens if t.text == "children"][0]
        assert children.lemma == "child"

    def test_mice(self, pipeline):
        tokens = _lemmatize(pipeline, "The mice escaped.")
        mice = [t for t in tokens if t.text == "mice"][0]
        assert mice.lemma == "mouse"


class TestNounSuffixStripping:
    def test_dogs(self, pipeline):
        tokens = _lemmatize(pipeline, "The dogs barked.")
        dogs = [t for t in tokens if t.text == "dogs"][0]
        assert dogs.lemma == "dog"

    def test_churches(self, pipeline):
        tokens = _lemmatize(pipeline, "The churches are old.")
        churches = [t for t in tokens if t.text == "churches"][0]
        assert churches.lemma == "church"

    def test_stories(self, pipeline):
        tokens = _lemmatize(pipeline, "The stories were good.")
        stories = [t for t in tokens if t.text == "stories"][0]
        assert stories.lemma == "story"


class TestAdjectives:
    def test_better(self, pipeline):
        tokens = _lemmatize(pipeline, "This is better.")
        better = [t for t in tokens if t.text == "better"][0]
        assert better.lemma == "good"

    def test_worst(self, pipeline):
        tokens = _lemmatize(pipeline, "This is the worst.")
        worst = [t for t in tokens if t.text == "worst"][0]
        assert worst.lemma == "bad"


class TestOutputAlwaysLowercase:
    def test_capitalized_input(self, pipeline):
        tokens = _lemmatize(pipeline, "The Dogs BARKED Loudly.")
        for t in tokens:
            assert t.lemma == t.lemma.lower(), f"{t.text} -> {t.lemma} not lowercase"


class TestSilentEFixes:
    """VCe restoration should not over-fire for stems ending in -er/-ew/-ow."""

    def test_viewed_to_view(self, pipeline):
        tokens = _lemmatize(pipeline, "He viewed the scene.")
        viewed = [t for t in tokens if t.text == "viewed"][0]
        assert viewed.lemma == "view", f"Expected 'view', got '{viewed.lemma}'"

    def test_pioneered_to_pioneer(self, pipeline):
        tokens = _lemmatize(pipeline, "She pioneered the technique.")
        pioneered = [t for t in tokens if t.text == "pioneered"][0]
        assert pioneered.lemma == "pioneer", f"Expected 'pioneer', got '{pioneered.lemma}'"

    def test_followed_to_follow(self, pipeline):
        tokens = _lemmatize(pipeline, "He followed the path.")
        followed = [t for t in tokens if t.text == "followed"][0]
        assert followed.lemma == "follow", f"Expected 'follow', got '{followed.lemma}'"

    def test_viewing_to_view(self, pipeline):
        tokens = _lemmatize(pipeline, "He was viewing the scene.")
        viewing = [t for t in tokens if t.text == "viewing"][0]
        assert viewing.lemma == "view", f"Expected 'view', got '{viewing.lemma}'"

    def test_following_to_follow(self, pipeline):
        tokens = _lemmatize(pipeline, "He was following the path.")
        following = [t for t in tokens if t.text == "following"][0]
        assert following.lemma == "follow", f"Expected 'follow', got '{following.lemma}'"

    def test_hoped_to_hope(self, pipeline):
        """Regression: VCe should still work for legitimate cases."""
        tokens = _lemmatize(pipeline, "She hoped for rain.")
        hoped = [t for t in tokens if t.text == "hoped"][0]
        assert hoped.lemma == "hope", f"Expected 'hope', got '{hoped.lemma}'"

    def test_named_to_name(self, pipeline):
        """Regression: VCe should still work for legitimate cases."""
        tokens = _lemmatize(pipeline, "They named the child.")
        named = [t for t in tokens if t.text == "named"][0]
        assert named.lemma == "name", f"Expected 'name', got '{named.lemma}'"

    def test_advanced_to_advance(self, pipeline):
        """Exception dict entry for consonant-cluster + silent-e."""
        tokens = _lemmatize(pipeline, "The army advanced quickly.")
        advanced = [t for t in tokens if t.text == "advanced"][0]
        assert advanced.lemma == "advance", f"Expected 'advance', got '{advanced.lemma}'"

    def test_enhanced_to_enhance(self, pipeline):
        """Exception dict entry for consonant-cluster + silent-e."""
        tokens = _lemmatize(pipeline, "They enhanced the system.")
        enhanced = [t for t in tokens if t.text == "enhanced"][0]
        assert enhanced.lemma == "enhance", f"Expected 'enhance', got '{enhanced.lemma}'"

    def test_surfaced_to_surface(self, pipeline):
        """Exception dict entry for consonant-cluster + silent-e."""
        tokens = _lemmatize(pipeline, "Issues surfaced during testing.")
        surfaced = [t for t in tokens if t.text == "surfaced"][0]
        assert surfaced.lemma == "surface", f"Expected 'surface', got '{surfaced.lemma}'"
