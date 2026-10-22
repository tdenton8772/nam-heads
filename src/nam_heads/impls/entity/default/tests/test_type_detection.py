"""Tests for EntityEncoder._detect_type() — codebook-based entity classification."""

import pytest
from unittest.mock import patch
from ..head import EntityEncoder
from ..codebook.lookup import classify_entity


@pytest.fixture
def encoder():
    """Create an EntityEncoder with mocked initialization."""
    with patch.object(EntityEncoder, "_initialize"):
        enc = EntityEncoder.__new__(EntityEncoder)
        enc.config = {}
        enc.head_type = "entity"
        return enc


class TestCodebookDeterminism:
    """Codebook classification must be deterministic — same input, same output."""

    def test_same_input_same_output(self, encoder):
        t1 = encoder._detect_type("albert_einstein", "PROPN_PROPN", "Albert Einstein",
                                   token_count=2, is_propn=True)
        t2 = encoder._detect_type("albert_einstein", "PROPN_PROPN", "Albert Einstein",
                                   token_count=2, is_propn=True)
        assert t1 == t2

    def test_returns_string(self, encoder):
        result = encoder._detect_type("something")
        assert isinstance(result, str)
        assert len(result) > 0


class TestCodebookClassification:
    """Codebook should produce sensible entity types for known patterns.

    The k=16 codebook has 6 types: compound_noun, event, long_description,
    person, proper_noun, rejection_generic_noun.  Tests accept any valid
    non-rejection label for entity-like inputs, and accept rejection for
    generic nouns.
    """

    NON_REJECTION = {"compound_noun", "event", "long_description",
                     "person", "proper_noun"}

    def test_proper_noun_pair(self, encoder):
        """Two-token proper noun should NOT be rejected."""
        result = encoder._detect_type(
            "albert_einstein", "PROPN_PROPN", "Albert Einstein",
            token_count=2, is_propn=True, confidence=0.6,
        )
        assert result in self.NON_REJECTION

    def test_single_common_noun(self, encoder):
        """Single common noun may be rejected (that is correct behavior)."""
        result = encoder._detect_type(
            "telescope", "NOUN", "telescope",
            token_count=1, is_propn=False,
        )
        # rejection_generic_noun is acceptable here — the whole point of retraining
        assert isinstance(result, str) and len(result) > 0

    def test_acronym(self, encoder):
        """Short all-caps proper noun should NOT be rejected."""
        result = encoder._detect_type(
            "nasa", "PROPN", "NASA",
            token_count=1, is_propn=True,
        )
        assert result in self.NON_REJECTION

    def test_identifier_with_digits(self, encoder):
        """Alphanumeric identifier should get some classification."""
        result = encoder._detect_type(
            "gs_2435", "NOUN", "GS-2435",
            token_count=1, is_propn=False, has_digit=True, has_hyphen=True,
        )
        # With k=16, may land in any non-rejection cluster
        assert isinstance(result, str) and len(result) > 0

    def test_defaults_work(self, encoder):
        """Calling with only normalized (all defaults) should not error."""
        result = encoder._detect_type("something")
        assert isinstance(result, str)


class TestClassifyEntityFunction:
    """Test the module-level classify_entity convenience function."""

    def test_classify_entity_matches_detect_type(self, encoder):
        """classify_entity should return the same result as _detect_type."""
        result_fn = classify_entity(
            normalized="albert_einstein", pos_pattern="PROPN_PROPN",
            text="Albert Einstein", token_count=2, is_propn=True,
        )
        result_method = encoder._detect_type(
            "albert_einstein", "PROPN_PROPN", "Albert Einstein",
            token_count=2, is_propn=True,
        )
        assert result_fn == result_method

    def test_all_defaults(self):
        """classify_entity with all defaults should produce a valid type."""
        result = classify_entity(normalized="word")
        assert isinstance(result, str)
        assert len(result) > 0
