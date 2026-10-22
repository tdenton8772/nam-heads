"""Tests for query-path entity resolution.

Verifies that _encode_internal / encode_query uses the rule NLP backend
to parse query text when no spacy_doc is provided in context, and that
codebook rejection is skipped for query mode (user intent, not noise).
"""

import pytest
from ..head import EntityEncoder
from ..normalization import extract_candidates
from nam_heads.impls.nlp.rule import RuleNLPService


@pytest.fixture(scope="module")
def svc():
    service = RuleNLPService()
    service.start()
    yield service
    service.stop()


@pytest.fixture(scope="module")
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 20
    enc.create_new_entities = True
    enc.min_confidence = 0.35
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    # Memory-backed store
    from ..storage.factory import create_entity_store
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


class TestParseTextFallback:
    """_parse_text should produce a ParsedDocument with noun chunks."""

    def test_single_word_produces_doc(self, encoder):
        doc = encoder._parse_text("anarchism")
        assert doc is not None
        assert hasattr(doc, "sentences")
        assert len(doc) >= 1

    def test_single_word_produces_candidates(self, encoder):
        """Single lowercase word should produce candidates via rule NLP."""
        doc = encoder._parse_text("anarchism")
        candidates = extract_candidates("anarchism", spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "anarchism" in norms, f"Expected 'anarchism' in candidates, got: {norms}"

    def test_multi_word_query(self, encoder):
        doc = encoder._parse_text("Albert Einstein")
        candidates = extract_candidates("Albert Einstein", spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert any("einstein" in n for n in norms), f"Expected einstein candidate, got: {norms}"

    def test_full_question(self, encoder):
        doc = encoder._parse_text("What is anarchism?")
        candidates = extract_candidates("What is anarchism?", spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert any("anarchism" in n for n in norms), f"Expected anarchism candidate, got: {norms}"


class TestEncodeQueryProducesCandidates:
    """encode_query should find entities even without context['spacy_doc'].

    Query mode skips codebook rejection filtering -- the user's query
    text is their intent, not noise to be filtered.
    """

    def test_single_word_query(self, encoder):
        """'anarchism' query with no context should produce entity addresses."""
        result = encoder.encode_query("anarchism")
        assert len(result.addresses) > 0, (
            f"Expected addresses for 'anarchism', got none. "
            f"metadata={result.metadata}"
        )

    def test_question_query(self, encoder):
        """'What is anarchism?' should produce entity addresses."""
        result = encoder.encode_query("What is anarchism?")
        entity_norms = [a.metadata.get("normalized", "") for a in result.addresses]
        assert any("anarchism" in n for n in entity_norms), (
            f"Expected anarchism entity, got: {entity_norms}"
        )

    def test_proper_noun_query(self, encoder):
        """'Albert Einstein' should produce entity addresses."""
        result = encoder.encode_query("Albert Einstein")
        assert len(result.addresses) > 0, "Expected addresses for 'Albert Einstein'"

    def test_no_context_still_works(self, encoder):
        """Passing context=None should not crash -- rule NLP parses the text."""
        result = encoder.encode_query("Ankara", context=None)
        assert result is not None
        assert result.head_type == "entity"
        assert len(result.addresses) > 0, "Expected addresses for 'Ankara'"

    def test_generic_noun_allowed_in_query(self, encoder):
        """Query mode never rejects — the user typed it intentionally."""
        result = encoder.encode_query("democracy")
        assert len(result.addresses) > 0, \
            "Query mode should allow generic nouns like 'democracy'"


class TestFullTextInjectionGuard:
    """Full-text injection should NOT fire when NLP found a proper noun.

    "assassinated Abraham Lincoln" → NLP extracts "Abraham Lincoln" (PROPN).
    Full-text injection must NOT add "assassinated_abraham_lincoln" as an
    entity — the verb is not part of the entity name.
    """

    def test_verb_plus_propn_no_phantom(self, encoder):
        """'assassinated Abraham Lincoln' should produce Lincoln, not the verb+name."""
        result = encoder.encode_query("assassinated Abraham Lincoln")
        norms = [a.metadata.get("normalized", "") for a in result.addresses]
        assert any("abraham_lincoln" in n for n in norms), (
            f"Expected 'abraham_lincoln' in entities, got: {norms}"
        )
        assert not any("assassinated" in n for n in norms), (
            f"Verb 'assassinated' should not appear in entity names, got: {norms}"
        )

    def test_preposition_plus_propn_no_phantom(self, encoder):
        """'about Albert Einstein' should produce Einstein, not 'about_albert_einstein'."""
        result = encoder.encode_query("about Albert Einstein")
        norms = [a.metadata.get("normalized", "") for a in result.addresses]
        assert not any("about" in n for n in norms), (
            f"Preposition 'about' should not appear in entity names, got: {norms}"
        )

    def test_compound_noun_still_injected(self, encoder):
        """'acid rain' — NLP might only extract 'rain', so full-text should fire."""
        result = encoder.encode_query("acid rain")
        norms = [a.metadata.get("normalized", "") for a in result.addresses]
        assert any("acid_rain" in n or "acid rain" in n for n in norms), (
            f"Compound noun 'acid_rain' should be injected, got: {norms}"
        )


class TestIngestAlsoKeepsGenericNouns:
    """Ingest mode must keep generic nouns so queries can find records by topic."""

    def test_encode_keeps_generic_noun(self, encoder):
        """encode() (ingest mode) should keep generic nouns as entities."""
        result = encoder.encode("anarchism")
        norms = [a.metadata.get("normalized", "") for a in result.addresses]
        assert any("anarchism" in n for n in norms), (
            f"Ingest should keep 'anarchism' as entity, got: {norms}"
        )
