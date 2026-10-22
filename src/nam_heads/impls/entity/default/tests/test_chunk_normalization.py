"""Tests for noun-chunk-based entity candidate extraction.

Verifies that extract_candidates() uses ParsedDocument.noun_chunks
when available, producing the same normalized forms as the legacy
anchor-expansion path.
"""

import pytest
from ..normalization import (
    extract_candidates,
    normalize_mention,
    EntityCandidate,
)
from nam_heads.sdk import EncoderHead
from src.nlp.interfaces import (
    Token, NounChunk, ParsedDocument, Entity, Sentence,
    _extract_noun_chunks,
)
from nam_heads.impls.nlp.rule import RuleNLPService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token(idx, text, pos, dep, head_idx, **kw):
    return Token(
        text=text, lemma=kw.get("lemma", text.lower()), pos=pos, dep=dep,
        ent_type="", ent_iob="O", head_idx=head_idx, idx=idx,
        is_stop=kw.get("is_stop", False),
        is_punct=kw.get("is_punct", False),
    )


def _make_doc(tokens, entities=None):
    text = " ".join(t.text for t in tokens)
    sent = Sentence(start_token=0, end_token=len(tokens), text=text)
    return ParsedDocument(text=text, tokens=tokens,
                          entities=entities or [], sentences=[sent])


# ---------------------------------------------------------------------------
# Unit tests: _candidates_from_chunks via extract_candidates
# ---------------------------------------------------------------------------

class TestChunkPathBasic:
    def test_single_propn(self):
        """Single proper noun chunk → candidate."""
        tokens = [
            _make_token(0, "Find", "VERB", "ROOT", 0),
            _make_token(1, "Einstein", "PROPN", "dobj", 0),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "einstein" in norms

    def test_compound_propn(self):
        """'George Washington' compound → single candidate."""
        tokens = [
            _make_token(0, "George", "PROPN", "compound", 1),
            _make_token(1, "Washington", "PROPN", "nsubj", 2),
            _make_token(2, "was", "AUX", "ROOT", 2),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "george_washington" in norms

    def test_det_stripped(self):
        """Determiners are stripped: 'the cat' → 'cat'."""
        tokens = [
            _make_token(0, "the", "DET", "det", 1),
            _make_token(1, "cat", "NOUN", "nsubj", 2),
            _make_token(2, "ran", "VERB", "ROOT", 2),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "cat" in norms
        assert not any("the" in n for n in norms)

    def test_compound_noun(self):
        """'navy pilot' compound → single normalized form."""
        tokens = [
            _make_token(0, "a", "DET", "det", 2),
            _make_token(1, "navy", "NOUN", "compound", 2),
            _make_token(2, "pilot", "NOUN", "dobj", 3),
            _make_token(3, "flew", "VERB", "ROOT", 3),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "navy_pilot" in norms

    def test_pronoun_skipped(self):
        """Pronouns should not produce entity candidates."""
        tokens = [
            _make_token(0, "I", "PRON", "nsubj", 1),
            _make_token(1, "run", "VERB", "ROOT", 1),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        assert len(candidates) == 0

    def test_low_salience_skipped(self):
        """Low-salience head nouns like 'thing' are skipped."""
        tokens = [
            _make_token(0, "the", "DET", "det", 1),
            _make_token(1, "thing", "NOUN", "nsubj", 2, lemma="thing"),
            _make_token(2, "broke", "VERB", "ROOT", 2),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        assert len(candidates) == 0


class TestChunkPathScoring:
    def test_propn_higher_score(self):
        """PROPN heads should score higher than NOUN heads."""
        tokens_propn = [
            _make_token(0, "Einstein", "PROPN", "nsubj", 1),
            _make_token(1, "ran", "VERB", "ROOT", 1),
        ]
        tokens_noun = [
            _make_token(0, "cats", "NOUN", "nsubj", 1),
            _make_token(1, "ran", "VERB", "ROOT", 1),
        ]
        doc_propn = _make_doc(tokens_propn)
        doc_noun = _make_doc(tokens_noun)
        cands_propn = extract_candidates(doc_propn.text, spacy_doc=doc_propn)
        cands_noun = extract_candidates(doc_noun.text, spacy_doc=doc_noun)
        assert len(cands_propn) == 1
        assert len(cands_noun) == 1
        assert cands_propn[0].confidence > cands_noun[0].confidence

    def test_multi_token_bonus(self):
        """Multi-token chunks get a score bonus."""
        tokens_single = [
            _make_token(0, "Einstein", "PROPN", "nsubj", 1),
            _make_token(1, "ran", "VERB", "ROOT", 1),
        ]
        tokens_multi = [
            _make_token(0, "Albert", "PROPN", "compound", 1),
            _make_token(1, "Einstein", "PROPN", "nsubj", 2),
            _make_token(2, "ran", "VERB", "ROOT", 2),
        ]
        doc_single = _make_doc(tokens_single)
        doc_multi = _make_doc(tokens_multi)
        cands_single = extract_candidates(doc_single.text, spacy_doc=doc_single)
        cands_multi = extract_candidates(doc_multi.text, spacy_doc=doc_multi)
        assert cands_multi[0].confidence > cands_single[0].confidence


class TestChunkPathDedup:
    def test_no_duplicates(self):
        """Same normalized form from different chunks → deduplicated."""
        tokens = [
            _make_token(0, "Cat", "NOUN", "nsubj", 2),
            _make_token(1, "and", "CCONJ", "cc", 0),
            _make_token(2, "cat", "NOUN", "conj", 0),  # conj of first 'cat'
            _make_token(3, "played", "VERB", "ROOT", 3),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert norms.count("cat") == 1



class TestPropnTailExtraction:
    """PROPN tail extraction: mixed NOUN+PROPN chunks emit the PROPN suffix as a separate candidate."""

    def test_title_prefixed_person(self):
        """'States president Abraham Lincoln' → 'Abraham Lincoln' alone."""
        tokens = [
            _make_token(0, "assassinated", "VERB", "ROOT", 0),
            _make_token(1, "States", "NOUN", "compound", 4),
            _make_token(2, "president", "NOUN", "compound", 4),
            _make_token(3, "Abraham", "PROPN", "compound", 4),
            _make_token(4, "Lincoln", "PROPN", "dobj", 0),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "abraham_lincoln" in norms, f"Expected abraham_lincoln in {norms}"
        # The words leading into the name are not part of any candidate.
        assert not any("lincoln" in n and "president" in n for n in norms)

    def test_all_propn_no_split(self):
        """All-PROPN chunk 'George Washington' should NOT emit a tail."""
        tokens = [
            _make_token(0, "George", "PROPN", "compound", 1),
            _make_token(1, "Washington", "PROPN", "nsubj", 2),
            _make_token(2, "was", "AUX", "ROOT", 2),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        # Only one candidate, no split
        assert norms.count("washington") == 0 or "george_washington" in norms
        assert len([n for n in norms if "washington" in n]) == 1

    def test_all_noun_no_split(self):
        """All-NOUN chunk 'navy pilot' should NOT emit a tail."""
        tokens = [
            _make_token(0, "a", "DET", "det", 2),
            _make_token(1, "navy", "NOUN", "compound", 2),
            _make_token(2, "pilot", "NOUN", "dobj", 3),
            _make_token(3, "flew", "VERB", "ROOT", 3),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "navy_pilot" in norms
        # Only one compound, no split
        assert "pilot" not in norms

    def test_prime_minister_modi(self):
        """'prime minister Narendra Modi' → also emit 'Narendra Modi'."""
        tokens = [
            _make_token(0, "prime", "ADJ", "compound", 3),
            _make_token(1, "minister", "NOUN", "compound", 3),
            _make_token(2, "Narendra", "PROPN", "compound", 3),
            _make_token(3, "Modi", "PROPN", "nsubj", 4),
            _make_token(4, "spoke", "VERB", "ROOT", 4),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "narendra_modi" in norms, f"Expected narendra_modi in {norms}"

    def test_propn_tail_is_propn_flagged(self):
        """PROPN tail candidate should have is_propn=True."""
        tokens = [
            _make_token(0, "assassinated", "VERB", "ROOT", 0),
            _make_token(1, "president", "NOUN", "compound", 3),
            _make_token(2, "Abraham", "PROPN", "compound", 3),
            _make_token(3, "Lincoln", "PROPN", "dobj", 0),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        tail = [c for c in candidates if c.normalized == "abraham_lincoln"]
        assert len(tail) == 1
        assert tail[0].is_propn is True

    def test_single_propn_after_noun(self):
        """Single PROPN after NOUN: 'county Lincoln' → also emit 'Lincoln'."""
        tokens = [
            _make_token(0, "county", "NOUN", "compound", 1),
            _make_token(1, "Lincoln", "PROPN", "nsubj", 2),
            _make_token(2, "grew", "VERB", "ROOT", 2),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "lincoln" in norms, f"Expected lincoln in {norms}"

    def test_non_contiguous_propn_no_split(self):
        """Non-contiguous PROPN: 'Lincoln County Montana' where County breaks the chain."""
        tokens = [
            _make_token(0, "Lincoln", "PROPN", "compound", 2),
            _make_token(1, "County", "NOUN", "compound", 2),
            _make_token(2, "Montana", "PROPN", "nsubj", 3),
            _make_token(3, "grew", "VERB", "ROOT", 3),
        ]
        doc = _make_doc(tokens)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        # Non-contiguous PROPN → no tail (reset at NOUN)
        # Only the full chunk should exist
        assert "lincoln_county_montana" in norms


class TestNormalizeMention:
    def test_snake_case(self):
        assert normalize_mention("George Washington") == "george_washington"

    def test_strip_determiners_manual(self):
        # normalize_mention itself doesn't strip determiners — that's _clean_span_tokens
        assert normalize_mention("the cat") == "the_cat"

    def test_diacritics(self):
        assert normalize_mention("François") == "francois"

    def test_hyphen(self):
        assert normalize_mention("gs-2435 pump") == "gs_2435_pump"


# ---------------------------------------------------------------------------
# Integration: full parser → chunks → candidates
# ---------------------------------------------------------------------------

class TestChunkNormalizationIntegration:
    """Test the full path: rule-based NLP → noun chunks → entity candidates."""

    @pytest.fixture(scope="class")
    def svc(self):
        service = RuleNLPService()
        service.start()
        yield service
        service.stop()

    def test_george_washington(self, svc):
        doc = svc.parse("Where was George Washington born?")
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert any("george" in n and "washington" in n for n in norms), \
            f"Expected george_washington, got: {norms}"

    def test_navy_pilot(self, svc):
        doc = svc.parse("pretend i am a navy pilot in 1952 selected for the mercury program")
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        has_pilot = any("pilot" in n for n in norms)
        has_program = any("program" in n for n in norms)
        assert has_pilot, f"Expected pilot candidate, got: {norms}"
        assert has_program, f"Expected program candidate, got: {norms}"

    def test_microsoft(self, svc):
        doc = svc.parse("who founded Microsoft?")
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert any("microsoft" in n for n in norms), \
            f"Expected microsoft, got: {norms}"

    def test_no_pronoun_candidates(self, svc):
        """Pronouns should never appear as entity candidates."""
        doc = svc.parse("I went to the store")
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        for c in candidates:
            assert c.normalized != "i", f"Pronoun 'I' should not be a candidate"

    def test_candidates_all_have_score(self, svc):
        doc = svc.parse("Albert Einstein developed the theory of relativity")
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        for c in candidates:
            assert c.confidence >= 0.15, f"'{c.text}' has score {c.confidence} < 0.15"

    def test_hash_stability(self, svc):
        """Same input → same normalized form → same entity hash."""
        from ..head import _hash_entity_id
        doc1 = svc.parse("George Washington")
        doc2 = svc.parse("George Washington")
        cands1 = extract_candidates(doc1.text, spacy_doc=doc1)
        cands2 = extract_candidates(doc2.text, spacy_doc=doc2)
        assert len(cands1) > 0
        assert len(cands2) > 0
        hash1 = _hash_entity_id("person", cands1[0].normalized)
        hash2 = _hash_entity_id("person", cands2[0].normalized)
        assert hash1 == hash2

    def test_booth_article_finds_lincoln(self, svc):
        """Booth's article mentions 'president Abraham Lincoln' → extract 'abraham_lincoln'."""
        text = (
            "John Wilkes Booth. John Wilkes Booth was an American stage actor "
            "who assassinated United States president Abraham Lincoln at "
            "Ford's Theatre in Washington, D.C."
        )
        doc = svc.parse(text)
        candidates = extract_candidates(doc.text, spacy_doc=doc)
        norms = [c.normalized for c in candidates]
        assert "abraham_lincoln" in norms, (
            f"Expected 'abraham_lincoln' from Booth article, got: {norms}"
        )
