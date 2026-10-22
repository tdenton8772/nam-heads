"""Tests for post-codebook compound noun meaningfulness filter."""

import pytest
from nam_heads.impls.entity.default.codebook.compound_filter import (
    is_meaningful_compound,
    _COMPOUND_TYPES,
)


class TestPassThroughForNonCompound:
    """Non-compound types should always pass through."""

    def test_person_passes(self):
        assert is_meaningful_compound(
            "albert_einstein", "PROPN_PROPN", "Albert Einstein",
            2, True, entity_type="person",
        )

    def test_proper_noun_passes(self):
        assert is_meaningful_compound(
            "ankara", "PROPN", "Ankara", 1, True, entity_type="proper_noun",
        )

    def test_rejection_generic_noun_still_filtered_by_rules(self):
        """rejection_generic_noun falls through to rules — 'way' fails on generic root."""
        assert not is_meaningful_compound(
            "way", "NOUN", "way", 1, False, entity_type="rejection_generic_noun",
        )

    def test_rejection_generic_noun_passes_for_real_compound(self):
        """rejection_generic_noun cluster is greedy — 'acid_rain' passes all rules."""
        assert is_meaningful_compound(
            "acid_rain", "NOUN_NOUN", "acid rain", 2, False,
            entity_type="rejection_generic_noun",
        )

    def test_rejection_low_salience_is_never_meaningful(self):
        assert not is_meaningful_compound(
            "capital", "NOUN", "capital", 1, False, entity_type="rejection_low_salience",
        )

    def test_none_type_triggers_filter(self):
        """None entity_type should still apply compound rules."""
        # "all_institutions" has generic modifier → rejected
        assert not is_meaningful_compound(
            "all_institutions", "DET_NOUN", "all institutions",
            2, False, entity_type=None,
        )


class TestMeaningfulCompoundsPass:
    """Known meaningful concepts must survive the filter."""

    @pytest.mark.parametrize("norm,pos,text,tc", [
        ("quantum_physics", "NOUN_NOUN", "quantum physics", 2),
        ("civil_war", "ADJ_NOUN", "civil war", 2),
        ("human_rights", "ADJ_NOUN", "human rights", 2),
        ("climate_change", "NOUN_NOUN", "climate change", 2),
        ("nuclear_weapon", "ADJ_NOUN", "nuclear weapon", 2),
        ("solar_system", "ADJ_NOUN", "solar system", 2),
        ("operating_system", "NOUN_NOUN", "operating system", 2),
        ("background_radiation", "NOUN_NOUN", "background radiation", 2),
        ("political_philosophy", "ADJ_NOUN", "political philosophy", 2),
        ("machine_learning", "NOUN_NOUN", "machine learning", 2),
        ("black_hole", "ADJ_NOUN", "black hole", 2),
        ("immune_system", "ADJ_NOUN", "immune system", 2),
    ])
    def test_meaningful_compound_passes(self, norm, pos, text, tc):
        assert is_meaningful_compound(
            norm, pos, text, tc, False, entity_type="compound_noun",
        ), f"{norm} should pass"


class TestNoiseCompoundsRejected:
    """Known noise phrases must be rejected."""

    def test_det_in_pos_rejected(self):
        """'all institutions' (DET_NOUN) → noise."""
        assert not is_meaningful_compound(
            "all_institutions", "DET_NOUN", "all institutions",
            2, False, entity_type="compound_noun",
        )

    def test_det_state_rejected(self):
        """'the state' (DET_NOUN) → noise."""
        assert not is_meaningful_compound(
            "state", "DET_NOUN", "the state",
            2, False, entity_type="compound_noun",
        )

    def test_verb_in_pos_rejected(self):
        """'seeks to abolish' with VERB in POS → noise."""
        assert not is_meaningful_compound(
            "seeks", "VERB_NOUN", "seeks",
            1, False, entity_type="compound_noun",
        )

    def test_generic_modifier_rejected(self):
        """'various forms' → noise (generic modifier)."""
        assert not is_meaningful_compound(
            "various_forms", "ADJ_NOUN", "various forms",
            2, False, entity_type="compound_noun",
        )

    def test_generic_modifier_important(self):
        assert not is_meaningful_compound(
            "important_role", "ADJ_NOUN", "important role",
            2, False, entity_type="compound_noun",
        )

    def test_generic_modifier_significant(self):
        assert not is_meaningful_compound(
            "significant_impact", "ADJ_NOUN", "significant impact",
            2, False, entity_type="compound_noun",
        )

    def test_generic_modifier_large(self):
        assert not is_meaningful_compound(
            "large_number", "ADJ_NOUN", "large number",
            2, False, entity_type="compound_noun",
        )

    def test_generic_root_rejected(self):
        """'replacement' with generic root → noise."""
        assert not is_meaningful_compound(
            "replacement", "DET_NOUN", "the replacement",
            2, False, entity_type="compound_noun",
        )

    def test_verb_stem_modifier_rejected(self):
        """'perpetuate_authority' → verb stem as modifier → noise."""
        assert not is_meaningful_compound(
            "perpetuate_authority", "NOUN_NOUN", "perpetuate authority",
            2, False, entity_type="compound_noun",
        )

    def test_three_content_words_rejected(self):
        """'voluntary_free_associations' (3 content parts) → too long."""
        assert not is_meaningful_compound(
            "voluntary_free_associations", "ADJ_ADJ_NOUN",
            "voluntary free associations",
            3, False, entity_type="compound_noun",
        )

    def test_propn_always_passes(self):
        """Even with generic root, PROPN compounds survive."""
        assert is_meaningful_compound(
            "united_state", "PROPN_PROPN", "United States",
            2, True, entity_type="compound_noun",
        )

    def test_propn_in_pos_pattern_passes(self):
        """Any PROPN token in POS pattern -> meaningful, even if is_propn=False.
        Rule-NLP sets is_propn from root token only ('Base' is NOUN),
        but 'Edwards' is PROPN, so the chunk is a named entity."""
        assert is_meaningful_compound(
            "edwards_air_force_base", "PROPN_PROPN_PROPN_NOUN",
            "Edwards Air Force Base", 4, False, entity_type="compound_noun",
        )


class TestAnarchismRecordEntities:
    """Verify filter produces correct results for the Anarchism record."""

    def test_anarchism_record_entities(self):
        """From the actual Anarchism record NLP output."""
        candidates = [
            # (norm, pos, text, tc, propn, expected_pass)
            ("political_philosophy", "DET_ADJ_NOUN", "political philosophy", 3, False, False),  # DET in POS
            ("seeks", "DET_NOUN", "that seeks", 2, False, False),  # DET in POS
            ("all_institutions", "DET_NOUN", "all institutions", 2, False, False),  # DET
            ("perpetuate_authority", "DET_NOUN_NOUN", "perpetuate authority", 3, False, False),  # DET
            ("state", "DET_NOUN", "the state", 2, False, False),  # DET + generic root
            ("capitalism", "NOUN", "capitalism", 1, False, True),  # single meaningful noun
            ("anarchism_advocates", "NOUN_NOUN", "Anarchism advocates", 2, False, True),  # borderline: both parts are content words
            ("replacement", "DET_NOUN", "the replacement", 2, False, False),  # DET + generic root
            ("stateless_societies", "ADJ_NOUN", "stateless societies", 2, False, True),  # specific adj + noun
            ("voluntary_free_associations", "ADJ_ADJ_NOUN", "voluntary free associations", 3, False, False),  # 3 content parts
        ]

        passed = []
        rejected = []
        for norm, pos, text, tc, propn, expected in candidates:
            result = is_meaningful_compound(
                norm, pos, text, tc, propn, entity_type="compound_noun",
            )
            if result:
                passed.append(norm)
            else:
                rejected.append(norm)
            assert result == expected, \
                f"{norm}: expected {'pass' if expected else 'reject'}, got {'pass' if result else 'reject'}"

        # Verify overall: most noise is filtered
        assert len(rejected) >= 6, f"Expected >=6 rejections, got {len(rejected)}: {rejected}"
        # Only meaningful entities pass: capitalism, anarchism_advocates, stateless_societies
        assert len(passed) <= 4, f"Expected <=4 passes, got {len(passed)}: {passed}"


class TestCompoundTypeCoverage:
    """All compound-like types should trigger the filter."""

    def test_compound_type(self):
        assert "compound" in _COMPOUND_TYPES

    def test_quality_type(self):
        assert "quality" in _COMPOUND_TYPES

    def test_long_description_type(self):
        assert "long_description" in _COMPOUND_TYPES

    def test_compound_noun_type(self):
        assert "compound_noun" in _COMPOUND_TYPES


class TestConceptVerdict:
    """The concept gate's verdict (entity/concepts.py) in the compound filter."""

    def test_not_a_concept_is_never_meaningful(self):
        assert not is_meaningful_compound("quantum_widget", "ADJ_NOUN", "quantum widget", 2, False,
                                          entity_type="compound_noun", concept=False)

    def test_not_a_concept_even_for_a_non_compound_type(self):
        assert not is_meaningful_compound("state_board", "NOUN_NOUN", "state boards", 2, False,
                                          entity_type="organization", concept=False)

    def test_a_concept_passes_the_word_list_rules(self):
        # "high" is a generic modifier, three words is a description: a seed
        # term overrides both.
        assert not is_meaningful_compound("high_school", "ADJ_NOUN", "high school", 2, False,
                                          entity_type="compound_noun")
        assert is_meaningful_compound("high_school", "ADJ_NOUN", "high school", 2, False,
                                      entity_type="compound_noun", concept=True)
        assert is_meaningful_compound("gross_domestic_product", "ADJ_ADJ_NOUN", "gross domestic product",
                                      3, False, entity_type="compound_noun", concept=True)

    def test_a_concept_with_an_article_is_still_a_description(self):
        assert not is_meaningful_compound("high_school", "ADJ_NOUN", "the high school", 2, False,
                                          entity_type="compound_noun", has_article=True, concept=True)

    def test_a_name_ignores_the_verdict(self):
        assert is_meaningful_compound("royal_navy", "PROPN_PROPN", "Royal Navy", 2, True,
                                      entity_type="organization", is_name=True, concept=False)

    def test_no_verdict_is_the_old_rule(self):
        assert is_meaningful_compound("quantum_physics", "ADJ_NOUN", "quantum physics", 2, False,
                                      entity_type="compound_noun", concept=None)
