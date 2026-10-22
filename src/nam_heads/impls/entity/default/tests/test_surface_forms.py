"""Tests for EntityEncoder._surface_forms_with_aliases() — stop words, caps, expansion."""

import os
import pytest
from unittest.mock import patch
from ..head import EntityEncoder


@pytest.fixture
def encoder():
    """Create an EntityEncoder with mocked initialization."""
    with patch.object(EntityEncoder, "_initialize"):
        enc = EntityEncoder.__new__(EntityEncoder)
        enc.config = {}
        enc.head_type = "entity"
        return enc


class TestDefaultNoComponentExpansion:
    """By default (NAM_ENTITY_COMPONENT_EXPANSION=0), single-word components are NOT stored."""

    def test_person_no_components(self, encoder):
        forms = encoder._surface_forms_with_aliases("albert_einstein", entity_type="person")
        assert "albert_einstein" in forms
        assert "albert" not in forms
        assert "einstein" not in forms

    def test_org_no_components(self, encoder):
        forms = encoder._surface_forms_with_aliases("united_nations", entity_type="organization")
        assert "united_nations" in forms
        assert "united" not in forms
        assert "nations" not in forms

    def test_first_last_shortening_still_works(self, encoder):
        """First-last shortening is separate from component expansion."""
        forms = encoder._surface_forms_with_aliases("martin_luther_king", entity_type="person")
        assert "martin_luther_king" in forms
        assert "martin_king" in forms
        assert "martin" not in forms
        assert "king" not in forms


class TestStopWordFiltering:
    """Stop words should be filtered from component expansion (when enabled)."""

    def test_stop_words_filtered(self, encoder):
        """'winnie_the_pooh' (person) -> 'the' should not appear in forms."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "winnie_the_pooh", entity_type="person",
            )
        assert "the" not in forms
        assert "winnie" in forms
        assert "pooh" in forms

    def test_stop_word_first_filtered(self, encoder):
        """'the_beatles' (group) -> 'the' filtered out."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "the_beatles", entity_type="group",
            )
        assert "the" not in forms
        assert "beatles" in forms

    def test_born_filtered(self, encoder):
        """'born' is a stop word and should be filtered from components."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "first_known_species", entity_type="person",
            )
        assert "first" not in forms
        assert "known" not in forms

    def test_short_parts_still_filtered(self, encoder):
        """Parts shorter than 3 chars are filtered by length, not stop words."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "jo_smith", entity_type="person",
            )
        # "jo" is < 3 chars, filtered by length
        assert "jo" not in forms
        assert "smith" in forms


class TestSurfaceFormCap:
    """MAX_SURFACE_FORMS=5 prevents pathological expansion."""

    def test_surface_form_cap(self, encoder):
        """9-word entity name -> max 5 forms total."""
        forms = encoder._surface_forms_with_aliases(
            "alpha_bravo_charlie_delta_echo_foxtrot_golf_hotel_india",
            entity_type="person",
        )
        assert len(forms) <= encoder.MAX_SURFACE_FORMS

    def test_cap_preserves_canonical(self, encoder):
        """The canonical form should always be the first entry."""
        name = "alpha_bravo_charlie_delta_echo_foxtrot_golf_hotel_india"
        forms = encoder._surface_forms_with_aliases(
            name, entity_type="person",
        )
        assert forms[0] == name

    def test_cap_with_extra_aliases(self, encoder):
        """Extra aliases + expansion should still be capped."""
        forms = encoder._surface_forms_with_aliases(
            "alpha_bravo_charlie_delta_echo_foxtrot",
            extra_aliases=["alt_form"],
            entity_type="person",
        )
        assert len(forms) <= encoder.MAX_SURFACE_FORMS


class TestObjectNoExpansion:
    """Object-typed entities should not get component expansion."""

    def test_object_no_expansion(self, encoder):
        """'control_system' (object) -> only canonical form."""
        forms = encoder._surface_forms_with_aliases(
            "control_system", entity_type="object",
        )
        assert forms == ["control_system"]

    def test_object_with_alias(self, encoder):
        """Object with an extra alias -> canonical + alias only."""
        forms = encoder._surface_forms_with_aliases(
            "control_system",
            extra_aliases=["control_systems"],
            entity_type="object",
        )
        assert forms == ["control_system", "control_systems"]

    def test_location_no_expansion(self, encoder):
        """Locations should not get component expansion either."""
        forms = encoder._surface_forms_with_aliases(
            "new_york", entity_type="location",
        )
        assert forms == ["new_york"]


class TestPersonExpansionWhenEnabled:
    """Person-typed entities get component expansion when NAM_ENTITY_COMPONENT_EXPANSION=1."""

    def test_person_expansion_works(self, encoder):
        """'albert_einstein' (person) -> includes individual components."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "albert_einstein", entity_type="person",
            )
        assert "albert_einstein" in forms
        assert "albert" in forms
        assert "einstein" in forms

    def test_person_three_parts_gets_first_last(self, encoder):
        """'martin_luther_king' (person) -> includes first_last shortening."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "martin_luther_king", entity_type="person",
            )
        assert "martin_luther_king" in forms
        assert "martin_king" in forms
        assert "martin" in forms
        assert "luther" in forms
        assert "king" in forms

    def test_org_expansion_works(self, encoder):
        """Organizations also get component expansion."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "united_nations", entity_type="organization",
            )
        assert "united_nations" in forms
        assert "united" in forms
        assert "nations" in forms

    def test_no_duplicate_forms(self, encoder):
        """Surface forms should not contain duplicates."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "smith_smith", entity_type="person",
            )
        assert len(forms) == len(set(forms))


class TestTitleStrippedSurfaceForms:
    """Title-stripped surface forms enable fuzzy resolution for titled entities."""

    def test_president_lincoln(self, encoder):
        """'president_abraham_lincoln' stores 'abraham_lincoln' as surface form."""
        forms = encoder._surface_forms_with_aliases(
            "president_abraham_lincoln", entity_type="full_name",
        )
        assert "president_abraham_lincoln" in forms
        assert "abraham_lincoln" in forms

    def test_dr_king(self, encoder):
        """'dr_martin_luther_king' stores 'martin_luther_king' as surface form."""
        forms = encoder._surface_forms_with_aliases(
            "dr_martin_luther_king", entity_type="full_name",
        )
        assert "dr_martin_luther_king" in forms
        assert "martin_luther_king" in forms

    def test_general_eisenhower(self, encoder):
        """'general_eisenhower' stores 'eisenhower' as surface form."""
        forms = encoder._surface_forms_with_aliases(
            "general_eisenhower", entity_type="titled_entity",
        )
        assert "general_eisenhower" in forms
        assert "eisenhower" in forms

    def test_vice_president_gore(self, encoder):
        """Multiple leading titles stripped: 'vice_president_gore' → 'gore'."""
        forms = encoder._surface_forms_with_aliases(
            "vice_president_gore", entity_type="titled_entity",
        )
        assert "vice_president_gore" in forms
        assert "gore" in forms

    def test_no_title_no_strip(self, encoder):
        """'albert_einstein' has no title prefix — no extra stripped form."""
        forms = encoder._surface_forms_with_aliases(
            "albert_einstein", entity_type="person",
        )
        assert "albert_einstein" in forms
        # No title-stripped form added (no leading title)
        assert len([f for f in forms if f not in ("albert_einstein", "albert", "einstein")]) == 0

    def test_non_expansion_type_no_strip(self, encoder):
        """Object types don't get title stripping (not in _COMPONENT_EXPANSION_TYPES)."""
        forms = encoder._surface_forms_with_aliases(
            "president_suite", entity_type="object",
        )
        assert forms == ["president_suite"]

    def test_first_last_uses_stripped_name(self, encoder):
        """first_last shortening skips title: 'colonel_james_smith' → 'james_smith' not 'colonel_smith'."""
        forms = encoder._surface_forms_with_aliases(
            "colonel_james_smith", entity_type="full_name",
        )
        assert "james_smith" in forms, f"Expected james_smith, got: {forms}"
        assert "colonel_smith" not in forms
