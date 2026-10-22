"""Tests for the WordNet-generated verb synonym dictionary and category preseeding."""

import json
import os
import pytest

from nam_heads.impls.affordance.default.definitions import VERB_SYNONYMS, VERB_CATEGORIES


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SYNONYM_PATH = os.path.join(DATA_DIR, "verb_synonyms.json")


class TestSynonymDictLoading:
    """Verify the synonym dict loads and has expected structure."""

    def test_dict_loads(self):
        assert isinstance(VERB_SYNONYMS, dict)
        assert len(VERB_SYNONYMS) > 1000, f"Expected >1000 entries, got {len(VERB_SYNONYMS)}"

    def test_json_file_exists(self):
        assert os.path.exists(SYNONYM_PATH), f"Missing {SYNONYM_PATH}"

    def test_no_self_mappings(self):
        for k, v in VERB_SYNONYMS.items():
            assert k != v, f"Self-mapping found: {k} -> {v}"

    def test_all_strings(self):
        for k, v in VERB_SYNONYMS.items():
            assert isinstance(k, str), f"Non-string key: {k}"
            assert isinstance(v, str), f"Non-string value: {v}"

    def test_no_multi_word_entries(self):
        for k, v in VERB_SYNONYMS.items():
            assert " " not in k, f"Multi-word key: {k}"
            assert " " not in v, f"Multi-word value: {v}"


class TestCuratedMappings:
    """Verify curated cross-synset merges are present."""

    @pytest.mark.parametrize("variant,canonical", [
        ("hop", "jump"),
        ("sprint", "run"),
        ("stroll", "walk"),
        ("construct", "build"),
        ("smash", "break"),
        ("yell", "shout"),
        ("ponder", "think"),
        ("comprehend", "understand"),
        ("devour", "eat"),
        ("toss", "throw"),
        ("shove", "push"),
        ("sip", "drink"),
        ("haul", "pull"),
        ("slice", "cut"),
        ("punch", "hit"),
    ])
    def test_curated_mapping(self, variant, canonical):
        assert VERB_SYNONYMS.get(variant) == canonical, (
            f"Expected {variant} -> {canonical}, got {VERB_SYNONYMS.get(variant, variant)}"
        )


class TestProtectedVerbs:
    """Common verbs must NOT be remapped."""

    @pytest.mark.parametrize("verb", [
        "break", "build", "run", "walk", "jump", "fly", "eat", "drink",
        "think", "talk", "hit", "cut", "pull", "push", "throw",
        "start", "stop", "open", "close", "move", "turn",
        "hold", "keep", "find", "show", "tell", "say",
        "write", "read", "play", "work", "live", "die",
        "send", "bring", "buy", "sell", "carry", "catch",
        "teach", "learn", "fight", "cook", "wash", "clean",
        "save", "lead", "follow", "meet", "join", "use",
        "change", "fix", "create", "attack", "defend", "protect",
        "drive", "ride", "climb", "swim", "fire", "shoot",
        "trigger", "launch", "slide", "twist", "design",
    ])
    def test_common_verb_not_remapped(self, verb):
        assert verb not in VERB_SYNONYMS, (
            f"Protected verb '{verb}' is remapped to '{VERB_SYNONYMS[verb]}'"
        )


class TestNoChainDrift:
    """Ensure no transitive chains cause drift to unrelated verbs."""

    def test_no_deep_chains(self):
        """Following a mapping should not lead to a verb that is also mapped."""
        chain_issues = []
        for k, v in VERB_SYNONYMS.items():
            if v in VERB_SYNONYMS:
                chain_issues.append(f"{k} -> {v} -> {VERB_SYNONYMS[v]}")
        assert not chain_issues, (
            f"Found {len(chain_issues)} chain mappings:\n"
            + "\n".join(chain_issues[:10])
        )


class TestCategoryPreseeding:
    """Verify WordNet-generated categories are merged into VERB_CATEGORIES."""

    def test_expanded_beyond_seed(self):
        """VERB_CATEGORIES should have >1000 verbs after merging generated categories."""
        total = sum(len(v) for v in VERB_CATEGORIES.values())
        assert total > 1000, f"Expected >1000 seeded verbs, got {total}"

    def test_original_seed_preserved(self):
        """Hand-curated seed verbs are still present."""
        assert "build" in VERB_CATEGORIES["physical"]
        assert "think" in VERB_CATEGORIES["cognitive"]
        assert "speak" in VERB_CATEGORIES["communicative"]
        assert "govern" in VERB_CATEGORIES["institutional"]
        assert "create" in VERB_CATEGORIES["creative"]
        assert "travel" in VERB_CATEGORIES["movement"]
        assert "change" in VERB_CATEGORIES["transformative"]

    def test_generated_categories_present(self):
        """Auto-classified verbs from WordNet are present."""
        assert "accelerate" in VERB_CATEGORIES["transformative"]
        assert "absorb" in VERB_CATEGORIES["transformative"]

    def test_all_seven_categories(self):
        """All 7 categories exist and have entries."""
        expected = {"physical", "cognitive", "communicative", "institutional",
                    "creative", "movement", "transformative"}
        assert set(VERB_CATEGORIES.keys()) == expected
        for cat in expected:
            assert len(VERB_CATEGORIES[cat]) > 0

    def test_synonym_targets_have_categories(self):
        """Most synonym canonical forms should have a category assignment."""
        all_categorized = set()
        for verbs in VERB_CATEGORIES.values():
            all_categorized |= verbs
        canonicals = set(VERB_SYNONYMS.values())
        covered = canonicals & all_categorized
        coverage = len(covered) / len(canonicals) if canonicals else 0
        # The sense-preserving fold keeps fewer, more common canonicals
        # (715); verb_categories.json is unchanged, covering 48% of them.
        assert coverage > 0.45, (
            f"Only {coverage:.0%} of synonym canonicals have categories "
            f"({len(covered)}/{len(canonicals)})"
        )


class TestMeaningChangingFoldsRemoved:
    """Folds the 2026-10-07 hand audit found changing the meaning in
    encyclopedic text (folds firing on 5,000 wiki records): the verb stays
    itself."""

    @pytest.mark.parametrize("verb", [
        "ensure", "intertwine", "educate", "train", "conquer", "border",
        "encompass", "revere", "popularize", "illuminate", "prescribe", "game",
        "finish", "regain", "advocate", "excel", "destine", "spring", "react",
        "ordain", "evoke", "recall", "steep", "piece", "number", "fleet",
        "soar", "amplify", "march", "storm", "sanction", "vacate", "fabricate",
        "dash",
    ])
    def test_not_folded(self, verb):
        assert verb not in VERB_SYNONYMS, f"{verb} -> {VERB_SYNONYMS.get(verb)}"

    @pytest.mark.parametrize("variant,canonical", [
        ("found", "establish"), ("defeat", "overcome"), ("emphasize", "stress"),
        ("construct", "build"), ("select", "choose"), ("purchase", "buy"),
        ("instruct", "teach"), ("dedicate", "devote"), ("originate", "arise"),
    ])
    def test_shared_sense_folds_kept(self, variant, canonical):
        assert VERB_SYNONYMS.get(variant) == canonical


class TestBuildRules:
    def test_wilson_lower(self):
        from nam_heads.impls.affordance.default.build_synonyms import wilson_lower
        assert wilson_lower(0, 0) == 0.0
        assert wilson_lower(2, 3) < 0.5          # revere: 2 of 3 is no evidence
        assert wilson_lower(3, 5) < 0.5          # border: 3 of 5
        assert wilson_lower(9, 13) >= 0.5        # found: 9 of 13
        assert wilson_lower(26, 27) > 0.8

    def test_unfolded_words_carry_forward(self):
        from nam_heads.impls.affordance.default.build_synonyms import unfolded_words
        prev = {"ensure": "assure", "hop": "jump"}
        new = {"hop": "jump"}
        assert unfolded_words([], prev, new) == ["ensure"]
        # a word stays on the list until it folds again
        assert unfolded_words(["ensure"], new, new) == ["ensure"]
        assert unfolded_words(["ensure"], new, {"hop": "jump", "ensure": "secure"}) == []

    def test_unfolded_file_matches_the_table(self):
        from nam_heads.impls.affordance.default.build_synonyms import UNFOLDED_FILE
        with open(os.path.join(DATA_DIR, UNFOLDED_FILE)) as f:
            unfolded = json.load(f)
        assert unfolded == sorted(unfolded)
        assert not set(unfolded) & set(VERB_SYNONYMS)
        assert {"ensure", "educate", "train", "conquer"} <= set(unfolded)

    def test_shipped_table_is_the_build_output(self):
        """The shipped table is what build_synonyms produces (WordNet 3.0)."""
        nltk = pytest.importorskip("nltk")
        try:
            from nltk.corpus import wordnet as wn
            wn.synsets("test")
        except LookupError:
            pytest.skip("wordnet corpus not installed")
        from nam_heads.impls.affordance.default.build_synonyms import build_synonym_dict
        assert build_synonym_dict() == VERB_SYNONYMS
