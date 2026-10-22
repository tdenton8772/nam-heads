"""Predicate extraction for the affordance axis (rule NLP backend, shipped path)."""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.ontology import bindings as B
from nam_heads.impls.affordance.default.predicates import (
    PHRASAL, extract_predicates, normalize_verb, seed_synonyms, verb_bases,
)
from nam_heads.impls.affordance.default.definitions import (
    FUNCTION_VERBS_DEFAULT, ONTOLOGY_FUNCTION_VERBS, ontology_lookup,
)


def extract_profile(label):
    return {"function_verbs": ontology_lookup(ONTOLOGY_FUNCTION_VERBS, label,
                                              FUNCTION_VERBS_DEFAULT)}


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def values(nlp, text, **kw):
    return [p.value for p in extract_predicates(nlp.parse(text), **kw)]


# ---- lemma repair ----------------------------------------------------------

@pytest.mark.parametrize("text,rule_lemma,want", [
    ("called", "cal", "call"),
    ("joined", "joine", "join"),
    ("changed", "chang", "change"),
    ("became", "became", "become"),
    ("defeated", "defeate", "overcome"),   # fold: shared dominant sense
    ("licensed", "licens", "license"),
    ("added", "ad", "add"),
    ("planned", "plann", "plan"),
    ("travelled", "travell", "travel"),
    ("leaves", "leaf", "leave"),
    ("born", "bear", "bear"),
    ("founded", "found", "establish"),    # fold: found = establish
    ("found", "find", "find"),
    ("organised", "organis", "organize"),
    ("studied", "study", "study"),
])
def test_normalize_repairs_rule_lemmatizer(text, rule_lemma, want):
    assert normalize_verb(text, rule_lemma) == want


def test_normalize_unknown_word_falls_back_to_rule_lemma():
    assert normalize_verb("zxqvled", "zxqvl") == "zxqvl"


def test_shared_fold_is_sense_preserving():
    """The predicate path uses the shared fold (definitions.VERB_SYNONYMS),
    rebuilt so distinct meanings stay distinct."""
    syn = seed_synonyms()
    for wrong in ("consist", "earn", "gain", "derive", "celebrate", "observe",
                  "base", "contain", "engage", "promote", "race", "enrol"):
        assert wrong not in syn or syn[wrong] not in (
            "brood", "deduce", "detect", "integrate", "prosecute", "encourage", "rush", "recruit"), wrong
    assert normalize_verb("constructed", "construct") == "build"    # curated pair
    assert normalize_verb("purchased", "purchase") == "buy"
    assert normalize_verb("consists", "consist") == "consist"
    assert normalize_verb("earned", "earn") == "earn"
    assert normalize_verb("based", "base") == "base"
    assert all(" " not in v and "_" not in v for v in syn.values())


def test_phrasal_read_before_fold(nlp):
    # pick -> choose in the fold, but "pick up" is its own value only if
    # seeded/learned; the fold must not turn the particle reading into
    # "choose": the plain verb folds, the phrasal check sees "pick".
    assert values(nlp, "She picked the winner.")[:1] == ["choose"]


# ---- extraction ------------------------------------------------------------

def test_encyclopedic_predicates_survive(nlp):
    v = values(nlp, "The company was founded in 1998. He led the army and won the battle.")
    assert "establish" in v      # "founded"; legacy light_verbs listed "found"
    assert "lead" in v           # legacy IT firewall listed "lead"
    assert "win" in v


def test_proper_name_verb_tokens_skipped(nlp):
    v = values(nlp, "He moved to the United States in 1990.")
    assert "unite" not in v and "move" in v


def test_phrasal_values(nlp):
    assert "know_as" in values(nlp, "The town is also known as Springfield.")
    assert "grow_up" in values(nlp, "She grew up in Ohio.")
    assert "name_after" in values(nlp, "The street was named after his father.")


def test_plain_light_verbs_skipped(nlp):
    v = values(nlp, "They made a decision and took the train.")
    assert "make" not in v and "take" not in v


def test_aspectual_yields_to_governed_verb(nlp):
    v = values(nlp, "He began playing football in 1990.")
    assert "begin" not in v and "play" in v


def test_function_verb_yields(nlp):
    v = values(nlp, "Aspirin is used to treat pain.")
    assert "treat" in v and "use" not in v


def test_copular_values(nlp):
    assert values(nlp, "PROSITE is a protein database.", emit_copular=True)[:1] == ["classify"]
    assert values(nlp, "The river is very long.", emit_copular=True)[:1] == ["describe"]
    assert "classify" not in values(nlp, "PROSITE is a protein database.", emit_copular=False)


def test_copular_values_off_by_default(nlp):
    assert values(nlp, "PROSITE is a protein database.") == []
    assert values(nlp, "The river is very long.") == []


def test_passive_auxiliary_is_not_copular(nlp):
    v = values(nlp, "The bridge was built in 1890.")
    assert v == ["build"]


def test_nominal_gerund_skipped(nlp):
    v = values(nlp, "After a year of planning, the team won.")
    assert "plan" not in v and "win" in v


def test_found_after_do_support(nlp):
    # resolves to the verb found (folded to establish), not find (light)
    assert "establish" in values(nlp, "What did Larry Page found?")


def test_token_indices_point_at_the_verb(nlp):
    d = nlp.parse("The company was founded in 1998.")
    p = extract_predicates(d)[0]
    assert d[p.token_i].text == "founded" and p.value == "establish" and p.lemma == "found"


def test_deterministic(nlp):
    t = "Lurçat was born in Bruyères, studied at the École in Nancy and worked in Paris."
    assert values(nlp, t) == values(nlp, t)


def test_empty_and_none(nlp):
    assert extract_predicates(None) == []
    assert values(nlp, "Paris, France.") == []


# ---- ontology-conditioned extraction ----------------------------------------

def test_artifact_profile_turns_design_into_function(nlp):
    t = "The valve was designed to prevent backflow."
    art = extract_profile(B.MATERIAL_ARTIFACT)
    v_art = values(nlp, t, sentence_profile=lambda si: art)
    v_def = values(nlp, t)
    assert "prevent" in v_art and "design" not in v_art
    assert "design" in v_def                 # for an agent, designing is the act


def test_function_verb_needs_to_or_for(nlp):
    art = extract_profile(B.MATERIAL_ARTIFACT)
    v = values(nlp, "The sonata was constructed using sonata form.", sentence_profile=lambda si: art)
    assert "build" in v


def test_light_verbs_come_from_config(nlp):
    t = "They made a decision."
    assert "make" not in values(nlp, t)
    assert "make" in values(nlp, t, light=frozenset())


def test_founded_is_not_light(nlp):
    # config light_verbs no longer lists "found" (lemma of "founded").
    v = values(nlp, "The company was founded in 1998.")
    assert v == ["establish"]


def test_adjective_after_be_is_copular(nlp):
    # the rule tagger tags "red"/"unincorporated" VERB; neither is a verb form
    assert values(nlp, "The pump is red.", emit_copular=True) == ["describe"]
    assert values(nlp, "The village is unincorporated.", emit_copular=True) == ["describe"]
    assert values(nlp, "The bridge was built in 1890.") == ["build"]


def test_adjective_after_be_is_no_verb_without_copular(nlp):
    # copular values off: the tagger's VERB adjective is still not a verb
    assert values(nlp, "The pump is red.", emit_copular=False) == []
    assert values(nlp, "The village is unincorporated.", emit_copular=False) == []
