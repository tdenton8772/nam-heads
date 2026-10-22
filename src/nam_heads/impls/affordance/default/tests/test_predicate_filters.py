"""Discourse-frame firewall, idioms, aspectual yield across an object and the
extraction gates (predicates.py), on the rule NLP backend (shipped path).

Each case pairs a framing / non-predicate use with the predicate use of the
same verb, so a filter that drops a verb everywhere fails here.
"""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.affordance.default import predicates as P
from nam_heads.impls.affordance.default.config import DEFAULT_CONFIG
from nam_heads.impls.affordance.default.predicates import extract_predicates


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def values(nlp, text, **kw):
    return [p.value for p in extract_predicates(nlp.parse(text), **kw)]


# ---- discourse frame: always ------------------------------------------------

@pytest.mark.parametrize("text,verb", [
    ("For five years, this award highlighted the global artists.", "highlight"),
    ("If you're wondering who Brigham Young was, he was a religious leader.", "wonder"),
    ("The film reflects the mood of the era.", "reflect"),
    ("His career reminds us of the importance of hard work.", "remind"),
    ("The language embodies more than just communication.", "embody"),
])
def test_frame_verbs_always_dropped(nlp, text, verb):
    assert verb not in values(nlp, text)


def test_frame_verbs_come_from_config(nlp):
    t = "For five years, this award highlighted the global artists."
    assert "highlight" in DEFAULT_CONFIG["frame_verbs"]
    assert "highlight" in values(nlp, t, frame=(frozenset(), frozenset()))


# ---- discourse frame: positional ---------------------------------------------

@pytest.mark.parametrize("frame_text,pred_text,verb", [
    # ", V-ing" sentence adjunct
    ("NASA launched Apollo 8, marking the first time humans left orbit.",
     "The river marked the border between the two countries.", "mark"),
    # a span of time as subject
    ("April 2 has witnessed pivotal moments in history.",
     "He witnessed the signing of the treaty.", "witness"),
    ("The 1990s witnessed a surge in crime.",
     "He witnessed the signing of the treaty.", "witness"),
    # present tense with an "a/an" or wh-clause object
    ("The county represents a fascinating intersection of cultures.",
     "He represented Ohio in the Senate.", "represent"),
    ("The constituency illustrates how urban politics works.",
     "He illustrated the book in 1950.", "illustrate"),
    # copular remain / stand
    ("The company remains a celebrated name in pottery.",
     "He remained in Paris until 1920.", "remain"),
    ("Today, it stands as the most populous city.",
     "He stood trial in 1950.", "stand"),
    # the reader as subject
    ("We explore the city.", "Explorers explore the city.", "explore"),
])
def test_positional_frame_verbs(nlp, frame_text, pred_text, verb):
    assert verb not in values(nlp, frame_text)
    assert verb in values(nlp, pred_text)


def test_story_begins_is_frame(nlp):
    assert "begin" not in values(nlp, "The story of the manuscript begins in Iceland.")


def test_frame_position_function(nlp):
    d = nlp.parse("NASA launched Apollo 8 , marking the first time .")
    i = next(t.i for t in d if t.text == "marking")
    assert P.frame_position(d, i, 0, len(d))
    d = nlp.parse("The river marked the border .")
    i = next(t.i for t in d if t.text == "marked")
    assert not P.frame_position(d, i, 0, len(d))


# ---- idioms ------------------------------------------------------------------

@pytest.mark.parametrize("text,verb", [
    ("The ongoing research sheds light on how pests spread.", "shed"),
    ("He played a key role in the war.", "play"),
    ("It shows the profound role language plays in sustaining history.", "play"),
    ("The Aegean Sea holds a special place in history.", "hold"),
    ("Their dedication bore fruit as they claimed the title.", "bear"),
])
def test_idioms_without_value(nlp, text, verb):
    assert verb not in values(nlp, text)


def test_idioms_with_value(nlp):
    assert values(nlp, "He laid the groundwork for universities like BYU.") == ["establish"]
    assert values(nlp, "The creators brought a richly developed character to life.") == ["create"]


def test_idiom_verb_with_other_object_is_a_predicate(nlp):
    assert "play" in values(nlp, "Ford played the role of Han Solo.")
    assert "play" in values(nlp, "She played the piano in the band.")
    assert "hold" in values(nlp, "He held the title until 1990.")


def test_idiom_kind_is_recorded(nlp):
    p = extract_predicates(nlp.parse("He laid the groundwork for universities."))[0]
    assert (p.value, p.lemma, p.kind) == ("establish", "lay", "idiom")


# ---- aspectual yield across an object ---------------------------------------

@pytest.mark.parametrize("text,gone,kept", [
    ("Rocher aimed to provide local employment.", "aim", "provide"),
    ("This concept allowed researchers to calculate probabilities.", "allow", None),
    ("The crisis led the military to intervene.", "lead", "intervene"),
    ("The study helps us appreciate the web of life.", "help", None),
    ("His leadership helped shape the community.", "help", None),
    ("Pests continue to impact commercial crops.", "continue", None),
])
def test_aspectual_yields_across_object(nlp, text, gone, kept):
    v = values(nlp, text)
    assert gone not in v
    if kept:
        assert kept in v


def test_aspectual_verb_without_governed_verb_is_kept(nlp):
    assert "lead" in values(nlp, "He led the army to victory.")
    assert "enable" in values(nlp, "The method enables the partitioning of data.")
    assert "help" in values(nlp, "The Red Cross helped refugees in 1950.")


# ---- extraction gates --------------------------------------------------------

def test_base_form_after_noun_with_object_is_a_verb(nlp):
    assert "accept" in values(nlp, "Courts accept them as evidence.")
    assert "determine" in values(nlp, "Protons determine the atomic mass of an element.")
    assert "ensure" in values(nlp, "Organizations like the Centre ensure that generations remember.")


def test_base_form_after_determiner_or_possessive_is_a_noun(nlp):
    assert values(nlp, "They obtained a construction permit.") == ["obtain"]
    v = values(nlp, "The study deepens our grasp of diversity.")
    assert "grasp" not in v and "understand" not in v


def test_gerund_after_noun_with_object_is_a_verb(nlp):
    assert "affect" in values(nlp, "The disorders affecting voluntary muscles are rare.")
    assert "plan" not in values(nlp, "After a year of planning, the team won.")


@pytest.mark.parametrize("text,verb", [
    ("Botswana is a landlocked country.", "landlock"),
    ("His enduring presence inspired fans.", "endure"),
    ("Their most celebrated moment came in 2005.", "celebrate"),
    ("Reflexes differ from learned actions.", "learn"),
])
def test_attributive_participles_are_not_predicates(nlp, text, verb):
    assert verb not in values(nlp, text)


def test_participle_after_relative_word_is_a_verb(nlp):
    assert "gain" in values(nlp, "It is a book that quickly gained worldwide acclaim.")


def test_base_check_ignores_the_synonym_fold(nlp):
    """The noun gate tests the surface against its base, not its folded
    value: "analyze" folds to "study", and a plural subject keeps it a verb."""
    assert "study" in values(nlp, "Researchers analyze how electrons shift.")


# ---- spelling ----------------------------------------------------------------

def test_british_our_spelling_folds(nlp):
    assert P.resolve_base("honoured")[0] == "honor"
    assert "honor" in values(nlp, "The car was honoured in 1920.")


def test_deterministic(nlp):
    t = ("NASA launched Apollo 8, marking the first time humans left orbit. "
         "The study helps us appreciate the web of life. He laid the groundwork.")
    assert values(nlp, t) == values(nlp, t)
