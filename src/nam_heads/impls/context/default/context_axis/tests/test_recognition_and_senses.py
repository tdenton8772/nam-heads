"""Context recognition gaps and wrong senses (2026-10-07 audit).

Recognition: domain triggers beyond exact tokens (plurals, two-word
triggers, music / politics / games / comics / economics fields); one value
per period (football, the Renaissance, WWII); "mid-20th century",
"17th-century", three-digit years, named wars, dynasties.
Senses: "Song dynasty" is no music, "musical interests" are music,
"Hindu Kush" is no Hinduism, "modern media" is no modern era.

Every test parses with the rule NLP backend -- the backend production runs.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.context_axis import domains, temporal
from nam_heads.impls.context.default.context_axis.extract import extract_context
from nam_heads.impls.context.default.context_axis.lexicon import Lexicon, candidate_queries, seed_tables
from nam_heads.impls.context.default.context_axis.seed import seed_entries, seed_store
from nam_heads.impls.context.default.storage.memory_store import MemoryContextStore

_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        _STORE = MemoryContextStore()
        seed_store(_STORE)
    return _STORE


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def values(nlp, text, prefix=None):
    d = nlp.parse(text)
    lex = Lexicon(_store().get_batch(sorted(candidate_queries(list(d)))))
    return {v.value for v in extract_context(d, lex) if prefix is None or v.value.startswith(prefix)}


# ---- domain triggers ------------------------------------------------------------
@pytest.mark.parametrize("text,field", [
    ("The scene shaped electronic music for a decade .", "d:music"),
    ("Brussels is a centre for international politics .", "d:politics"),
    ("Several architects designed the square .", "d:architecture"),          # plural
    ("Many universities opened that year .", "d:education"),                 # -ies plural
    ("Board games involve moving pieces on a surface .", "d:games"),          # two-word trigger
    ("It broadened the domain of role-playing games .", "d:games"),
    ("The hero first appeared in Marvel Comics .", "d:comics"),
    ("Economists forecast inflation for next year .", "d:economics"),
    ("He was a philosopher of the Stoic school .", "d:philosophy"),
    ("It is a rare dental tumor of the jaw .", "d:medicine"),
    ("An allele is a version of a gene .", "d:genetics"),
])
def test_domain_triggers(nlp, text, field):
    assert field in values(nlp, text, "d:")


def test_plural_rule_only_for_trigger_singulars():
    assert domains.singular("architects") == "architect"
    assert domains.singular("universities") == "university"
    assert domains.singular("class") is None and domains.singular("bus") is None
    assert domains.singular("artists") is None            # artists: musicians as often as painters
    assert domains.field_of({"church": "christianity"}, "churches") == "christianity"
    assert domains.field_of({"church": "christianity"}, "news") is None


def test_two_word_triggers_are_seeded():
    assert seed_tables().fields.get("board game") == "games"
    assert any(p == "cx:t:board game" for p, _d, _v, _c in seed_entries())


# ---- one value per period / sport ----------------------------------------------
def test_football_is_one_sport_per_document(nlp):
    assert values(nlp, "He played football for the club for a decade .", "d:") == {"d:association_football"}
    assert "d:association_football" in values(nlp, "He was a footballer and later coached football .", "d:")
    v = values(nlp, "He played football as a quarterback in the NFL .", "d:")
    assert "d:american_football" in v and "d:association_football" not in v
    assert "d:american_football" in values(nlp, "American football is popular .", "d:")
    assert not any(x == "d:football" for x in values(nlp, "The football season began .", "d:"))


@pytest.mark.parametrize("text", [
    "The Renaissance changed painting .",
    "The madrigal flourished in the Renaissance .",
    "Art saw a renaissance of interest .",
])
def test_renaissance_is_one_value(nlp, text):
    assert values(nlp, text, "t:") == {"t:E-renaissance"}


def test_periods_resolve_to_canonical_slugs():
    assert temporal.canonical("C20/D1940", "wwii") == ("C20", "world_war_ii")
    assert temporal.canonical("C12", "medieval") == ("", "middle_ages")
    assert temporal.canonical("", "renaissance_era") == ("", "renaissance")
    assert temporal.canonical("C16", "reformation") == ("C16", "reformation")
    assert temporal.canonical("C9", "unknown_era") == ("C9", "unknown_era")


def test_seeded_periods_are_canonical():
    st = seed_tables()
    assert st.periods[("the", "renaissance")] == ("", "renaissance")
    assert st.periods[("second", "world", "war")] == ("C20", "world_war_ii")
    assert st.era_words["renaissance"] == "renaissance"
    assert st.era_words["victorian"] == "victorian_era" and st.era_paths["victorian_era"] == "C19"


# ---- temporal patterns ------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Hollywood stars came during the mid-20th century .", "t:C20"),
    ("It dates to late 17th-century Iceland .", "t:C17"),
    ("The style spread in the mid-nineteenth century .", "t:C19"),
    ("It was made in 5th-century BC Greece .", "t:C-5"),
    ("In 405 , Mesrop Mashtots created the alphabet .", "t:C5/D400/Y405"),
    ("By 802 the city had fallen .", "t:C9/D800/Y802"),
    ("He completed the sphere in either 977 or 979 .", "t:C10/D970/Y979"),
    ("The battle came during the First English Civil War .", "t:C17/E-english_civil_war"),
    ("It was a defeat in the Wars of the Three Kingdoms .", "t:C17/E-wars_of_the_three_kingdoms"),
    ("He fought in the Thirty Years' War .", "t:C17/E-thirty_years_war"),
    ("It ended the Mexican–American War .", "t:C19/D1840/E-mexican_american_war"),
    ("The Football War was brief .", "t:E-football_war"),
    ("He worked during the early Song dynasty .", "t:E-song_dynasty"),
    ("It reflects order in the early modern period .", "t:E-early_modern_period"),
])
def test_temporal_patterns(nlp, text, expected):
    assert expected in values(nlp, text, "t:")


@pytest.mark.parametrize("text", [
    "The club had 405 members .",
    "It grew in 300 cases .",
    "The Star Wars film opened .",
    "He studied the Civil War .",
    "She read The War of the Worlds .",
    "He was Secretary of War .",
    "Its legacy extends into modern media .",
    "The industrial city grew .",
])
def test_temporal_non_periods(nlp, text):
    assert values(nlp, text, "t:") == set()


def test_three_digit_year_rules_on_plain_tokens():
    def toks(*ws, pos=None):
        return [SimpleNamespace(text=w, i=k, pos_=(pos or {}).get(k, "X")) for k, w in enumerate(ws)]
    years = set()
    assert temporal._three_digit_year(toks("in", "405", ","), 1, years)
    assert not temporal._three_digit_year(toks("in", "405", "cases", pos={2: "NOUN"}), 1, years)
    assert not temporal._three_digit_year(toks("had", "405", "."), 1, years)
    assert not temporal._three_digit_year(toks("in", "045", "."), 1, years)
    assert temporal._three_digit_year(toks("977", "or", "979"), 2, {0})


# ---- senses -----------------------------------------------------------------------
def test_song_dynasty_is_no_music(nlp):
    v = values(nlp, "The astronomer worked during the early Song dynasty .")
    assert "t:E-song_dynasty" in v and "d:music" not in v


def test_musical_before_a_noun_is_music(nlp):
    assert values(nlp, "It showed the country's diverse musical interests .", "d:") == {"d:music"}
    assert "d:theatre" in values(nlp, "She starred in a Broadway musical in 1990 .", "d:")
    assert values(nlp, "It holds a place in American musical theater history .", "d:") == {"d:theatre"}


def test_proper_name_headed_by_a_religious_adjective(nlp):
    assert "d:buddhism_hinduism" not in values(nlp, "The central Hindu Kush range splits the land .", "d:")
    assert "d:buddhism_hinduism" in values(nlp, "Hindu temples line the river .", "d:")
    assert "d:christianity" in values(nlp, "He led the Catholic Church in Spain .", "d:")
    assert "d:olympic_games" in values(nlp, "She won at the Olympic Games in Turin .", "d:")


def test_first_name_is_no_field(nlp):
    v = values(nlp, "In 1972 astronauts Gene Cernan and Harrison Schmitt flew .", "d:")
    assert "d:genetics" not in v and "d:spaceflight" in v
    assert "d:genetics" in values(nlp, "Each gene has two alleles .", "d:")
    # the rule NER labels "Chinese Buddhism" PERSON: its last word is no given name
    assert "d:buddhism_hinduism" in values(nlp, "His story shaped Chinese Buddhism and martial traditions .", "d:")


def test_field_hockey_is_not_ice_hockey(nlp):
    v = values(nlp, "She reached the pinnacle of Spanish field hockey .", "d:")
    assert v == {"d:field_hockey"}
    assert values(nlp, "He played ice hockey in Canada .", "d:") == {"d:ice_hockey"}


def test_trigger_inside_a_place_name_is_no_field(nlp):
    assert "saint lucia" in seed_tables().places and "bishop auckland" in seed_tables().places
    v = values(nlp, "She grew up in Saint Lucia and moved to Bishop Auckland in 1990 .")
    assert "d:christianity" not in v
    assert "d:christianity" in values(nlp, "The bishop moved to Durham in 1990 .", "d:")
