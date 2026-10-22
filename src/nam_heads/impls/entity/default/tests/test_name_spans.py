"""Name spans: names read from capitalisation, whatever the POS tagger said.

The rule tagger looks a word up in its dictionary before it checks case, so
"Young" is ADJ, "United" VERB, "Armed" VERB, and the noun chunker cut
"Brigham Young", "United States Mint" and "Angolan Armed Forces" apart. The
entity head recovers the span itself (normalization.name_spans); the NLP
backend is not changed (the ontology codebook pins its output).
"""

import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from ..head import EntityEncoder, _name_type
from ..normalization import EntityCandidate, extract_candidates, name_spans
from ..storage.factory import create_entity_store


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service(backend="rule")


@pytest.fixture
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.20
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


def spans(nlp, text):
    return [s for _a, _b, s in name_spans(nlp.parse(text))]


def cands(nlp, text):
    return {c.normalized: c for c in extract_candidates(text, spacy_doc=nlp.parse(text))}


def kept(encoder, nlp, text):
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    return {(a.metadata or {})["normalized"]: a for a in res.addresses
            if (a.metadata or {}).get("meaningful", True)}


# ---------------------------------------------------------------------------
# name_spans
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,want", [
    # "X of Y" and other connectors
    ("the Battle of Lostwithiel unfolded in Cornwall", ["Battle of Lostwithiel"]),
    ("the Wars of the Three Kingdoms ended", ["Wars of the Three Kingdoms"]),
    ("a painting by Leonardo da Vinci", ["Leonardo da Vinci"]),
    ("the game Bunnies & Burrows was published", ["Bunnies & Burrows"]),
    ("they crowned Edward the Confessor king", ["Edward the Confessor"]),
    # capitalised words the tagger tagged ADJ / VERB
    ("he studied at Brigham Young University in Utah", ["Brigham Young University"]),
    ("the coin came from the United States Mint in 1990", ["United States Mint"]),
    ("he joined the Angolan Armed Forces", ["Angolan Armed Forces"]),
    # abbreviations and initials
    ("he played for Great Wyrley F.C. in 1980", ["Great Wyrley F.C."]),
    ("John F. Kennedy visited the U.S. Army base in St. Louis",
     ["John F. Kennedy", "U.S. Army", "St. Louis"]),
    # trailing number / regnal numeral
    ("the crew of Apollo 15 landed", ["Apollo 15"]),
    ("King Charles I of England fled", ["King Charles I"]),
    ("released on PlayStation 2 and Samurai Shodown VI", ["PlayStation 2", "Samurai Shodown VI"]),
])
def test_spans(nlp, text, want):
    assert spans(nlp, text) == want


def test_a_sentence_period_is_not_an_abbreviation(nlp):
    assert spans(nlp, "It served in World War II. The base closed.") == ["World War II"]
    assert spans(nlp, "He joined Esteghlal F.C. In 2005 he left.") == ["Esteghlal F.C."]


def test_sentence_initial_words_do_not_open_a_name(nlp):
    assert spans(nlp, "The European Union met. In Paris it rained.") == ["European Union"]
    # a sentence-initial common noun joined to a name only by "of"
    assert spans(nlp, "Residents of Brussels enjoy the parks.") == []


def test_years_dates_and_pronouns_stay_out(nlp):
    assert spans(nlp, "From August 21 to September 2, 1644, the army waited.") == []
    assert spans(nlp, "In 1990 Smith moved to Paris.") == []


def test_no_span_crosses_an_apostrophe(nlp):
    assert spans(nlp, "It was NASA’s Apollo program.") == []


def test_spans_are_deterministic(nlp):
    text = "the Battle of Lostwithiel and the United States Mint and Great Wyrley F.C."
    assert name_spans(nlp.parse(text)) == name_spans(nlp.parse(text))


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def test_span_candidate_shape(nlp):
    c = cands(nlp, "he played for Great Wyrley F.C. in 1980")["great_wyrley_fc"]
    assert c.text == "Great Wyrley F . C ."       # token by token, as the parse has it
    assert c.is_name and c.is_propn and c.confidence == 0.6
    assert c.pos_pattern == "PROPN_PROPN_PROPN_PROPN" and c.token_count == 4


def test_chunk_inside_a_name_gives_way(nlp):
    by = cands(nlp, "the coin came from the United States Mint in 1990")
    assert "united_states_mint" in by and "mint" not in by and "states" not in by


def test_proper_noun_after_of_stands_on_its_own(nlp):
    by = cands(nlp, "the Battle of Lostwithiel unfolded in Cornwall")
    assert "battle_of_lostwithiel" in by and "lostwithiel" in by and "battle" not in by


def test_after_of_only_a_complete_tail_stands_alone(nlp):
    text = "His albums, such as The Life of Joseph W. McVey, are critical."
    by = cands(nlp, text)
    assert "the_life_of_joseph_w_mcvey" in by
    assert "joseph_w" not in by                     # cut short of "McVey"


def test_pronoun_i_is_not_a_regnal_number(nlp):
    assert spans(nlp, 'the song "I Think I Love You" was released') == []
    assert spans(nlp, "It ended World War I in 1918.") == ["World War I"]


def test_name_plus_lowercase_words_is_the_name(nlp):
    by = cands(nlp, "The company was founded by entrepreneur Yves Rocher in 1959.")
    assert "yves_rocher" in by and "entrepreneur_yves_rocher" not in by
    by = cands(nlp, "Andhra Pradesh stretches along the coast.")
    assert "andhra_pradesh" in by and "andhra_pradesh_stretches" not in by


def test_propn_tail_only_after_a_lowercase_prefix(nlp):
    by = cands(nlp, "He met president Lincoln in 1861.")
    assert "lincoln" in by
    by = cands(nlp, "She moved to New York in 1990.")
    assert "new_york" in by and "york" not in by


def test_curly_apostrophe_s_is_dropped(nlp):
    by = cands(nlp, "The Earth’s exploration began early.")
    assert "s_exploration" not in by and "exploration" in by


def test_pos_pattern_without_the_article(nlp):
    by = cands(nlp, "He walked into the town and saw a decisive engagement.")
    town, eng = by["town"], by["decisive_engagement"]
    assert town.pos_pattern == "NOUN" and town.token_count == 1 and town.has_article
    assert eng.pos_pattern == "ADJ_NOUN" and eng.token_count == 2 and eng.has_article
    assert town.confidence == 0.25                # no multi-token bonus from "the"


def test_is_name_flags(nlp):
    by = cands(nlp, "Management was weak. The rocket flew to the Moon and the town.")
    assert by["moon"].is_name                       # capitalised inside a sentence
    assert not by["management"].is_name            # capitalised because it opens one
    assert not by["town"].is_name


def test_candidates_in_text_order(nlp):
    text = "Cornwall saw the Battle of Lostwithiel and the River Fowey valley."
    starts = [c.start for c in extract_candidates(text, spacy_doc=nlp.parse(text))]
    assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# Through the head: names are never rejected or filtered
# ---------------------------------------------------------------------------

def test_names_the_codebook_rejected_are_kept(encoder, nlp):
    text = ("He led the LDS Church in Utah. Apollo 8 orbited the Moon. "
            "He was picked in the NFL Draft by Denver.")
    k = kept(encoder, nlp, text)
    for n in ("lds_church", "moon", "nfl_draft", "apollo_8"):
        assert n in k, n
    assert k["moon"].metadata["entity_type"] != "rejection_generic_noun"


def test_names_the_compound_filter_dropped_are_kept(encoder, nlp):
    text = ("Brussels hosts the European Union. The Royal Navy fought the Roman Empire "
            "before the English Civil War.")
    k = kept(encoder, nlp, text)
    for n in ("european_union", "royal_navy", "roman_empire", "english_civil_war"):
        assert n in k, n


def test_name_ids_keep_their_form(encoder, nlp):
    k = kept(encoder, nlp, "He moved to the United States and joined the Angolan Armed Forces.")
    import hashlib
    for n in ("united_states", "angolan_armed_forces"):
        assert k[n].concept == hashlib.sha256(n.encode()).hexdigest()[:12]


def test_name_type_relabels_only_names():
    name = EntityCandidate(text="Moon", normalized="moon", confidence=0.45, is_name=True)
    noun = EntityCandidate(text="moon", normalized="moon", confidence=0.25)
    assert _name_type(name, "rejection_generic_noun") == "proper_noun"
    assert _name_type(noun, "rejection_generic_noun") == "rejection_generic_noun"
    assert _name_type(name, "person") == "person"


def test_query_resolves_a_name_span_to_the_ingest_id(encoder, nlp):
    text = "The coin came from the United States Mint in 1990."
    k = kept(encoder, nlp, text)
    items = encoder.resolve_query_entities("What is the United States Mint?")
    assert items and items[0]["entity_id"] == k["united_states_mint"].concept


def test_canonical_keeps_ss_us_is_words(encoder):
    lc = {"business": "busines", "campus": "campu", "crisis": "crisi", "cats": "cat"}
    assert encoder._canonical_entity_id("small_business", lc, set()) == "small_business"
    assert encoder._canonical_entity_id("campus", lc, set()) == "campus"
    assert encoder._canonical_entity_id("crisis", lc, set()) == "crisis"
    assert encoder._canonical_entity_id("cats", lc, set()) == "cat"


@pytest.mark.parametrize("text,name,dropped", [
    ("Cuba Cola kept its presence in southern Sweden for years.", "sweden", "southern_sweden"),
    ("They are related to frogs in the genus Huia today.", "huia", "genus_huia"),
    ("Along with his brother Bukka, he was captured.", "bukka", "brother_bukka"),
    ("In 731 the English monk Bede completed his history.", "bede", "english_monk_bede"),
    ("It lies west of the district capital Düren today.", "duren", "district_capital_duren"),
    ("He met president Lincoln in 1861.", "lincoln", "president_lincoln"),
])
def test_words_leading_into_a_name_are_dropped(nlp, text, name, dropped):
    by = cands(nlp, text)
    assert name in by and dropped not in by, sorted(by)


def test_capitalised_lead_or_connector_keeps_the_name_whole(nlp):
    by = cands(nlp, "She moved to New York in 1990.")
    assert "new_york" in by and "york" not in by
    by = cands(nlp, "The museum shows works by van Gogh from Arles.")
    assert "gogh" not in by, sorted(by)


@pytest.mark.parametrize("text,whole", [
    ("He was admitted to the prestigious French Académie des Beaux-Arts in 1770.", "beaux_arts"),
    ("The sultan Muhammad bin Tughlaq moved the capital.", "tughlaq"),
])
def test_a_name_particle_never_leads_into_a_name(nlp, text, whole):
    by = cands(nlp, text)
    assert whole not in by, sorted(by)
