"""Context values conditioned by the entity's OWN evidence (2026-10-07 audit).

* a place never takes itself, a place inside it, or a sibling; it takes the
  container of its own path (emitted by the head), and a place that is not
  the record's topic keeps only where it is;
* the record's class never removes a dimension from an entity;
* per dimension the most specific candidate is kept;
* "Earl of Essex" is a person's title, not the county;
* institution slugs keep accented letters' base letters, a person is not an
  institution, "City Hall" is no city organization, a numbered unit keeps
  its number.

Every test parses with the rule NLP backend -- the backend production runs.
"""
import os

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.context_axis import profiles as P
from nam_heads.impls.context.default.context_axis.encoder import ParticularContextEncoder
from nam_heads.impls.context.default.context_axis.extract import extract_context
from nam_heads.impls.context.default.context_axis.institutions import slug
from nam_heads.impls.context.default.context_axis.lexicon import Lexicon, candidate_queries
from nam_heads.impls.context.default.context_axis.seed import seed_store
from nam_heads.impls.context.default.storage.memory_store import MemoryContextStore
from nam_heads.ontology import bindings as B

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


def extract(nlp, text):
    d = nlp.parse(text)
    return extract_context(d, Lexicon(_store().get_batch(sorted(candidate_queries(list(d))))))


def bundles(nlp, text, surfaces, title="", ontology=None):
    """Bundle ``text`` with the context head and an entity head result naming
    ``surfaces``; returns {surface: contexts}."""
    from src.addressing.bundler import EntityBundler
    from src.encoders.registry import EncodingResult, SemanticAddress
    doc = nlp.parse(text)
    ctx = ParticularContextEncoder(head_type="context",
                                   config={"store": _store(), "learn": False}).encode(text, {"spacy_doc": doc})
    ent = EncodingResult("entity", [
        SemanticAddress("entity", f"entity:{s.lower().replace(' ', '_')}", metadata={"surface_form": s})
        for s in surfaces], {})
    enc = {"entity": ent, "context": ctx,
           "attribute": EncodingResult("attribute", [], {}),
           "affordance": EncodingResult("affordance", [], {})}
    if ontology:
        enc["ontology"] = EncodingResult("ontology", [SemanticAddress("ontology", ontology)], {})
    out = {}
    for b in EntityBundler().bundle(enc, doc, title=title):
        out[b.entity.split(":", 1)[1]] = [c for c in b.contexts if c != "__null__"]
    return out


def key(s):
    return s.lower().replace(" ", "_")


# ---- place rules (pure) -------------------------------------------------------
@pytest.mark.parametrize("value,own,conflict", [
    ("g:NA/US/US-IL", "g:NA/US/US-IL", True),                 # itself
    ("g:EU/BE/gn2800866", "g:EU/BE", True),                   # Brussels inside Belgium
    ("g:NA/US/US-UT", "g:NA/US/US-IL", True),                 # sibling states
    ("g:AS/IR", "g:AS/AF", True),                             # sibling countries
    ("g:EU/GB/GB-ENG/s-essex", "g:EU/GB/GB-ENG/s-cornwall", True),  # sibling counties
    ("g:NA/US", "g:NA/US/US-IL", False),                      # its container
    ("g:EU", "g:EU/GB/GB-ENG/s-cornwall", False),             # a coarser container
    ("g:AF/r-southern_africa", "g:AF/BW", False),             # a region overlaps
    ("g:EU/x-czechoslovakia", "g:EU/CZ", False),              # a historical country overlaps
    ("g:EU/GB/GB-ENG/gn2643743", "g:EU/GB/GB-ENG/s-cumbria", False),  # a city beside a county
    ("g:x-roman_empire", "g:EU/IT", False),
    ("g:AF/BW", "g:AF/r-southern_africa", True),             # may lie inside the region
    ("g:EU/IT/gn3169070", "g:EU/r-balkans", True),
    ("g:EU/IT", "g:x-roman_empire", True),
    ("g:AF", "g:AF/r-southern_africa", False),               # the region's container
    ("g:EU/r-balkans", "g:EU/r-scandinavia", False),          # regions overlap
])
def test_place_conflict(value, own, conflict):
    assert P.place_conflict(value, own) is conflict


def test_container_of():
    assert P.container_of("g:EU/GB/GB-ENG/s-cornwall") == "g:EU/GB/GB-ENG"
    assert P.container_of("g:EU") is None
    assert P.container_of("g:x-roman_empire") is None
    assert P.container_of("t:C19/D1860") is None


def test_entity_profile_from_evidence():
    assert P.entity_profile([], True) is P.DEFAULT
    assert P.entity_profile(["g:NA/US/US-IL"], True) is P.PLACE
    assert P.entity_profile(["g:NA/US/US-IL"], None) is P.PLACE      # no title: nothing removed
    assert P.entity_profile(["g:NA/US/US-IL"], False) is P.PLACE_ONLY
    assert P.PLACE.dimensions == P.ALL and not P.PLACE.demonyms
    assert P.PLACE_ONLY.dimensions == {"geographic"}


def test_admit_container_only_value_is_no_mention():
    cont = {"dimension": "geographic", "spans": [], "containers": ["g:NA/US/US-IL"], "source": "place_container"}
    assert not P.admit(P.DEFAULT, cont, {3})
    named_too = dict(cont, spans=[[9, 10]])                # the text also names it
    assert P.admit(P.DEFAULT, named_too, {3})


def test_admit_place_conflicts_with_own_place():
    utah = {"dimension": "geographic", "spans": [[9, 10]], "source": "gazetteer"}
    assert not P.admit(P.PLACE, utah, {7}, ["g:NA/US/US-IL"], "g:NA/US/US-UT")
    assert P.admit(P.DEFAULT, utah, {7}, [], "g:NA/US/US-UT")
    year = {"dimension": "temporal", "spans": [[2, 3]]}
    assert not P.admit(P.PLACE_ONLY, year, {7})
    assert P.admit(P.PLACE, year, {7})


def test_select_keeps_the_most_specific():
    meta = {"g:NA/US": {"dimension": "geographic", "spans": [[5, 6]], "demonym_spans": [[5, 6]]},
            "g:NA/US/US-CA/gn5368361": {"dimension": "geographic", "spans": [[11, 13]]},
            "t:C19/D1880": {"dimension": "temporal", "spans": [[15, 16]]},
            "t:C19/D1880/Y1885": {"dimension": "temporal", "spans": [[30, 31]]}}
    out = P.select(P.DEFAULT, {3}, list(meta), meta)
    assert out == ["g:NA/US/US-CA/gn5368361", "t:C19/D1880/Y1885"]


def test_select_keeps_a_much_nearer_named_coarser_place():
    # "people migrated from North India near the Yamuna River", with
    # "Andhra Pradesh" named far earlier: India is said of the river.
    meta = {"g:AS/IN/s-andhra_pradesh": {"dimension": "geographic", "spans": [[2, 4]]},
            "g:AS/IN": {"dimension": "geographic", "spans": [[30, 31]]}}
    assert P.select(P.DEFAULT, {33, 34}, list(meta), meta) == ["g:AS/IN"]
    # the same coarser place as a demonym gives way
    meta["g:AS/IN"]["demonym_spans"] = [[30, 31]]
    assert P.select(P.DEFAULT, {33, 34}, list(meta), meta) == ["g:AS/IN/s-andhra_pradesh"]


def test_select_container_of_own_place_is_nearest():
    meta = {"g:NA/US": {"dimension": "geographic", "spans": [], "containers": ["g:NA/US/US-IL"]},
            "g:x-roman_empire": {"dimension": "geographic", "spans": [[8, 9]]}}
    assert P.select(P.PLACE, {7}, list(meta), meta, ["g:NA/US/US-IL"]) == ["g:NA/US"]


# ---- the head emits containers ------------------------------------------------
def test_head_emits_the_container_of_each_named_place(nlp):
    vals = {v.value: v for v in extract(nlp, "They traveled from Illinois to Utah in 1846.")}
    c = vals["g:NA/US"]
    assert c.spans == [] and c.sentences == [] and c.source == "place_container"
    assert c.containers == ["g:NA/US/US-IL", "g:NA/US/US-UT"]
    assert vals["g:NA/US/US-IL"].containers == []


def test_container_merges_with_a_mention_of_it(nlp):
    vals = {v.value: v for v in extract(nlp, "Brussels is the capital of Belgium.")}
    be = vals["g:EU/BE"]
    assert be.spans and be.containers == ["g:EU/BE/gn2800866"] and be.source != "place_container"


def test_demonym_spans_are_marked(nlp):
    vals = {v.value: v for v in extract(nlp, "He was an American architect who lived in America.")}
    us = vals["g:NA/US"]
    assert len(us.spans) == 2 and us.demonym_spans == [us.spans[0]]


def test_encoder_metadata_carries_containers_and_demonym_spans(nlp):
    text = "They were American pioneers who traveled from Illinois to Utah."
    r = ParticularContextEncoder(head_type="context", config={"store": _store(), "learn": False}).encode(
        text, {"spacy_doc": nlp.parse(text)})
    meta = {a.concept: a.metadata for a in r.addresses}
    assert meta["g:NA/US"]["containers"] == ["g:NA/US/US-IL", "g:NA/US/US-UT"]
    assert meta["g:NA/US"]["demonym_spans"] and meta["g:NA/US"]["spans"]


# ---- bundler: the entity's own evidence ---------------------------------------
def test_a_place_takes_its_container_never_a_sibling(nlp):
    bs = bundles(nlp, "Mormon pioneers traveled from Illinois to Utah in 1846.",
                 ["Mormon pioneers", "Illinois", "Utah"], title="Brigham Young")
    assert bs["illinois"] == ["g:NA/US"]
    assert bs["utah"] == ["g:NA/US"]
    assert "t:C19/D1840/Y1846" in bs["mormon_pioneers"]


def test_a_place_never_takes_a_place_inside_it(nlp):
    bs = bundles(nlp, "Brussels is the largest city of Belgium.", ["Brussels", "Belgium"],
                 title="Brussels")
    assert "g:EU/BE/gn2800866" not in bs["belgium"]
    assert bs["belgium"] == ["g:EU"]
    assert bs["brussels"] == ["g:EU/BE"]


def test_a_place_that_is_not_the_topic_keeps_only_where_it_is(nlp):
    text = "In 1644 the royalist army won a battle in Cornwall."
    bs = bundles(nlp, text, ["royalist army", "Cornwall"], title="Battle of Lostwithiel")
    assert bs["cornwall"] == ["g:EU/GB/GB-ENG"]
    assert "t:C17/D1640/Y1644" in bs["royalist_army"]
    # the topic place keeps every dimension
    bs = bundles(nlp, text, ["royalist army", "Cornwall"], title="Cornwall")
    assert set(bs["cornwall"]) == {"g:EU/GB/GB-ENG", "t:C17/D1640/Y1644", "d:military"}
    # no title (a query): the topic is unknown and nothing is removed
    bs = bundles(nlp, text, ["royalist army", "Cornwall"])
    assert "t:C17/D1640/Y1644" in bs["cornwall"]


def test_record_class_never_removes_a_dimension(nlp):
    text = "Apollo 15 was one of NASA's Apollo missions to the Moon in 1971."
    for cls in (B.PROCESS, B.GEOSPATIAL_REGION, B.QUALITY):
        bs = bundles(nlp, text, ["Apollo 15"], title="Apollo 15", ontology=cls)
        assert "d:spaceflight" in bs["apollo_15"], cls
        assert "t:C20/D1970/Y1971" in bs["apollo_15"], cls


def test_a_name_headed_by_another_word_is_not_its_place(nlp):
    # the second "Croatia" is outside the entity; its head is "Politics"
    text = "Politics of Croatia : Croatia is a parliamentary republic since 1990."
    bs = bundles(nlp, text, ["Politics of Croatia"], title="Politics of Croatia")
    assert "g:EU/HR" in bs["politics_of_croatia"]


def test_most_specific_place_wins_over_nationality(nlp):
    text = "Sumner Hunt was an American architect who moved to Los Angeles in the 1880s."
    bs = bundles(nlp, text, ["Sumner Hunt"], title="Sumner Hunt")
    assert "g:NA/US/US-CA/gn5368361" in bs["sumner_hunt"]
    assert "g:NA/US" not in bs["sumner_hunt"]


def test_a_demonym_entity_is_not_a_place(nlp):
    text = "The village is in Serbia, home to many Serbian farmers."
    bs = bundles(nlp, text, ["Serbian farmers", "Serbia"], title="Kucura")
    assert bs["serbian_farmers"] == ["g:EU/RS"]
    assert bs["serbia"] == ["g:EU"]


def test_containers_reach_no_fallback_bundle(nlp):
    from src.addressing.bundler import EntityBundler
    from src.encoders.registry import EncodingResult
    text = "They traveled from Illinois to Utah."
    doc = nlp.parse(text)
    ctx = ParticularContextEncoder(head_type="context",
                                   config={"store": _store(), "learn": False}).encode(text, {"spacy_doc": doc})
    enc = {"entity": EncodingResult("entity", [], {}), "context": ctx,
           "attribute": EncodingResult("attribute", [], {}),
           "affordance": EncodingResult("affordance", [], {})}
    (b,) = EntityBundler().bundle(enc, doc)
    assert set(b.contexts) == {"g:NA/US/US-IL", "g:NA/US/US-UT"}


def test_topics_from_the_title():
    from src.addressing.bundler import EntityBundler
    ents = [("e:a", "Botswana"), ("e:b", "Geography of Botswana"), ("e:c", "Southern Africa"),
            ("e:d", "Andhra Pradesh stretches")]
    assert EntityBundler._context_topics(ents, "Geography of Botswana") == {"e:a", "e:b"}
    assert EntityBundler._context_topics(ents, "Andhra Pradesh") == {"e:d"}
    assert EntityBundler._context_topics(ents, "Santa Margarita, California") == set()
    assert EntityBundler._context_topics([("e:x", "Yves Rocher")], "Yves Rocher (company)") == {"e:x"}
    assert EntityBundler._context_topics(ents, "") is None


def test_own_places_use_the_head_before_of():
    from src.addressing.bundler import EntityBundler
    meta = {"g:EU/HR": {"dimension": "geographic", "spans": [[2, 3]]},
            "g:EU/IS": {"dimension": "geographic", "spans": [[9, 10]]}}
    text = {0: "Politics", 1: "of", 2: "Croatia", 7: "late", 8: "17th-century", 9: "Iceland"}
    assert EntityBundler._context_own_places({0, 1, 2}, meta, text) == []
    assert EntityBundler._context_own_places({7, 8, 9}, meta, text) == ["g:EU/IS"]
    meta["g:EU/IS"]["demonym_spans"] = [[9, 10]]
    assert EntityBundler._context_own_places({9}, meta, text) == []


# ---- "Earl of Essex" ----------------------------------------------------------
def test_peerage_title_is_not_the_place(nlp):
    text = ("The Parliamentarian army was led by the Earl of Essex. "
            "Essex managed to escape with the cavalry.")
    assert not any(v.value.startswith("g:EU/GB/GB-ENG/s-essex") for v in extract(nlp, text))
    text2 = "The Earl of Essex owned land in Essex."
    vals = [v for v in extract(nlp, text2) if v.value == "g:EU/GB/GB-ENG/s-essex"]
    assert len(vals) == 1 and len(vals[0].spans) == 1      # only "in Essex"


# ---- institutions -------------------------------------------------------------
def test_slug_folds_accents_and_keeps_letters():
    assert slug("Juan Ponce de León") == "juan_ponce_de_leon"
    assert slug("Árni Magnússon Institute") == "arni_magnusson_institute"
    assert slug("Søren Kierkegaard Society") == "soren_kierkegaard_society"
    assert slug("University of Michigan") == "university_of_michigan"
    assert slug("McDonald's Corporation") == "mcdonald_corporation"


def institutions_of(nlp, text):
    return {v.value for v in extract(nlp, text) if v.dimension == "institutional"}


def test_person_is_not_an_institution(nlp):
    assert not any("ponce" in v for v in institutions_of(
        nlp, "The day saw Juan Ponce de León reach the coast of Florida."))
    assert "i:san_luis_obispo_de_tolosa" in institutions_of(
        nlp, "Serra founded it near Mission San Luis Obispo de Tolosa in 1772.")


def test_city_hall_is_not_a_city_organization(nlp):
    assert institutions_of(nlp, "He oversaw the building of the Los Angeles City Hall.") == set()


def test_numbered_unit_keeps_its_number(nlp):
    vals = institutions_of(nlp, "Organizations like No. 57 Squadron RAF keep the memory alive.")
    assert vals == {"i:no_57_squadron_raf"}
    assert "i:1st_battalion_royal_marines" in institutions_of(
        nlp, "He joined the 1st Battalion Royal Marines in 1940.")
    assert not any("squadron" in v for v in institutions_of(nlp, "The Squadron RAF flew at night."))
