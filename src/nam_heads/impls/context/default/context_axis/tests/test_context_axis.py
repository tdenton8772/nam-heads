"""Context axis: extraction, encoder (batch == single == query), profiles, bundler scoping.

Every test parses with the rule NLP backend -- the backend production runs.
"""
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.context_axis import profiles as P
from nam_heads.ontology import bindings as B
from nam_heads.impls.context.default.context_axis.encoder import ParticularContextEncoder
from nam_heads.impls.context.default.context_axis.extract import extract_context
from nam_heads.impls.context.default.context_axis.temporal import century_of, decade_path, year_path


from nam_heads.impls.context.default.context_axis.lexicon import Lexicon, candidate_queries
from nam_heads.impls.context.default.context_axis.seed import seed_store
from nam_heads.impls.context.default.storage.memory_store import MemoryContextStore


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def seeded_store():
    st = MemoryContextStore()
    seed_store(st)
    return st


_SEEDED = None


def _store():
    global _SEEDED
    if _SEEDED is None:
        _SEEDED = seeded_store()
    return _SEEDED


def lexicon_for(doc, store=None):
    st = store or _store()
    return Lexicon(st.get_batch(sorted(candidate_queries(list(doc)))))


def values(nlp, text, dim=None):
    d = nlp.parse(text)
    return {cv.value for cv in extract_context(d, lexicon_for(d))
            if dim is None or cv.dimension == dim}


def encoder(store=None, **cfg):
    return ParticularContextEncoder(head_type="context",
                                    config={"store": store or _store(), **cfg})


# ---- temporal ---------------------------------------------------------------
def test_year_path_nests_decade_and_century():
    assert year_path(1865) == "C19/D1860/Y1865"
    assert year_path(1905).startswith(decade_path(1900))
    assert century_of(1900) == 20 and century_of(-44) == -1


@pytest.mark.parametrize("text,expected", [
    ("He was born in May 1865 in a small town.", "t:C19/D1860/Y1865"),
    ("The band was popular in the 1990s.", "t:C20/D1990"),
    ("The church dates from the 14th century.", "t:C14"),
    ("In 44 BC the dictator was killed.", "t:C-1/D-40/Y-44"),
    ("The town was founded in AD 800 by monks.", "t:C9/D800/Y800"),
    ("The bridge was destroyed during World War II.", "t:C20/E-world_war_ii"),
])
def test_temporal_values(nlp, text, expected):
    assert expected in values(nlp, text, "temporal")


@pytest.mark.parametrize("text", [
    "The village has 2000 people living in it.",
    "The prize was worth $1500 at the time.",
    "WCTR is a radio station at 1530 AM in Maryland.",
    "She may visit the lake.",
])
def test_numbers_that_are_not_years(nlp, text):
    assert values(nlp, text, "temporal") == set()


# ---- geographic -------------------------------------------------------------
def test_place_split_by_rule_ner_is_rejoined(nlp):
    # The rule NER emits "Jersey" and "Carolina" as PERSON; the gazetteer
    # matches the full capitalized window.
    v = values(nlp, "He moved from New Jersey to North Carolina.", "geographic")
    assert {"g:NA/US/US-NJ", "g:NA/US/US-NC"} <= v


def test_city_followed_by_its_state(nlp):
    v = values(nlp, "She was born in Springfield, Illinois in 1950.", "geographic")
    assert any(x.startswith("g:NA/US/US-IL/gn") for x in v)


def test_person_name_is_not_a_place(nlp):
    v = values(nlp, "George Washington crossed the river with Jack London.", "geographic")
    assert not any(x.startswith("g:NA/US/US-WA") for x in v)
    assert not any("gn2643743" in x for x in v)  # London, England


def test_month_is_not_a_place(nlp):
    assert values(nlp, "The album was released in March by the label.", "geographic") == set()


def test_demonym_resolves_to_its_place(nlp):
    assert "g:NA/US" in values(nlp, "Bob Smith was an American actor.", "geographic")
    assert "g:EU/x-soviet_union" in values(nlp, "It was the first Soviet tank.", "geographic")


def test_demonym_inside_a_name_is_skipped(nlp):
    v = values(nlp, "News Corp sold it to Central European Media Enterprises.", "geographic")
    assert "g:EU" not in v


def test_ambiguous_city_resolved_by_the_document(nlp):
    v = values(nlp, "He grew up in Canada. He later lived in London and Ontario politics.",
               "geographic")
    assert "g:NA/CA/gn6058560" in v  # London, Ontario, not London, England


# ---- institutional ----------------------------------------------------------
def test_named_organizations(nlp):
    v = values(nlp, "He studied at the University of Michigan and joined the Royal Air Force.",
               "institutional")
    assert {"i:university_of_michigan", "i:royal_air_force"} <= v


def test_gazetteer_is_case_sensitive(nlp):
    # The rule NER labels lowercase "who"/"time" ORG; they are not evidence.
    assert values(nlp, "People who have time often read.", "institutional") == set()
    assert "i:nfl" in values(nlp, "He played in the NFL for ten years.", "institutional")


def test_things_named_after_orgs_and_places_are_not_orgs(nlp):
    v = values(nlp, "She won an Academy Award. The Soviet Union collapsed.", "institutional")
    assert v == set()


def test_no_default_institution(nlp):
    # An ORG span with no head word and no gazetteer hit emits nothing.
    assert values(nlp, "Acme Widgets Fabrication Lakeside Partners met.", "institutional") == set()


# ---- domain -----------------------------------------------------------------
def test_domain_from_words_never_from_type(nlp):
    assert "d:baseball" in values(nlp, "He was a pitcher in the minor leagues.", "domain")
    assert values(nlp, "Abraham Lincoln was a man.", "domain") == set()


# ---- encoder: one code path ---------------------------------------------------
TEXTS = [
    "Abraham Lincoln was born in Kentucky in 1809. He later lived in Illinois.",
    "The University of Michigan football team played in the 1990s.",
    "Anarchism is a political philosophy.",
    "",
]


def test_batch_equals_single_equals_query(nlp):
    enc = encoder(learn=False)
    ctxs = [{"spacy_doc": nlp.parse(t)} if t else {} for t in TEXTS]
    batch = enc.encode_batch_texts(TEXTS, ctxs)
    for t, c, b in zip(TEXTS, ctxs, batch):
        single = enc.encode(t, c)
        query = enc.encode_query(t, c)
        key = lambda r: [(a.concept, a.metadata.get("sentences")) for a in r.addresses]
        assert key(b) == key(single) == key(query)


def test_null_when_no_evidence_and_sentences_carried(nlp):
    enc = encoder(learn=False)
    r = enc.encode("A cat sat.", {"spacy_doc": nlp.parse("A cat sat.")})
    assert [a.concept for a in r.addresses] == ["__null__"]
    r = enc.encode(TEXTS[0], {"spacy_doc": nlp.parse(TEXTS[0])})
    byc = {a.concept: a.metadata for a in r.addresses}
    assert byc["t:C19/D1800/Y1809"]["sentences"] == [0]
    assert byc["g:NA/US/US-IL"]["sentences"] == [1]


def test_deterministic_across_instances(nlp):
    d = nlp.parse(TEXTS[0])
    a = encoder(learn=False).encode(TEXTS[0], {"spacy_doc": d})
    b = encoder(learn=False).encode(TEXTS[0], {"spacy_doc": d})
    assert [x.concept for x in a.addresses] == [x.concept for x in b.addresses]


# ---- profiles: the ontology conditions, never becomes the value ---------------
def test_profile_curies_exist():
    from nam_heads.ontology.registry import get_registry
    reg = get_registry()
    for c in P.PROFILES:
        assert reg.is_class(c), c


def test_profiles_inherit_by_registry_ancestry():
    assert P.profile_for("cco:ont00000564").name == "organization"  # Educational Organization
    assert P.profile_for("cco:ont00000005").name == "process"       # Act
    assert P.profile_for("cco:ont00000887").name == "place"         # City
    assert P.profile_for("BFO:0000029").name == "place"         # Site (back-off ancestor)


def test_abstain_labels_and_none_get_the_default_profile():
    # bfo:entity is the ontology head abstaining: no evidence, never a match,
    # and nothing inherits from it (an Organism gets DEFAULT, not a root profile).
    assert P.profile_for(B.ENTITY) is P.DEFAULT
    assert P.profile_for("cco:ont00000551") is P.DEFAULT       # Organism
    assert P.profile_for("person") is P.DEFAULT                # labels are never emitted
    assert P.profile_for(None) is P.DEFAULT


def test_admit_excludes_own_mention_and_never_a_dimension_by_class():
    meta_geo = {"dimension": "geographic", "spans": [[5, 6]], "source": "gazetteer"}
    assert P.admit(B.PERSON, meta_geo, {1, 2})
    assert not P.admit(P.DEFAULT, meta_geo, {5})             # the mention itself
    # The class is the record's: it never removes a dimension from an entity
    # (Apollo 15 lost d:spaceflight under the record class Act).
    for cls in (B.QUALITY, B.PROCESS, B.GEOSPATIAL_REGION, B.ENTITY, None):
        assert P.admit(cls, meta_geo, {1})
        assert P.admit(cls, {"dimension": "domain", "spans": [[7, 8]]}, {1})
    dem = {"dimension": "geographic", "spans": [[0, 1]], "source": "norp_demonym"}
    assert not P.admit(P.PLACE, dem, {4})                    # a place's demonym is its people
    assert P.admit(P.DEFAULT, dem, {4})                      # a person's is nationality
    assert P.admit(B.PERSON, {"dimension": "something_else"}, set())


def test_select_keeps_nearest_per_dimension():
    meta = {"t:a": {"dimension": "temporal", "spans": [[20, 21]]},
            "t:b": {"dimension": "temporal", "spans": [[3, 4]]},
            "g:x": {"dimension": "geographic", "spans": [[9, 10]]},
            "legacy": {"dimension": "domain_old"}}
    out = P.select(B.PERSON, {2}, ["t:a", "t:b", "g:x", "legacy"], meta)
    assert set(out) == {"t:b", "g:x", "legacy"}


def test_sentence_dimensions_fallback_to_record_ontology():
    ont = SimpleNamespace(addresses=[SimpleNamespace(concept=B.PERSON)],
                          metadata={"sentence_ontologies": {1: B.QUALITY}})
    dims = P.sentence_dimensions(ont)
    assert "geographic" in dims(0)        # record ontology: person
    assert "geographic" not in dims(1)    # sentence ontology: quality
    assert P.sentence_dimensions(None) is None


# ---- bundler scoping --------------------------------------------------------
def test_bundler_scopes_by_sentence_and_profile(nlp):
    from src.addressing.bundler import EntityBundler
    from src.encoders.registry import EncodingResult, SemanticAddress
    # The full name: a bare sentence-initial "Lincoln" is read as the city
    # Lincoln, Nebraska (rule NER GPE), and a place takes no sibling state.
    text = "Abraham Lincoln was born in Kentucky in 1809. Grant was a general."
    doc = nlp.parse(text)
    ctx = encoder(learn=False).encode(text, {"spacy_doc": doc})
    ent = EncodingResult("entity", [
        SemanticAddress("entity", "entity:lincoln", metadata={"surface_form": "Lincoln", "entity_type": "person"}),
        SemanticAddress("entity", "entity:grant", metadata={"surface_form": "Grant", "entity_type": "person"}),
        SemanticAddress("entity", "entity:kentucky", metadata={"surface_form": "Kentucky", "entity_type": "location"}),
    ], {})
    enc = {"entity": ent, "context": ctx,
           "attribute": EncodingResult("attribute", [], {}),
           "affordance": EncodingResult("affordance", [], {})}
    bs = {b.entity: b for b in EntityBundler().bundle(enc, doc)}
    assert set(bs["entity:lincoln"].contexts) == {"g:NA/US/US-KY", "t:C19/D1800/Y1809"}
    assert bs["entity:grant"].contexts == ["__null__"]
    assert "g:NA/US/US-KY" not in bs["entity:kentucky"].contexts  # not its own context


def test_city_name_heading_a_longer_name_is_not_the_city(nlp):
    v = values(nlp, "Violence spread in Clay County , Kentucky during the feud.", "geographic")
    assert "g:NA/US/US-KY" in v
    assert not any("/gn" in x for x in v)


# ---- the store: seed, lookups, learning ---------------------------------------
class CountingStore(MemoryContextStore):
    def __init__(self):
        super().__init__()
        self.batch_calls = self.single_gets = 0

    def get(self, p, d):
        self.single_gets += 1
        return super().get(p, d)

    def get_batch(self, queries):
        self.batch_calls += 1
        return {q: self._store.get(q) for q in queries}


def learned(store):
    return {k: v for k, v in store._store.items() if k[0].startswith("cx:L:")}


def test_one_store_round_trip_per_batch(nlp):
    st = CountingStore()
    seed_store(st)
    st.single_gets = 0
    texts = TEXTS[:3]
    encoder(st).encode_batch_texts(texts, [{"spacy_doc": nlp.parse(t)} for t in texts])
    assert st.batch_calls == 1 and st.single_gets == 0


def test_seed_served_locally_learned_read_from_store(nlp):
    # The sealed seed is decoded in memory; the store is asked only for the
    # learned entries a doc can use, so seed keys never cost a lookup.
    st = CountingStore()
    seed_store(st)
    d = nlp.parse("She moved to Chicago in 1950 and lived in Richmond .")
    q = candidate_queries(list(d))
    assert all(p.startswith("cx:L:") for p, _ in q)
    assert ("cx:L:richmond", "context_axis") in q      # ambiguous seed name
    assert ("cx:L:chicago", "context_axis") not in q   # unambiguous: no lookup
    v = {cv.value for cv in extract_context(d, lexicon_for(d, st))}
    assert any("gn4887398" in x for x in v)            # Chicago from the local seed


def test_local_seed_is_what_the_store_is_seeded_with():
    from nam_heads.impls.context.default.context_axis.lexicon import SeedTables, seed_tables
    from nam_heads.impls.context.default.context_axis.seed import seed_entries
    st = seeded_store()
    stored = [(p, d, v, 1.0) for (p, d), v in st._store.items() if p.startswith("cx:")]
    assert sorted(stored) == sorted(seed_entries())
    assert SeedTables(stored).places == seed_tables().places


NEW_PLACE = ["Smith was born in Waunakee , Wisconsin in 1950 .",
             "The band formed in Waunakee , Wisconsin and toured .",
             "Jones later lived in Waunakee for many years ."]


def test_new_place_learned_and_used_after_two_witnesses(nlp):
    st = seeded_store()
    enc = encoder(st)
    r0 = enc.encode(NEW_PLACE[2], {"spacy_doc": nlp.parse(NEW_PLACE[2])})
    assert not any("waunakee" in a.concept for a in r0.addresses)  # unknown, no container
    enc.encode(NEW_PLACE[0], {"spacy_doc": nlp.parse(NEW_PLACE[0])})
    r1 = enc.encode(NEW_PLACE[2], {"spacy_doc": nlp.parse(NEW_PLACE[2])})
    assert not any("waunakee" in a.concept for a in r1.addresses)  # one witness: not yet
    enc.encode(NEW_PLACE[1], {"spacy_doc": nlp.parse(NEW_PLACE[1])})
    r2 = enc.encode(NEW_PLACE[2], {"spacy_doc": nlp.parse(NEW_PLACE[2])})
    assert "g:NA/US/US-WI/n-waunakee" in {a.concept for a in r2.addresses}


def test_reencoding_the_same_record_adds_no_witness(nlp):
    st = seeded_store()
    enc = encoder(st)
    for _ in range(3):
        enc.encode(NEW_PLACE[0], {"spacy_doc": nlp.parse(NEW_PLACE[0])})
    r = enc.encode(NEW_PLACE[2], {"spacy_doc": nlp.parse(NEW_PLACE[2])})
    assert not any("waunakee" in a.concept for a in r.addresses)


ACRO = ["He founded the Northern Rail Workers Union ( NRWU ) in 1920 .",
        "The Northern Rail Workers Union ( NRWU ) struck in 1926 .",
        "Members of the NRWU marched ."]


def test_acronym_alias_learned(nlp):
    st = seeded_store()
    enc = encoder(st)
    enc.encode_batch_texts(ACRO[:2], [{"spacy_doc": nlp.parse(t)} for t in ACRO[:2]])
    r = enc.encode(ACRO[2], {"spacy_doc": nlp.parse(ACRO[2])})
    assert "i:northern_rail_workers_union" in {a.concept for a in r.addresses}


def test_query_never_writes(nlp):
    st = seeded_store()
    n = st.count()
    encoder(st).encode_query(NEW_PLACE[0], {"spacy_doc": nlp.parse(NEW_PLACE[0])})
    assert st.count() == n


def test_reseeding_keeps_learned_entries(nlp):
    st = seeded_store()
    encoder(st).encode(NEW_PLACE[0], {"spacy_doc": nlp.parse(NEW_PLACE[0])})
    before = learned(st)
    assert before
    seed_store(st)
    assert learned(st) == before


def test_two_runs_converge_to_the_same_store(nlp):
    corpus = NEW_PLACE + ACRO + TEXTS[:3] + [
        "He grew up in Canada . He later lived in London and Ontario politics .",
        "She studied in London , Ontario before moving .",
    ]
    a, b = seeded_store(), seeded_store()
    ea, eb = encoder(a), encoder(b)
    ea.encode_batch_texts(corpus, [{"spacy_doc": nlp.parse(t)} for t in corpus])
    rev = list(reversed(corpus))
    for t in rev:  # different order, batch size 1
        eb.encode(t, {"spacy_doc": nlp.parse(t)})
    for t in corpus[:4]:  # restream part of it: idempotent
        eb.encode(t, {"spacy_doc": nlp.parse(t)})
    assert learned(a) == learned(b) and learned(a)


def test_vote_rule_threshold_and_margin():
    from nam_heads.impls.context.default.context_axis.lexicon import merge_votes, resolve_votes
    e = merge_votes(None, "A", ["w1"])
    assert resolve_votes(e) is None                      # one witness
    e = merge_votes(e, "A", ["w2", "w1"])
    assert resolve_votes(e) == "A"                       # two distinct witnesses
    e = merge_votes(e, "B", ["w3"])
    assert resolve_votes(e) is None                      # 2 vs 1: no 2x margin
    e = merge_votes(e, "A", ["w4", "w5"])
    assert resolve_votes(e) == "A"                       # 4 vs 1
    assert merge_votes(merge_votes(None, "A", ["x"]), "B", ["y"]) == \
        merge_votes(merge_votes(None, "B", ["y"]), "A", ["x"])   # order-free


def test_ambiguous_city_learned_from_documents_that_settle_it(nlp):
    st = seeded_store()
    enc = encoder(st)
    settle = ["The team is based in Richmond , Virginia near the river .",
              "He taught school in Virginia and later lived in Richmond for years ."]
    open_ = "She retired in Richmond after the war ."
    r0 = enc.encode(open_, {"spacy_doc": nlp.parse(open_)})
    assert not any(a.concept.startswith("g:") for a in r0.addresses)
    enc.encode_batch_texts(settle, [{"spacy_doc": nlp.parse(t)} for t in settle])
    r1 = enc.encode(open_, {"spacy_doc": nlp.parse(open_)})
    assert any(a.concept.startswith("g:NA/US/US-VA/gn") for a in r1.addresses)


def test_ascii_folded_city_is_the_geonames_place(nlp):
    v = values(nlp, "She was born in Montreal , Quebec in 1950 .", "geographic")
    assert any(x.startswith("g:NA/CA/gn") for x in v)
    assert not any("/n-" in x for x in v)


def test_tail_of_a_longer_name_is_not_a_new_place(nlp):
    st = seeded_store()
    obs = []
    d = nlp.parse("The airline was based in Quebec City , Quebec until 1999 .")
    extract_context(d, lexicon_for(d, st).seed_only(), obs=obs)
    assert not any(o[0] == "place" and o[1] == "city" for o in obs)


# ---- the current head's tables, folded in ---------------------------------------
def test_legacy_domain_keywords_need_two_per_sentence(nlp):
    # "commerce" and "currency" are DOMAIN_KEYWORD_PATTERNS['economic'] only
    # ("economy" and "inflation" are field triggers of economics now).
    assert "d:economics" in values(nlp, "Commerce grew with a stable currency that year .", "domain")
    assert "d:economics" not in values(nlp, "Commerce grew .", "domain")


def test_legacy_era_words_only_without_a_stated_year(nlp):
    # "medieval" is the Middle Ages: one period, one value (temporal.canonical).
    assert "t:E-middle_ages" in values(nlp, "The castle is a medieval building .", "temporal")
    v = values(nlp, "The medieval castle was rebuilt in 1850 .", "temporal")
    assert "t:E-middle_ages" not in v and "t:C19/D1850/Y1850" in v


def test_legacy_named_periods_and_geographic_seed(nlp):
    # TEMPORAL_PERIODS has "wwii", the canonical World War II; GEOGRAPHIC_SEED
    # surfaces resolve via their canonical.
    assert values(nlp, "He served in WWII as a pilot .", "temporal") == {"t:C20/E-world_war_ii"}
    from nam_heads.impls.context.default.context_axis.lexicon import seed_tables
    from nam_heads.impls.context.default.definitions import GEOGRAPHIC_SEED
    missing = [k for k in GEOGRAPHIC_SEED if k not in seed_tables().places]
    assert missing == [], missing


def test_legacy_institutional_keywords_are_heads_with_kind_metadata(nlp):
    from nam_heads.impls.context.default.definitions import INSTITUTIONAL_KEYWORDS
    from nam_heads.impls.context.default.context_axis.lexicon import seed_tables
    for kind, words in INSTITUTIONAL_KEYWORDS.items():
        for w in words.split():
            assert w.capitalize() in seed_tables().org_heads, w
    enc = encoder(learn=False)
    t = "He prayed at the Great Mosque of Kairouan ."
    r = enc.encode(t, {"spacy_doc": nlp.parse(t)})
    inst = [a for a in r.addresses if a.concept.startswith("i:")]
    assert inst and inst[0].metadata.get("kind") == "religious"
    assert inst[0].concept != "i:religious"   # the kind is metadata, never the value


def test_ner_labels_other_backends_emit_are_evidence(nlp):
    from types import SimpleNamespace
    d = nlp.parse("The Treaty of Ghent ended the war in 1814 .")
    toks = list(d)
    ents = [SimpleNamespace(label_="LAW", start=1, end=4, text="Treaty of Ghent"),
            SimpleNamespace(label_="EVENT", start=1, end=4, text="Treaty of Ghent")]
    class Doc:
        def __init__(self): self.ents, self.sents = ents, list(d.sents)
        def __iter__(self): return iter(toks)
    v = {cv.value for cv in extract_context(Doc(), lexicon_for(d))}
    assert "d:law" in v


def test_org_span_without_head_word(nlp):
    from types import SimpleNamespace
    d = nlp.parse("She joined Acme Widgets Fabrication Partners in 1990 .")
    toks = list(d)
    class Doc:
        def __init__(self):
            self.ents = [SimpleNamespace(label_="ORG", start=2, end=6,
                                         text="Acme Widgets Fabrication Partners")]
            self.sents = list(d.sents)
        def __iter__(self): return iter(toks)
    v = {cv.value for cv in extract_context(Doc(), lexicon_for(d))}
    assert "i:acme_widgets_fabrication_partners" in v


# ---- knobs, metrics, value cap -------------------------------------------------
def test_value_cap_is_deterministic_and_bounded():
    from nam_heads.impls.context.default.context_axis.extract import MAX_VALUE_CHARS, cap_value
    long = "i:" + "very_long_institution_name_" * 4
    a, b = cap_value(long), cap_value(long)
    assert a == b and len(a) == MAX_VALUE_CHARS and a != cap_value(long + "x")
    assert cap_value("t:C19/D1860/Y1865") == "t:C19/D1860/Y1865"


def test_split_fragments_off_scopes_values_to_the_whole_record(nlp):
    t = "Lincoln was born in Kentucky . He later lived in Illinois ."
    on = encoder(learn=False).encode(t, {"spacy_doc": nlp.parse(t)})
    off = encoder(learn=False, split_fragments=False).encode(t, {"spacy_doc": nlp.parse(t)})
    ky_on = next(a for a in on.addresses if a.concept == "g:NA/US/US-KY")
    ky_off = next(a for a in off.addresses if a.concept == "g:NA/US/US-KY")
    assert ky_on.metadata["sentences"] == [0] and ky_off.metadata["sentences"] == [0, 1]


def test_max_per_dimension_reaches_the_bundler():
    meta = {"t:a": {"dimension": "temporal", "spans": [[20, 21]], "max_per_dimension": 2},
            "t:b": {"dimension": "temporal", "spans": [[3, 4]], "max_per_dimension": 2},
            "t:c": {"dimension": "temporal", "spans": [[40, 41]], "max_per_dimension": 2}}
    assert set(P.select(B.PERSON, {2}, ["t:a", "t:b", "t:c"], meta)) == {"t:a", "t:b"}
    for m in meta.values():
        m["max_per_dimension"] = 0
    assert len(P.select(B.PERSON, {2}, ["t:a", "t:b", "t:c"], meta)) == 3


def test_metrics_and_config_summary(nlp):
    enc = encoder(learn=False)
    enc.encode(TEXTS[0], {"spacy_doc": nlp.parse(TEXTS[0])})
    m = enc.get_metrics()
    assert "context_store_size" in m and m["context_store_lookups"] >= 0
    c = enc.config_summary()
    assert {"max_per_dimension", "split_fragments", "context_store_backend"} <= set(c)


# ---- ingest and query agree ------------------------------------------------------
def test_ingest_and_query_extract_the_same_with_or_without_sentence_ontology(nlp):
    from types import SimpleNamespace
    t = "Lincoln was born in Kentucky in 1809 . Anarchism is a philosophy ."
    d = nlp.parse(t)
    rec = SimpleNamespace(addresses=[SimpleNamespace(concept=B.PERSON)], metadata={})
    sent = SimpleNamespace(addresses=[SimpleNamespace(concept=B.PERSON)],
                           metadata={"sentence_ontologies": {0: B.GEOSPATIAL_REGION, 1: B.QUALITY}})
    enc = encoder(learn=False)
    key = lambda r: sorted(a.concept for a in r.addresses)
    ingest = enc.encode(t, {"spacy_doc": d, "ontology_result": sent})
    query = enc.encode_query(t, {"spacy_doc": d, "ontology_result": rec})
    assert key(ingest) == key(query)


# ---- production read path: real LMDB L2, KV_ON_MISS=0 ------------------------------
def test_head_on_cold_lmdb_kv_on_miss_off_works_from_the_seed(nlp, tmp_path):
    """Cold LMDB, KV never consulted: the head still emits the seed-derived
    values (the seed is served from memory), and its learned writes are
    queued for the LMDB relay with their cx:L: keys."""
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    from nam_heads.impls.context.default.storage.couchbase_kv_store import CouchbaseKVContextStore
    w = LmdbAttributeCache(str(tmp_path), map_size=16 * 1024 * 1024, readonly=False)
    if not w.available:
        pytest.skip("lmdb not available")
    w.batch_put([("ctx::__init__::x", b"{}")])
    lmdb = LmdbAttributeCache(str(tmp_path), map_size=16 * 1024 * 1024, readonly=True)

    class NoKV:
        def __getattr__(self, name):
            def fail(*a, **k):
                if name in ("pipeline_set",):
                    return []
                raise AssertionError(f"KV {name} called with KV_ON_MISS=0")
            return fail

    st = CouchbaseKVContextStore(NoKV(), lmdb_cache=lmdb, kv_on_cache_miss=False)
    enc = encoder(st)
    t = "Smith was born in Waunakee , Wisconsin in 1950 and moved to Chicago ."
    r = enc.encode(t, {"spacy_doc": nlp.parse(t)})
    got = {a.concept for a in r.addresses}
    assert {"t:C20/D1950/Y1950", "g:NA/US/US-WI"} <= got
    assert any("gn4887398" in x for x in got)            # Chicago, from the seed
    relay = st.drain_pending_lmdb_writes()
    assert relay and all(k.startswith("ctx::cx:L:") for k, _ in relay)
    lmdb.close(); w.close()


def test_every_context_axis_prefix_is_carried_by_the_relay(tmp_path):
    """context-lmdb-sync is prefix-agnostic: cx:s:, cx:t: (plain) and cx:L:
    (vote sets, merged) all land in LMDB."""
    import json as _json
    from src.storage.lmdb_node_cache import LmdbAttributeCache
    from services.nats_worker import merge_context_writes
    w = LmdbAttributeCache(str(tmp_path), map_size=16 * 1024 * 1024, readonly=False)
    if not w.available:
        pytest.skip("lmdb not available")
    plain = lambda v: _json.dumps({"canonical_value": v, "confidence": 1.0}).encode()
    writes = [(b"ctx::cx:s:atlantis::context_axis", plain('{"place":[["r-atlantis","region",0]]}')),
              (b"ctx::cx:t:moth::context_axis", plain('{"field":"lepidoptera"}')),
              (b"ctx::cx:L:irmo::context_axis",
               plain('{"place":{"votes":{"NA/US/US-SC/n-irmo":["w1"]}}}'))]
    w.batch_put(merge_context_writes(w, writes))
    for k, _ in writes:
        assert w.get(k.decode()) is not None
    w.close()
