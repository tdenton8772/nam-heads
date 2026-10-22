"""Predication-mode attribute extraction, ontology steering and anchored linking.

Runs on the production NLP (rule backend), so every case is the parse the
pipeline actually sees.
"""
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from nam_heads.impls.attribute.default import predication as P
from nam_heads.impls.attribute.default.head import AttributeEncoder
from nam_heads.impls.attribute.default.definitions import (
    ONTOLOGY_RELEVANCE, entity_admits, entity_class, get_ontology_relevance, relevance_row)
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import get_registry


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    svc = get_nlp_service()
    assert svc.backend_name == "rule"
    return svc


def values(nlp, text, query=False):
    return {av.value: av for av in P.attribute_values(nlp.parse(text), query=query)}


def subject(doc, av):
    return [" ".join(doc[i].text for i in range(s, e)) for s, e in av.anchors]


# ---------------------------------------------------------------- extraction

def test_copular_predicate_nominal_through_parenthetical(nlp):
    text = "Albert Brooks (born Albert Lawrence Einstein; July 22, 1947) is an American actor, comedian and director."
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc)}
    for v in ("american", "actor", "comedian", "director"):
        assert v in vals, vals.keys()
        assert "Albert Brooks" in subject(doc, vals[v])
    assert vals["actor"].link == "copular"
    assert "norp" in vals["american"].kinds
    assert "role_noun" in vals["actor"].kinds


def test_quantifier_head_is_followed_but_not_emitted(nlp):
    vals = values(nlp, "Anacondas are a group of large snakes of the genus Eunectes.")
    assert "large" in vals and "snake" in vals
    assert "group" not in vals          # quantifier, not a property
    assert "eunectes" not in vals       # proper name


def test_member_of_keeps_member_not_the_organisation(nlp):
    vals = values(nlp, "Olav Meisdalshagen was a Norwegian politician. He was a member of the Parliament of Norway.")
    assert {"norwegian", "politician", "member"} <= set(vals)
    assert "parliament" not in vals


def test_pronoun_resolves_to_main_subject(nlp):
    text = "Graeme Sturges was an Australian politician. He was a minister in the state government."
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc)}
    assert vals["minister"].link == "pronoun"
    assert subject(doc, vals["minister"]) == ["Graeme Sturges"]


def test_pronoun_without_main_subject_stays_on_the_pronoun(nlp):
    """No title and no copular subject to resolve "He" to: the value is kept,
    anchored on the pronoun itself -- no name's tokens overlap it, so only
    the bundler's topic carry-over (pronoun mentions extend the topic
    entity's tokens) can bind it."""
    text = "In 1990 the club moved. He was a minister."
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc)}
    assert subject(doc, vals["minister"]) == ["He"]


def test_title_requires_a_person_noun(nlp):
    vals = values(nlp, "The album features the guitarist Andy Lee. In his life Afonso travelled.")
    assert vals["guitarist"].link == "title"
    assert "life" not in vals


def test_appositive(nlp):
    text = "The bill was signed by Lincoln, the sixteenth president, in 1862."
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc)}
    assert "president" in vals
    assert subject(doc, vals["president"]) == ["Lincoln"]
    assert "sixteenth" not in vals  # ordinal


def test_as_role(nlp):
    vals = values(nlp, "Speer served as the Minister of Armaments from 1942.")
    assert "minister" in vals and vals["minister"].link == "as_role"


def test_negated_predicate_is_not_an_attribute(nlp):
    vals = values(nlp, "The river is not a canal.")
    assert "canal" not in vals


def test_modifier_inside_a_name_is_dropped(nlp):
    vals = values(nlp, "The bird lives in Central America and visits the National Park.")
    assert "central" not in vals and "national" not in vals


def test_directional_part_of_a_place_is_dropped(nlp):
    vals = values(nlp, "The species is found in southwestern Ecuador and northern Malawi.")
    assert "southwestern" not in vals and "northern" not in vals


def test_heading_modifiers_are_dropped(nlp):
    vals = values(nlp, "External links Professional career Smith joined the club.")
    assert "external" not in vals and "professional" not in vals


def test_ordinals_and_vague_adjectives_are_dropped(nlp):
    vals = values(nlp, "She was the first famous woman to fly. It was a notable small town.")
    assert "first" not in vals and "famous" not in vals and "notable" not in vals
    assert "small" in vals


def test_subject_with_of_phrase_anchors_on_the_head_noun(nlp):
    text = "The western part of Slovene Istria is bilingual."
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc)}
    for av in vals.values():
        for s, e in av.anchors:
            assert "Istria" not in [doc[i].text for i in range(s, e)]


def test_normalise_keeps_ss_us_bases():
    w = lambda text, lemma: P.Word(0, text, lemma, "NOUN", "head")
    assert P.normalise(w("genus", "genu")) == "genus"
    assert P.normalise(w("actress", "actres")) == "actress"
    assert P.normalise(w("actresses", "actress")) == "actress"
    assert P.normalise(w("actors", "actor")) == "actor"


def test_empty_and_none_documents(nlp):
    assert P.attribute_values(None) == []
    assert P.attribute_values(nlp.parse("")) == [] if nlp.parse("") is not None else True


# ------------------------------------------------- ingest / query agreement

@pytest.mark.parametrize("sentence,query", [
    ("Speer was a German architect.", "German architect"),
    ("Richard Tucker was an American astronomer.", "American astronomer?"),
    ("Kenia Sinclair is a Jamaican athlete.", "Jamaican athlete"),
])
def test_query_leading_np_matches_ingest_predicate(nlp, sentence, query):
    ingest = set(values(nlp, sentence))
    q = set(values(nlp, query, query=True))
    assert ingest and ingest == q


def test_query_with_wh_subject(nlp):
    q = set(values(nlp, "Who was the German architect?", query=True))
    assert {"german", "architect"} <= q


# ------------------------------------------------- ontology steer (ONTOLOGY_RELEVANCE)

def test_relevance_rows_resolve_through_the_registry():
    reg = get_registry()
    # a class and any subclass of it without a row of its own share one row
    assert relevance_row(B.PERSON) == ONTOLOGY_RELEVANCE[B.PERSON]
    for c in reg.descendants(B.PERSON):
        assert relevance_row(c) == ONTOLOGY_RELEVANCE[B.PERSON]
    assert relevance_row(B.FACILITY) == ONTOLOGY_RELEVANCE[B.MATERIAL_ARTIFACT]  # Facility -> artifact row
    # labels are never emitted and are not ontology evidence; abstain is neutral
    assert relevance_row("PERSON") == {} and relevance_row("person") == {}
    assert relevance_row(B.ENTITY) == {}
    # the steer stays a nudge (documented 0.8-1.3); exclusion is separate
    for row in ONTOLOGY_RELEVANCE.values():
        assert all(0.8 <= v <= 1.3 for v in row.values())
    assert relevance_row("no-such-label") == {} and relevance_row(None) == {}
    assert get_ontology_relevance("cco:not-a-class", "role") == 1.0


def test_entity_admits_from_the_relevance_table():
    assert not entity_admits(["role_noun"], None, B.MATERIAL_ARTIFACT)
    assert entity_admits(["noun"], None, B.MATERIAL_ARTIFACT)
    assert not entity_admits(["noun"], None, B.PERSON)
    assert entity_admits(["role_noun", "noun"], None, B.PERSON)   # any admitted kind
    assert entity_admits([], None, B.PERSON)                       # legacy values
    assert entity_admits(["adj"], "physical", B.PERSON)
    assert entity_admits(["noun"], None, B.ENTITY)                 # abstain: no evidence


def test_entity_class_accepts_curies_only():
    assert entity_class(B.PERSON) == B.PERSON
    assert entity_class("person") == B.ENTITY          # labels are not classes
    assert entity_class("ORG") == B.ENTITY
    assert entity_class("no-such-label") == B.ENTITY
    assert entity_class(None) == B.ENTITY


# ---------------------------------------------------------------- head

@pytest.fixture(scope="module")
def encoder():
    return AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})


def _ont(concept, sentence_ontologies=None):
    meta = {"sentence_ontologies": sentence_ontologies} if sentence_ontologies else {}
    return EncodingResult("ontology", [SemanticAddress("ontology", concept)], meta)


def test_head_emits_every_value_with_anchors_no_record_cap(nlp, encoder):
    text = ("Joel Greenblatt is an American academic, hedge fund manager, investor and writer. "
            "He is a graduate of the Wharton School.")
    doc = nlp.parse(text)
    res = encoder.encode_batch_texts([text], [{"spacy_doc": doc}])[0]
    concepts = {a.concept for a in res.addresses}
    assert {"american", "academic", "manager", "investor", "writer", "graduate"} <= concepts
    assert len(concepts) > encoder.max_attributes      # the old per-record cap of 3 is gone
    for a in res.addresses:
        assert a.metadata["anchors"] and a.metadata["kinds"]
    assert res.metadata["extraction"] == "legacy+predication"


def test_head_null_when_no_evidence(nlp, encoder):
    text = "In 1990 it moved."
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert [a.concept for a in res.addresses] == ["__null__"]


def test_head_batch_and_single_paths_agree(nlp, encoder):
    text = "Kenia Sinclair is a Jamaican athlete. Anacondas are large snakes."
    doc = nlp.parse(text)
    ctx = {"spacy_doc": doc, "ontology_result": _ont(B.PERSON)}
    batch = {a.concept for a in encoder.encode_batch_texts([text], [ctx])[0].addresses}
    single = {a.concept for a in encoder.encode(text, ctx).addresses}
    assert batch == single


def test_sentence_ontology_steers_evidence(nlp):
    # EVIDENCE_EXCLUSIONS is off by default; this tests the switch turned on.
    encoder = AttributeEncoder(head_type="attribute",
                               config={"category_store_backend": "memory", "evidence_exclusions": True})
    text = "Binchotan is a traditional charcoal. Mike Wilks is an artist."
    doc = nlp.parse(text)
    # sentence 0 is a material artifact (object), sentence 1 a person
    ctx = {"spacy_doc": doc, "ontology_result": _ont(B.PERSON, {0: B.MATERIAL_ARTIFACT, 1: B.PERSON})}
    got = {a.concept for a in encoder.encode_batch_texts([text], [ctx])[0].addresses}
    assert "charcoal" in got and "artist" in got
    # the same text with sentence 0 read as a person loses the kind noun
    ctx2 = {"spacy_doc": doc, "ontology_result": _ont(B.PERSON, {0: B.PERSON, 1: B.PERSON})}
    got2 = {a.concept for a in encoder.encode_batch_texts([text], [ctx2])[0].addresses}
    assert "charcoal" not in got2 and "artist" in got2


def test_steering_by_record_ontology_when_sentence_steering_is_off(nlp):
    enc = AttributeEncoder(head_type="attribute",
                           config={"category_store_backend": "memory", "sentence_steering": False,
                                   "evidence_exclusions": True})
    text = "Binchotan is a traditional charcoal."
    doc = nlp.parse(text)
    ctx = {"spacy_doc": doc, "ontology_result": _ont(B.MATERIAL_ARTIFACT, {0: B.PERSON})}
    assert "charcoal" in {a.concept for a in enc.encode_batch_texts([text], [ctx])[0].addresses}


def test_stopwords_min_len_and_wordlist_apply_to_predication_values(nlp):
    text = "Bob Smith is an old American carpenter."
    doc = nlp.parse(text)
    base = AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})
    res = {a.concept for a in base.encode_batch_texts([text], [{"spacy_doc": doc}])[0].addresses}
    # "old" is a config stopword: penalised, and cut by the elbow next to
    # the strong values of the same predication
    assert "old" not in res and {"american", "carpenter"} <= res
    no_stop = AttributeEncoder(head_type="attribute",
                               config={"category_store_backend": "memory", "stopwords": ["xyz"]})
    assert "old" in {a.concept for a in no_stop.encode_batch_texts([text], [{"spacy_doc": doc}])[0].addresses}
    long_only = AttributeEncoder(head_type="attribute",
                                 config={"category_store_backend": "memory", "min_len": 8})
    got = {a.concept for a in long_only.encode_batch_texts([text], [{"spacy_doc": doc}])[0].addresses}
    assert got == {"american", "carpenter"}
    wl = AttributeEncoder(head_type="attribute",
                          config={"category_store_backend": "memory", "wordlist": ["smith"]})
    got = {a.concept: a for a in wl.encode_batch_texts([text], [{"spacy_doc": doc}])[0].addresses}
    assert got["smith"].metadata["method"] == "wordlist"


def test_lemma_category_changes_the_selection(nlp, monkeypatch):
    """The acat category now reaches the address: a category the ontology
    weighs down is cut by the per-subject elbow."""
    from nam_heads.impls.attribute.default import definitions as D
    text = "Bob Smith is a tall American carpenter."
    doc = nlp.parse(text)
    enc = AttributeEncoder(head_type="attribute", config={"category_store_backend": "memory"})
    ctx = {"spacy_doc": doc, "ontology_result": _ont(B.PERSON, {0: B.PERSON})}
    assert "tall" in {a.concept for a in enc.encode_batch_texts([text], [ctx])[0].addresses}
    monkeypatch.setitem(D.ONTOLOGY_RELEVANCE[B.PERSON], "physical", 0.2)
    assert "tall" not in {a.concept for a in enc.encode_batch_texts([text], [ctx])[0].addresses}


def test_per_subject_bound(nlp):
    enc = AttributeEncoder(head_type="attribute",
                           config={"category_store_backend": "memory", "max_attributes_per_subject": 2})
    text = "Joel Greenblatt is an American academic, manager, investor and writer."
    res = enc.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert len([a for a in res.addresses if a.concept != "__null__"]) == 2


def test_legacy_mode_still_available(nlp):
    enc = AttributeEncoder(head_type="attribute",
                           config={"category_store_backend": "memory", "predication_mode": False})
    text = "The big red pump is offline due to low oil."
    res = enc.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert res.metadata["extraction"] != "predication"
    assert len(res.addresses) <= enc.max_attributes


# ---------------------------------------------------------------- bundler

def _entity(concept, surface, etype=None):
    meta = {"surface_form": surface, "meaningful": True}
    if etype:
        meta["entity_type"] = etype
    return SemanticAddress("entity", concept, 1.0, meta)


def _bundle_attrs(nlp, encoder, text, entities, conditioning, ontology=B.PERSON):
    import os
    os.environ["NAM_ATTRIBUTE_CONDITIONING"] = "1" if conditioning else "0"
    from src.addressing.bundler import EntityBundler
    from src.addressing.light_doc import doc_to_light
    doc = nlp.parse(text)
    attr = encoder.encode_batch_texts([text], [{"spacy_doc": doc}])[0]
    enc = {"entity": EncodingResult("entity", [_entity(c, s, t) for c, s, t in entities], {}),
           "attribute": attr,
           "affordance": EncodingResult("affordance", [SemanticAddress("affordance", "__null__")], {}),
           "context": EncodingResult("context", [SemanticAddress("context", "__null__")], {}),
           "ontology": _ont(ontology)}
    try:
        bundles = EntityBundler().bundle(enc, doc_to_light(doc))
    finally:
        os.environ.pop("NAM_ATTRIBUTE_CONDITIONING", None)
    return {b.entity: b.attributes for b in bundles}


def test_bundler_links_attributes_by_anchor_per_entity(nlp, encoder):
    text = "Albert Speer was a German architect. He met Adolf Hitler in Berlin."
    got = _bundle_attrs(nlp, encoder, text,
                        [("e:speer", "Albert Speer", "person"), ("e:hitler", "Adolf Hitler", "person")],
                        conditioning=False)
    assert set(got["e:speer"]) == {"architect", "german"}
    assert got["e:hitler"] == ["__null__"]                # nothing was predicated of him


def test_bundler_conditioning_uses_the_entity_ontology(nlp, encoder):
    # The entity's class is the record's (ontology head); a mis-ontologised
    # record (a film typed Person) loses the kind noun only when conditioned.
    text = "Mad Love is a period drama film."
    ents = [("e:ml", "Mad Love", None)]
    assert "film" in _bundle_attrs(nlp, encoder, text, ents, conditioning=False)["e:ml"]
    conditioned = _bundle_attrs(nlp, encoder, text, ents, conditioning=True)["e:ml"]
    assert "film" not in conditioned and "drama" in conditioned
    assert "film" in _bundle_attrs(nlp, encoder, text, ents, conditioning=True,
                                   ontology=B.MATERIAL_ARTIFACT)["e:ml"]


def test_lone_noun_conjunct_shares_the_compound_head(nlp):
    vals = values(nlp, "Robert Kral (born 1967) is an Australian film and television composer.")
    assert "noun_mod" in vals["film"].kinds and "composer" in vals
    vals = values(nlp, "Mary Lee is a singer and film producer.")
    assert "role_noun" in vals["singer"].kinds   # a person noun keeps its own head


def test_lowercase_propn_is_a_common_noun_head(nlp):
    vals = values(nlp, "Aki Jones (1982 - 2014) was an American football player, who played for Washington.")
    assert "player" in vals and "role_noun" in vals["player"].kinds
    assert "noun_mod" in vals["football"].kinds


def test_bundler_orders_by_head_emission_and_bounds(nlp, encoder, monkeypatch):
    text = "Joel Greenblatt is an American academic, hedge fund manager, investor and writer."
    ents = [("e:jg", "Joel Greenblatt", "person")]
    got = _bundle_attrs(nlp, encoder, text, ents, conditioning=False)["e:jg"]
    attr = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    emitted = [a.concept for a in attr.addresses]
    assert got == [c for c in emitted if c in got]          # head emission order
    monkeypatch.setenv("NAM_ATTRIBUTE_MAX_PER_BUNDLE", "2")
    assert len(_bundle_attrs(nlp, encoder, text, ents, conditioning=False)["e:jg"]) == 2


def test_bundler_keeps_the_dependency_walk_as_fallback(nlp):
    """A legacy (unanchored) value still links through the amod walk."""
    from src.addressing.bundler import EntityBundler
    from src.addressing.light_doc import doc_to_light
    text = "The red ball rolled."
    doc = nlp.parse(text)
    ball = [t.i for t in doc if t.text == "ball"][0]
    red = [t.i for t in doc if t.text == "red"][0]
    # Precondition on the in-repo, deterministic rule parser: a parser change
    # that drops the attachment must fail here, not skip the walk test.
    assert doc[red].head.i == ball, "rule parser no longer attaches red -> ball"
    enc = {"entity": EncodingResult("entity", [_entity("e:ball", "ball", "object")], {}),
           "attribute": EncodingResult("attribute", [SemanticAddress("attribute", "red", 1.0, {})], {}),
           "affordance": EncodingResult("affordance", [SemanticAddress("affordance", "__null__")], {}),
           "context": EncodingResult("context", [SemanticAddress("context", "__null__")], {}),
           "ontology": _ont("object")}
    bundles = EntityBundler().bundle(enc, doc_to_light(doc))
    assert bundles[0].attributes == ["red"]


@pytest.mark.parametrize("text,want", [
    # a participle before a proper adjective or an adjective (the tagger:
    # retired/VERB Iranian/PROPN, retired/VERB professional/ADJ)
    ("Hassan Nazari is a retired Iranian football defender born in 1956 in Abadan, Iran.",
     {"retired", "defender"}),
    ("Jones is a retired professional footballer.", {"retired", "footballer"}),
    ("Lesotho is a landlocked country.", {"landlocked", "country"}),
])
def test_participle_opening_the_predicate(nlp, text, want):
    got = values(nlp, text)
    assert want <= set(got), sorted(got)


def test_participle_before_a_name_is_not_a_kind(nlp):
    preds = P.extract_predications(nlp.parse("It is a renamed Paris."), "")
    assert not any(p.link == "copular" for p in preds)


_KINDS = {"genus", "asteroid", "capital", "district", "novel"}


def _is_kind(v):
    return True if v in _KINDS else (False if v in {"life", "case"} else None)


def _bound(nlp, text):
    doc = nlp.parse(text)
    return {av.value: (av.link, subject(doc, av)) for av in P.attribute_values(doc, is_kind=_is_kind)}


@pytest.mark.parametrize("text,value,name", [
    ("They are related to frogs in the genus Huia today.", "genus", "Huia"),
    ("It lies west of the district capital Düren.", "capital", "Düren"),
    ("It shares orbits with the large asteroid Juno, which spans 235 km.", "asteroid", "Juno"),
])
def test_a_known_kind_before_a_name_is_its_kind(nlp, text, value, name):
    got = _bound(nlp, text)
    assert got[value][0] == "kind_of" and name in got[value][1], got


@pytest.mark.parametrize("text", [
    "In his life Afonso travelled.",                 # not a kind
    "She wrote romance novels J. Smith admired.",    # plural
])
def test_no_kind_before_a_name(nlp, text):
    assert _bound(nlp, text) == {}


def test_unknown_noun_before_a_name_stays_out(nlp):
    doc = nlp.parse("They are related to frogs in the genus Huia today.")
    assert "genus" not in {av.value for av in P.attribute_values(doc, is_kind=lambda v: None)}


def test_person_titles_unchanged(nlp):
    assert _bound(nlp, "The album features the guitarist Andy Lee.")["guitarist"][0] == "title"
