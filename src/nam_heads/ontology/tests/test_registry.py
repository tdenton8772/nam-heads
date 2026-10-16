import json

import pytest

from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import BRANCHES, CLASSES_PATH, OntologyRegistry, get_registry


@pytest.fixture(scope="module")
def reg():
    return get_registry()


def test_sources_pinned(reg):
    assert set(reg.sources) == {"bfo", "cco"}
    for s in reg.sources.values():
        assert len(s["sha256"]) == 64 and s["url"].startswith("https://")


def test_every_named_class_is_a_class(reg):
    for c in B.named_classes():
        assert reg.is_class(c), c
    assert B.UNCLASSIFIED == B.ENTITY
    assert not any(c.startswith("http") for c in B.named_classes())


def test_labels_match_the_standard(reg):
    assert reg.label(B.PERSON) == "Person"
    assert reg.label(B.PROCESS) == "process"
    assert reg.label(B.ENTITY) == "entity"
    assert reg.iri(B.PERSON) == "https://www.commoncoreontologies.org/ont00001262"
    assert reg.iri(B.PROCESS) == "http://purl.obolibrary.org/obo/BFO_0000015"
    assert reg.label("cco:unknown") == "cco:unknown"


def test_person_chain(reg):
    # Person < Animal < Organism < object < material entity < independent continuant
    a = reg.ancestors(B.PERSON)
    for c in ("cco:ont00000562", "cco:ont00000551", "BFO:0000030", "BFO:0000040",
              B.INDEPENDENT_CONTINUANT, B.CONTINUANT, B.ENTITY):
        assert c in a
    assert B.PERSON not in a and B.PERSON in reg.ancestors(B.PERSON, inclusive=True)


def test_is_a_and_descendants(reg):
    assert reg.is_a(B.ACT_OF_REQUESTING, B.ACT)
    assert reg.is_a(B.ACT, B.PROCESS)
    assert not reg.is_a(B.PROCESS, B.ACT)
    assert reg.is_a(B.PERSON, B.PERSON)
    assert B.ORGANIZATION in reg.descendants("cco:ont00000300")  # Group of Agents
    assert B.GROUP_OF_PERSONS in reg.descendants("cco:ont00000300")
    assert "cco:ont00000300" not in reg.descendants("cco:ont00000300")
    assert set(reg.descendants(B.ENTITY)) == set(reg.classes()) - {B.ENTITY}


def test_branches(reg):
    assert reg.branch(B.PERSON) == B.INDEPENDENT_CONTINUANT
    assert reg.branch(B.GEOSPATIAL_REGION) == B.INDEPENDENT_CONTINUANT
    assert reg.branch(B.ACT) == B.OCCURRENT
    assert reg.branch(B.TEMPORAL_REGION) == B.OCCURRENT
    assert reg.branch(B.RELATIONAL_QUALITY) == B.SPECIFICALLY_DEPENDENT_CONTINUANT
    assert reg.branch(B.DOCUMENT_CONTENT_ENTITY) == B.GENERICALLY_DEPENDENT_CONTINUANT
    assert reg.branch(B.ENTITY) == B.ENTITY


def test_disjointness_from_asserted_axioms(reg):
    # continuant vs occurrent
    assert reg.disjoint(B.PERSON, B.PROCESS)
    assert reg.disjoint(B.ACT, B.GEOSPATIAL_REGION)
    # material vs immaterial entity
    assert reg.disjoint(B.PERSON, B.GEOSPATIAL_REGION)
    # independent vs generically dependent continuant
    assert reg.disjoint(B.ORGANIZATION, B.INFORMATION_CONTENT_ENTITY)
    # temporal region vs process
    assert reg.disjoint(B.TEMPORAL_REGION, B.ACT)
    # CCO: descriptive vs prescriptive ICE
    assert reg.disjoint(B.MEASUREMENT_ICE, B.OBJECTIVE)
    # symmetric
    assert reg.disjoint(B.PROCESS, B.PERSON)


def test_not_disjoint(reg):
    assert not reg.disjoint(B.PERSON, B.PERSON)
    assert not reg.disjoint(B.ACT, B.PROCESS)          # subclass
    assert not reg.disjoint(B.ENTITY, B.PERSON)        # root
    # No OWL axiom separates object from object aggregate.
    assert not reg.disjoint(B.PERSON, B.GROUP_OF_PERSONS)


def test_resolve_majority_wins(reg):
    assert reg.resolve([B.PERSON, B.GEOSPATIAL_REGION, B.GEOSPATIAL_REGION]) == B.GEOSPATIAL_REGION


def test_resolve_tie_goes_to_first(reg):
    assert reg.resolve([B.GROUP_OF_PERSONS, B.PERSON]) == B.GROUP_OF_PERSONS
    assert reg.resolve([B.PERSON, B.GROUP_OF_PERSONS]) == B.PERSON


def test_resolve_refines_to_subsumed_vote(reg):
    assert reg.resolve([B.PROCESS, B.PROCESS, B.ACT]) == B.ACT
    assert reg.resolve([B.ENTITY, B.PERSON]) == B.PERSON


def test_resolve_ignores_unknown_and_empty(reg):
    assert reg.resolve([]) is None
    assert reg.resolve(["person", "cco:nope"]) is None
    assert reg.resolve(["person", B.ORGANIZATION]) == B.ORGANIZATION


def test_tiers_follow_bfo_branches(reg):
    t = reg.tiers([B.ACT, B.PERSON, B.ENTITY, B.QUALITY, B.DOCUMENT_CONTENT_ENTITY, B.PROCESS])
    assert t[0][0] == B.ENTITY and B.PERSON in t[0]
    assert t[1] == [B.PROCESS, B.ACT]                  # shallower first
    assert t[2] == [B.QUALITY]
    assert t[3] == [B.DOCUMENT_CONTENT_ENTITY]
    assert reg.tiers([B.ENTITY]) == [[B.ENTITY]]
    assert reg.tiers([B.ACT]) == [[B.ACT]]


def test_registry_rejects_missing_binding_class():
    doc = json.loads(CLASSES_PATH.read_text())
    del doc["classes"][B.PERSON]
    with pytest.raises(ValueError, match="parent|bindings"):
        OntologyRegistry(doc)


def test_registry_rejects_dangling_parent():
    doc = json.loads(CLASSES_PATH.read_text())
    doc["classes"]["cco:x"] = {"iri": "x", "label": "x", "definition": "", "parents": ["cco:missing"]}
    with pytest.raises(ValueError, match="parent"):
        OntologyRegistry(doc)


def test_curies_have_no_key_separator(reg):
    for c in reg.classes():
        assert "|" not in c


def test_generator_curie():
    from nam_heads.ontology.generate import curie
    assert curie("http://purl.obolibrary.org/obo/BFO_0000015") == "BFO:0000015"
    assert curie("https://www.commoncoreontologies.org/ont00001262") == "cco:ont00001262"
    assert curie("http://example.org/x") == ""


def test_base_class_curies_are_short(reg):
    """A class CURIE is the address key's type segment; NAM's
    test_ontology_key_cap.py checks full keys against the 250-byte cap."""
    assert len(max(reg.classes(), key=len)) <= 16


def test_tbox_is_hermit_certified_and_complete():
    doc = json.loads(CLASSES_PATH.read_text())
    cert = doc["certification"]
    assert cert["consistent"] is True and cert["unsatisfiable"] == []
    # every labelled class of the merged BFO 2020 + CCO v2.2 TBox
    assert len(doc["classes"]) == 1437


def test_nearest_key_walks_ancestors(reg):
    city = "cco:ont00000887"
    table = {B.GEOSPATIAL_REGION: 1, B.PERSON: 2, B.ENTITY: 3}
    assert reg.nearest_key(city, table) == B.GEOSPATIAL_REGION
    assert reg.nearest_key(B.PERSON, table) == B.PERSON
    assert reg.nearest_key(B.ACT, table) is None          # root is never inherited
    assert reg.nearest_key(B.ENTITY, table) == B.ENTITY   # but matches itself
    assert reg.nearest_key("person", table) is None       # labels are not classes
    # deeper keyed ancestor wins: Act of Ownership has its own row before Act's
    assert reg.nearest_key(B.ACT_OF_OWNERSHIP, {B.ACT: 1, B.ACT_OF_OWNERSHIP: 2}) == B.ACT_OF_OWNERSHIP


def test_resolve_weighted_votes(reg):
    # one confident Person sentence outweighs two weak Group sentences
    assert reg.resolve([B.GROUP_OF_PERSONS, B.PERSON, B.GROUP_OF_PERSONS],
                       weights=[0.2, 0.9, 0.3]) == B.PERSON
    assert reg.resolve([B.PERSON], weights=[0.0]) is None
