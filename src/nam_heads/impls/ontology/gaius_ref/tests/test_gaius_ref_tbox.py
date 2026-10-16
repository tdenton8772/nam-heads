"""The class side is gaius's: its TBox, its verbalization rule, no remapping."""

import json
from types import SimpleNamespace

import pytest

from nam_heads.impls.ontology.gaius_ref import tbox as T
from nam_heads.ontology.registry import base_registry, get_registry


def cls(iri, label=(), comment=(), iao=None):
    o = SimpleNamespace(iri=iri, label=list(label), comment=list(comment))
    if iao is not None:
        o.IAO_0000115 = list(iao)
    return o


# ---- gaius's rules (agents/theta/tbox.py:39-65) -------------------------------

def test_label_is_the_first_label_else_the_iri_local_name():
    assert T._label_of(cls("https://x.org/a#Thing1", ["first", "second"])) == "first"
    assert T._label_of(cls("https://signals.zndx.org/sdg#hasChecksum")) == "hasChecksum"
    assert T._label_of(cls("https://www.commoncoreontologies.org/ont00001325/")) == "ont00001325"


def test_text_is_label_dot_definition_without_a_period_added():
    c = cls("https://x/A", ["artifact"], iao=["An object made by an Agent"])
    assert T._retrieval_text(c, "artifact") == "artifact. An object made by an Agent"


def test_every_iao_definition_is_joined_by_a_space_and_beats_comments():
    c = cls("https://x/A", ["a"], comment=["a comment"], iao=["one.", "  ", "two"])
    assert T._retrieval_text(c, "a") == "a. one. two"


def test_without_a_definition_the_first_nonblank_comment_is_used():
    c = cls("https://x/A", ["a"], comment=["  ", "first comment", "second"])
    assert T._retrieval_text(c, "a") == "a. first comment"


def test_without_definition_or_comment_the_text_is_the_label_exactly():
    assert T._retrieval_text(cls("https://x/A", ["Person"]), "Person") == "Person"


def test_nothing_is_lowercased_or_normalised():
    c = cls("https://x/A", ["MiXed"], iao=["Keeps  double spaces"])
    assert T._retrieval_text(c, "MiXed") == "MiXed. Keeps  double spaces"


def test_thing_and_nothing_are_skipped():
    assert T._skip("http://www.w3.org/2002/07/owl#Thing")
    assert T._skip("http://www.w3.org/2002/07/owl#Nothing")
    assert not T._skip("https://signals.zndx.org/sdg#Person")


# ---- the artifact ----------------------------------------------------------------

def test_artifact_is_the_pinned_sdg_tbox():
    doc = T.load()
    src = doc["source"]
    assert src["gaius_commit"] == T.GAIUS_COMMIT
    assert src["sdg_corpora_commit"] == T.SDG_CORPORA_COMMIT
    assert src["owl_sha256"] == T.OWL_SHA256 and src["import_sha256"] == T.IMPORT_SHA256
    rows = doc["classes"]
    # HERMIT_CERTIFICATE.md: 918 named classes; gaius's loader drops the class
    # typing of the 13 property/class puns (tbox.py:35-82)
    assert len(rows) == 905
    hosts = {}
    for r in rows:
        host = r["iri"].split("/")[2]
        hosts[host] = hosts.get(host, 0) + 1
    assert hosts == {"signals.zndx.org": 735, "www.commoncoreontologies.org": 154,
                     "purl.obolibrary.org": 14, "www.w3.org": 2}


def test_artifact_texts_follow_the_rule():
    for r in T.load()["classes"]:
        assert r["text"] == r["label"] or r["text"].startswith(r["label"] + ". ")
        assert r["text"] == r["text"].strip()


def test_known_classes_read_as_gaius_reads_them():
    texts = {r["iri"]: r["text"] for r in T.load()["classes"]}
    # SDG's own label for CCO Person comes first; CCO's has no IAO_0000115
    assert texts["https://www.commoncoreontologies.org/ont00001262"] == "person"
    # an sdg:-declared CCO class with a placeholder label stays its local name
    assert texts["https://www.commoncoreontologies.org/ont00001325"] == "ont00001325"
    assert texts["https://signals.zndx.org/sdg#MolecularDynamicsSimulationProcess"].startswith(
        "molecular dynamics simulation process. A MolecularDynamicsSimulationProcess is a process")
    assert texts["http://www.w3.org/2001/XMLSchema#string"] == "string"


PUNNED = ["http://purl.obolibrary.org/obo/BFO_0000054", "http://purl.obolibrary.org/obo/BFO_0000055",
          "https://signals.zndx.org/sdg#bindsProvider", "https://signals.zndx.org/sdg#hasChecksum",
          "https://signals.zndx.org/sdg#hasConfidence", "https://signals.zndx.org/sdg#hasCreatedDate",
          "https://signals.zndx.org/sdg#hasDimensionKind", "https://signals.zndx.org/sdg#hasEffectiveDate",
          "https://signals.zndx.org/sdg#hasEncoding", "https://signals.zndx.org/sdg#hasEnforcement",
          "https://signals.zndx.org/sdg#hasLabelText", "https://signals.zndx.org/sdg#publishes_journal",
          "https://www.commoncoreontologies.org/ont00001787"]


def test_the_13_punned_properties_are_not_classes():
    doc = T.load()
    assert doc["source"]["punned_class_typing_dropped"] == PUNNED
    iris = {r["iri"] for r in doc["classes"]}
    assert not iris & set(PUNNED)
    ext = T.load_extension()
    assert not {v["iri"] for v in ext["classes"].values()} & set(PUNNED)


def test_artifact_order_is_fixed_by_iri():
    """gaius's own enumeration order changes with every fresh .owlready cache
    (rdflib writes it in hash-seed order); the artifact keeps one order."""
    iris = [r["iri"] for r in T.load()["classes"]]
    assert iris == sorted(iris)


# ---- gaius's loaders (tbox.py:35-82, 128-149), on a tiny OWL and on the real one ------

TINY_OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#">
  <owl:Ontology rdf:about="https://example.org/tiny">
    <owl:imports rdf:resource="https://example.org/tiny/imports/cco-module"/>
  </owl:Ontology>
  <owl:Class rdf:about="https://example.org/tiny#Thingy"><rdfs:label>thingy</rdfs:label>
    <rdfs:subClassOf rdf:resource="https://example.org/mod#Base"/></owl:Class>
  <owl:Class rdf:about="https://example.org/tiny#hasPart"/>
  <owl:ObjectProperty rdf:about="https://example.org/tiny#hasPart"/>
  <owl:Class rdf:about="https://example.org/mod#Base"/>
</rdf:RDF>
"""
TINY_TTL = """@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<https://example.org/tiny/imports/cco-module> a owl:Ontology .
<https://example.org/mod#Base> a owl:Class ; rdfs:label "base from the import" .
"""


@pytest.fixture
def owlready2_path():
    owlready2 = pytest.importorskip("owlready2")
    pytest.importorskip("rdflib")
    before = list(owlready2.onto_path)
    yield owlready2
    owlready2.onto_path[:] = before


def test_loaders_drop_puns_and_resolve_the_import_locally(tmp_path, owlready2_path):
    (tmp_path / "imports").mkdir()
    owl = tmp_path / "tiny.owl"
    owl.write_text(TINY_OWL)
    (tmp_path / "imports" / "cco-module.ttl").write_text(TINY_TTL)
    T.load_pinned_imports(owl)
    ready = T.ontology_for_owlready(owl)
    assert ready == tmp_path / ".owlready" / "tiny.nopun.owl"
    assert (tmp_path / ".owlready" / "cco-module").is_file()
    onto = owlready2_path.World().get_ontology(ready.resolve().as_uri()).load()
    iris = {c.iri for c in onto.classes()}
    assert "https://example.org/tiny#hasPart" not in iris               # the pun is not a class
    assert "https://example.org/tiny#Thingy" in iris
    base = onto.world["https://example.org/mod#Base"]
    assert list(base.label) == ["base from the import"]                # read from the local import
    assert T.punned_iris(owl) == ["https://example.org/tiny#hasPart"]


def test_extract_on_the_real_pinned_owl(owlready2_path):
    """The real certified OWL through gaius's loaders gives exactly the
    committed artifact. Set NAM_SDG_CORPORA_DIR to an sdg-corpora checkout at
    the pinned commit; NAM_GAIUS_REF_REQUIRED=1 makes its absence a failure."""
    import os
    from pathlib import Path

    root = os.environ.get("NAM_SDG_CORPORA_DIR")
    owl = Path(root) / "ontology" / "sdg-ontology.owl" if root else None
    if owl is None or not owl.is_file():
        if os.environ.get("NAM_GAIUS_REF_REQUIRED") == "1":
            pytest.fail("NAM_GAIUS_REF_REQUIRED=1 but NAM_SDG_CORPORA_DIR has no ontology/sdg-ontology.owl")
        pytest.skip("set NAM_SDG_CORPORA_DIR to an sdg-corpora checkout at %s" % T.SDG_CORPORA_COMMIT[:8])
    assert T.file_sha256(owl) == T.OWL_SHA256
    doc = T.extract(owl)
    assert len(doc["classes"]) == 905
    committed = T.load()
    assert [(r["iri"], r["label"], r["text"]) for r in doc["classes"]] == \
           [(r["iri"], r["label"], r["text"]) for r in committed["classes"]]
    assert doc["source"]["punned_class_typing_dropped"] == PUNNED


def test_load_refuses_another_pin(tmp_path):
    doc = json.loads(T.TBOX_PATH.read_text())
    doc["source"]["sdg_corpora_commit"] = "0" * 40
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(T.TBoxError):
        T.load.__wrapped__(str(p))


def test_load_refuses_an_empty_class_text(tmp_path):
    doc = json.loads(T.TBOX_PATH.read_text())
    doc["classes"][0]["text"] = " "
    p = tmp_path / "t.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(T.TBoxError):
        T.load.__wrapped__(str(p))


# ---- CURIEs: the same IRI, never another class ---------------------------------------

def test_curie_is_the_registrys_own_for_a_registry_iri():
    reg = get_registry()
    ri = T.registry_iris()
    assert T.curie(reg.iri("cco:ont00001262"), ri) == "cco:ont00001262"
    assert T.curie(reg.iri("BFO:0000040"), ri) == "BFO:0000040"


def test_curie_keeps_classes_outside_the_registry():
    ri = T.registry_iris()
    assert T.curie("https://signals.zndx.org/sdg#Individual", ri) == "sdg:Individual"
    assert T.curie("http://www.w3.org/2001/XMLSchema#string", ri) == "xsd:string"
    assert T.curie("https://example.org/x#Y", ri) == "https://example.org/x#Y"


def test_every_tbox_class_has_a_distinct_curie():
    ri = T.registry_iris()
    curies = [T.curie(r["iri"], ri) for r in T.load()["classes"]]
    assert len(set(curies)) == len(curies)
    reg = get_registry()
    for r, c in zip(T.load()["classes"], curies):
        if reg.is_class(c):
            assert reg.iri(c) == r["iri"]


# ---- the registry extension (artifacts/sdg_ontology.json) ------------------------------

def test_extension_holds_exactly_the_tbox_classes_bfo_cco_lacks():
    ext = T.load_extension()
    base = base_registry()
    base_iris = {base.iri(c) for c in base.classes()}
    rows = T.load()["classes"]
    assert {v["iri"] for v in ext["classes"].values()} == {r["iri"] for r in rows if r["iri"] not in base_iris}
    assert len(ext["emittable"]) == len(rows) == 905
    assert ext["base_sources"] == base.sources
    assert ext["source"]["owl_sha256"] == T.OWL_SHA256


def test_extension_labels_are_gaius_labels():
    rows = {r["iri"]: r for r in T.load()["classes"]}
    for c, v in T.load_extension()["classes"].items():
        assert v["label"] == rows[v["iri"]]["label"]
        text = rows[v["iri"]]["text"]
        assert text == (f"{v['label']}. {v['definition']}" if v["definition"] else v["label"])


def test_extension_parents_keep_every_parent_gaius_reads():
    """gaius's tbox.py reads is_a; the registry adds the named conjuncts of
    equivalences and never drops an is_a parent."""
    to_curie = T.iri_to_curie()
    base_iris = {base_registry().iri(c): c for c in base_registry().classes()}
    rows = {r["iri"]: r for r in T.load()["classes"]}
    for c, v in T.load_extension()["classes"].items():
        for p in rows[v["iri"]]["parents"]:
            if p.startswith("http://www.w3.org/2002/07/owl#"):
                continue                                  # a punned property's owl:ObjectProperty
            assert (to_curie.get(p) or base_iris.get(p)) in v["parents"], (c, p)


def test_extension_hierarchy_is_acyclic_and_grounded():
    reg = base_registry().extended(T.load_extension())
    own = reg.extension["classes"]
    for c in own:
        assert c not in reg.ancestors(c)
    grounded = sum(1 for c in own if reg.branch(c) != "BFO:0000001")
    assert grounded == 505                               # of 737, under a BFO branch


def test_iri_to_curie_covers_the_tbox():
    m = T.iri_to_curie()
    assert set(m) == {r["iri"] for r in T.load()["classes"]}
    assert len(set(m.values())) == 905
