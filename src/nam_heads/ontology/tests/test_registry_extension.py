"""An ontology implementation's own classes join the registry while it is
selected, without changing any BFO/CCO answer."""

import copy
import json
from pathlib import Path

import pytest

from nam_heads.ontology import active
from nam_heads.ontology import bindings as B
from nam_heads.ontology import registry as R
from nam_heads.ontology.lookup import lookup, nearest

SDG = (Path(__file__).resolve().parents[2] / "impls" / "ontology" / "gaius_ref"
       / "artifacts" / "sdg_ontology.json")


@pytest.fixture(autouse=True)
def builtin_registry():
    R.use_extension(None)
    yield
    R.use_extension(None)


def toy(**over):
    base = R.base_registry()
    doc = {
        "schema": R.EXTENSION_SCHEMA, "name": "toy", "base_sources": base.sources,
        "prefixes": {"toy": "https://example.org/toy#"},
        "classes": {
            "toy:Scientist": {"iri": "https://example.org/toy#Scientist", "label": "scientist",
                              "parents": [B.PERSON]},
            "toy:Physicist": {"iri": "https://example.org/toy#Physicist", "label": "physicist",
                              "parents": ["toy:Scientist"]},
            "toy:Loose": {"iri": "https://example.org/toy#Loose", "label": "loose", "parents": []},
        },
        "emittable": ["toy:Physicist", "toy:Scientist", "toy:Loose", B.PERSON],
    }
    doc.update(over)
    return doc


def write(tmp_path, doc, name="ext.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc))
    return str(p)


# ---- the default is the built-in registry -------------------------------------

def test_default_is_the_builtin_registry():
    assert R.get_registry() is R.base_registry()
    assert R.active_extension() is None and R.get_registry().extension is None


# ---- extended() ---------------------------------------------------------------------

def test_extension_classes_join_with_their_hierarchy():
    reg = R.base_registry().extended(toy())
    assert reg.is_class("toy:Physicist") and reg.label("toy:Physicist") == "physicist"
    assert reg.is_a("toy:Physicist", B.PERSON) and reg.is_a("toy:Physicist", B.ENTITY)
    assert reg.parents("toy:Physicist") == ["toy:Scientist"]
    assert reg.branch("toy:Physicist") == R.BRANCHES[0]
    assert reg.branch("toy:Loose") == B.ENTITY and reg.ancestors("toy:Loose") == frozenset()
    assert reg.extension["name"] == "toy" and reg.extension["classes"] == {
        "toy:Scientist", "toy:Physicist", "toy:Loose"}


def test_every_bfo_cco_answer_is_unchanged():
    base = R.base_registry()
    reg = base.extended(toy())
    assert reg.sources == base.sources
    for c in base.classes():
        assert reg.ancestors(c) == base.ancestors(c)
        assert (reg.label(c), reg.depth(c), reg.branch(c), reg.iri(c)) == \
               (base.label(c), base.depth(c), base.branch(c), base.iri(c))
    assert reg.disjoint(B.PERSON, B.PROCESS) == base.disjoint(B.PERSON, B.PROCESS)


def test_extension_class_backs_off_to_the_nearest_bfo_cco_key():
    reg = R.base_registry().extended(toy())
    assert reg.nearest_key("toy:Physicist", {B.PERSON, B.ORGANIZATION}) == B.PERSON
    assert reg.nearest_key("toy:Loose", {B.PERSON}) is None
    assert reg.resolve(["toy:Scientist", "toy:Physicist", "toy:Scientist"]) == "toy:Physicist"


@pytest.mark.parametrize("change,why", [
    (lambda d: d.update(schema="other/1"), "schema"),
    (lambda d: d.update(name="Bad Name"), "name"),
    (lambda d: d.update(base_sources={"bfo": {}}), "another BFO/CCO release"),
    (lambda d: d["classes"].update({B.PERSON: {"iri": "x", "label": "p", "parents": []}}), "redefines"),
    (lambda d: d["classes"].update({"toy:Dup": {"iri": R.base_registry().iri(B.PERSON),
                                                "label": "p", "parents": []}}), "IRI is a BFO/CCO"),
    (lambda d: d["classes"]["toy:Loose"].update(parents=["toy:Nowhere"]), "is not a class"),
    (lambda d: d.update(emittable=["toy:Nowhere"]), "emittable"),
    (lambda d: d.update(emittable=[]), "emittable"),
    (lambda d: d["classes"]["toy:Loose"].pop("label"), "iri and label"),
])
def test_bad_extensions_are_refused(change, why):
    doc = toy()
    change(doc)
    with pytest.raises(ValueError, match=why):
        R.base_registry().extended(doc)


def test_an_extended_registry_is_not_extended_again():
    with pytest.raises(ValueError, match="cannot be extended again"):
        R.base_registry().extended(toy()).extended(toy(name="toy2"))


# ---- use_extension ---------------------------------------------------------------------

def test_use_extension_swaps_the_process_registry_and_back(tmp_path):
    path = write(tmp_path, toy())
    assert R.use_extension(path) == "toy"
    assert R.active_extension() == "toy" and R.get_registry().is_class("toy:Physicist")
    assert R.use_extension(path) == "toy"                 # idempotent
    assert R.use_extension(None) is None
    assert R.get_registry() is R.base_registry() and not R.get_registry().is_class("toy:Physicist")


def test_a_bad_artifact_raises_and_leaves_the_registry(tmp_path):
    good = write(tmp_path, toy())
    R.use_extension(good)
    bad = write(tmp_path, toy(base_sources={}), "bad.json")
    with pytest.raises(ValueError):
        R.use_extension(bad)
    assert R.active_extension() == "toy"
    with pytest.raises(OSError):
        R.use_extension(str(tmp_path / "missing.json"))
    assert R.active_extension() == "toy"


def test_lookup_tables_resolve_extension_classes_only_while_active(tmp_path):
    table = {B.PERSON: "person-row"}
    assert lookup(table, "toy:Physicist", default="neutral") == "neutral"
    R.use_extension(write(tmp_path, toy()))
    assert nearest("toy:Physicist", table) == B.PERSON
    assert lookup(table, "toy:Physicist", default="neutral") == "person-row"
    assert lookup(table, "toy:Loose", default="neutral") == "neutral"


def test_active_enumerates_the_extension_emittable_classes(tmp_path):
    before = active.emittable_classes()
    R.use_extension(write(tmp_path, toy()))
    got = active.emittable_classes()
    assert {"toy:Physicist", "toy:Scientist", "toy:Loose"} <= set(got) and set(before) <= set(got)
    R.use_extension(None)
    assert active.emittable_classes() == before


def test_unknown_classes_are_counted_and_logged_once(caplog):
    n = R.dropped_classes["count"]
    with caplog.at_level("WARNING", logger="nam_heads.ontology.registry"):
        R.note_unknown_class("zz:Never", "test")
        R.note_unknown_class("zz:Never", "test")
    assert R.dropped_classes["count"] == n + 2
    assert sum("zz:Never" in r.getMessage() for r in caplog.records) <= 1


# ---- gaius_ref's SDG ontology ----------------------------------------------------------

def test_sdg_extension_loads_and_keeps_bfo_cco():
    doc = json.loads(SDG.read_text())
    base = R.base_registry()
    reg = base.extended(copy.deepcopy(doc))
    assert reg.extension["name"] == "sdg"
    assert len(reg.extension["classes"]) == 737 and len(reg.extension["emittable"]) == 905
    own = reg.extension["classes"]
    assert sum(c.startswith("sdg:") for c in own) == 735
    assert all(not base.is_class(c) for c in own)
    # 168 of gaius's 905 are BFO/CCO classes, keyed by their BFO/CCO CURIE
    assert sum(base.is_class(c) for c in reg.extension["emittable"]) == 168
    assert B.PERSON in reg.extension["emittable"] and B.ENTITY in reg.extension["emittable"]
    assert reg.is_a("sdg:Individual", "BFO:0000040")          # material entity
    for c in base.classes():
        assert reg.ancestors(c) == base.ancestors(c)
