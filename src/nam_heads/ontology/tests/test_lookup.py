"""Shared ontology-keyed table lookup (nam_heads/ontology/lookup.py).

Tables are keyed by BFO/CCO class; the ontology head emits only classes, so
there is no label path. bfo:entity (the head abstaining) is no evidence.
"""
from nam_heads.ontology import bindings as B
from nam_heads.ontology import lookup as L

TABLE = {B.MATERIAL_ARTIFACT: "artifact", B.ORGANIZATION: "org"}


def test_exact_class():
    assert L.lookup(TABLE, B.ORGANIZATION) == "org"
    assert L.lookup(TABLE, B.MATERIAL_ARTIFACT) == "artifact"


def test_subclass_inherits_nearest_key():
    assert L.lookup(TABLE, B.FACILITY) == "artifact"        # Facility is_a Material Artifact
    assert L.lookup(TABLE, "cco:ont00000443") == "org"      # Commercial Organization


def test_labels_are_not_resolved():
    # Lowercase labels are never emitted; they are not ontology evidence.
    assert L.lookup(TABLE, "object", default="d") == "d"
    assert L.lookup(TABLE, "organization", default="d") == "d"


def test_bfo_entity_is_neutral_never_a_match():
    root_keyed = {B.ENTITY: "never", B.ORGANIZATION: "org"}
    assert L.lookup(root_keyed, B.ENTITY, default="d") == "d"          # abstain
    assert L.lookup(root_keyed, B.TEMPORAL_REGION, default="d") == "d"  # no inheritance from root
    assert L.ontology_class(B.ENTITY) is None


def test_no_evidence_gives_default():
    assert L.lookup(TABLE, None, default=1) == 1
    assert L.lookup(TABLE, "not-a-class", default=1) == 1
    assert L.lookup({}, B.PERSON, default=1) == 1


def test_head_keys_stop_resolution_at_the_nearest_head_key():
    # With the head's ONTOLOGY_KEYS, a class resolves to the nearest key ANY
    # table has; a table without that key gives the default (as
    # _get_ontology_type + table.get does).
    keys = {B.FACILITY, B.MATERIAL_ARTIFACT}
    assert L.lookup(TABLE, B.FACILITY, default="d", keys=keys) == "d"
    assert L.lookup(TABLE, B.FACILITY, default="d") == "artifact"


def test_uses_registry_nearest_key(monkeypatch):
    class Reg:
        def is_class(self, c):
            return c.startswith(("cco:", "BFO:"))

        def nearest_key(self, c, keys):
            return B.ORGANIZATION if B.ORGANIZATION in keys else None

    monkeypatch.setattr(L, "_registry", lambda: Reg())
    assert L.lookup(TABLE, "cco:ont00000443") == "org"


def test_sentence_ontology_attach_alias():
    from nam_heads.sdk import sentence_ontology as S
    assert S.attach is S.attach_sentence_ontologies
    assert S.annotate_sentence_ontologies is S.attach_sentence_ontologies
