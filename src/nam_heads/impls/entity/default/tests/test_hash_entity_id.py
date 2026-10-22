"""Tests for hash-based entity ID generation."""
import re
import pytest

from nam_heads.impls.entity.default.head import _hash_entity_id


def test_hash_deterministic():
    """Same input always produces the same hash."""
    id1 = _hash_entity_id("person", "albert_einstein")
    id2 = _hash_entity_id("person", "albert_einstein")
    assert id1 == id2


def test_hash_different_inputs():
    """Different canonicals produce different hashes."""
    id1 = _hash_entity_id("person", "albert_einstein")
    id2 = _hash_entity_id("person", "isaac_newton")
    assert id1 != id2


def test_hash_format():
    """Entity ID is a 12-char hex hash (no type prefix)."""
    eid = _hash_entity_id("person", "albert_einstein")
    assert re.match(r"^[a-f0-9]{12}$", eid), f"Bad format: {eid}"


def test_hash_type_independent():
    """Different types with same canonical produce the SAME ID.

    Entity type is intentionally excluded from the hash to ensure
    ingest/query convergence when the ontology head classifies
    differently for article text vs query word.
    """
    id_person = _hash_entity_id("person", "einstein")
    id_object = _hash_entity_id("object", "einstein")
    id_concept = _hash_entity_id("concept", "einstein")
    assert id_person == id_object == id_concept


def test_hash_empty_input():
    """Empty normalized string still produces a valid ID."""
    eid = _hash_entity_id("object", "")
    assert re.match(r"^[a-f0-9]{12}$", eid)


def test_hash_unicode_input():
    """Unicode input produces a valid deterministic hash."""
    id1 = _hash_entity_id("person", "müller")
    id2 = _hash_entity_id("person", "müller")
    assert id1 == id2
    assert re.match(r"^[a-f0-9]{12}$", id1)
