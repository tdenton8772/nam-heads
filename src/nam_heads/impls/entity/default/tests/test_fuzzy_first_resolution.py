"""Tests for fuzzy-first entity resolution in EntityEncoder."""
import os
import re
import pytest
from unittest.mock import patch

from nam_heads.impls.entity.default.head import EntityEncoder, _hash_entity_id
from nam_heads.impls.entity.default.storage.memory_store import MemoryEntityStore
from nam_heads.impls.entity.default.comparators import get_comparator


@pytest.fixture
def store():
    return MemoryEntityStore()


@pytest.fixture
def encoder(store):
    """Create an EntityEncoder with memory store and no parser."""
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 10
    enc.create_new_entities = True
    enc.min_confidence = 0.35
    enc.enable_fuzzy = True
    enc.fuzzy_max_candidates = 50
    enc.nlp = None
    enc.store = store
    enc.config = {}
    return enc


def _register_entity(store, entity_id, surface_forms, entity_type):
    """Helper: create entity and register fuzzy candidate keys.

    Registers keys with BOTH the entity's own comparator and the default
    comparator, to ensure resolution works regardless of the codebook type
    assigned at query time.
    """
    store.create_entity(
        entity_id=entity_id,
        surface_forms=surface_forms,
        metadata={"entity_type": entity_type},
    )
    # Register with own type comparator + default (object) comparator
    comparators = [get_comparator(entity_type)]
    if entity_type != "object":
        comparators.append(get_comparator("object"))
    for comp in comparators:
        for sf in surface_forms:
            keys = comp.generate_keys(sf)
            if keys:
                store.store_candidate_keys(entity_id, sf, keys)


def test_fuzzy_resolves_existing_entity(store, encoder):
    """Query: a fuzzy store hit is kept as an alternate candidate; the
    primary concept is hash(canonical), where ingest writes the mention."""
    _register_entity(store, "abc123def456", ["albert_einstein", "einstein"], "person")

    result = encoder.encode_query("Albert Einstein")
    assert result.addresses, "Should resolve at least one address"
    resolved_ids = [a.concept for a in result.addresses]
    alt_ids = [eid for a in result.addresses
               for eid in (a.metadata or {}).get("alt_entity_ids", [])]
    assert "abc123def456" in alt_ids, (
        f"Expected abc123def456 among alternates, got {alt_ids}"
    )
    assert _hash_entity_id("", "albert_einstein") in resolved_ids


def test_fuzzy_miss_creates_hash_id(store, encoder):
    """When fuzzy finds no match, ingest creates a hash-based entity ID."""
    result = encoder.encode("Abraham Lincoln was the 16th president.")
    assert result.addresses, "Should create at least one address"
    for addr in result.addresses:
        # All entity IDs should be hash-based: 12 hex chars (no type prefix)
        eid = addr.concept
        assert re.match(r"^[a-f0-9]{12}$", eid), f"Not a hash ID: {eid}"


def test_same_canonical_same_hash():
    """Two pods processing the same canonical produce the same hash ID."""
    id1 = _hash_entity_id("person", "abraham_lincoln")
    id2 = _hash_entity_id("person", "abraham_lincoln")
    assert id1 == id2


def test_query_path_uses_hash_id(store, encoder):
    """Query path emits hash-based IDs for unresolved entities."""
    result = encoder.encode_query("Nikola Tesla invented the AC motor")
    # With no entities in store, query should emit unresolved addresses
    for addr in result.addresses:
        eid = addr.concept
        assert re.match(r"^[a-f0-9]{12}$", eid), f"Not a hash ID: {eid}"


def test_ingest_query_convergence(store, encoder):
    """Ingest creates entity, query resolves to same entity via fuzzy."""
    # Ingest creates entity
    ingest_result = encoder.encode("Albert Einstein developed the theory of relativity.")
    ingest_addresses = ingest_result.addresses
    assert ingest_addresses, "Ingest should produce addresses"

    # Find the einstein entity
    einstein_addr = None
    for addr in ingest_addresses:
        if "einstein" in addr.metadata.get("normalized", "").lower():
            einstein_addr = addr
            break

    if einstein_addr is None:
        # Maybe it's labeled differently -- just use first one
        einstein_addr = ingest_addresses[0]

    ingest_entity_id = einstein_addr.concept

    # Query should resolve to the same entity
    query_result = encoder.encode_query("Albert Einstein")
    query_addresses = query_result.addresses
    assert query_addresses, "Query should produce addresses"

    # At least one address should match the ingest entity
    query_entity_ids = [a.concept for a in query_addresses]
    assert ingest_entity_id in query_entity_ids, (
        f"Query should resolve to same entity. Ingest: {ingest_entity_id}, "
        f"Query: {query_entity_ids}"
    )


def test_batch_fuzzy_resolution(store, encoder):
    """Batch encode ignores fuzzy hits that are not hash(canonical)."""
    _register_entity(store, "abc123def456", ["albert_einstein", "einstein"], "person")
    _register_entity(store, "def456ghi789", ["isaac_newton", "newton"], "person")

    results = encoder.encode_batch_texts([
        "Albert Einstein was a physicist.",
        "Isaac Newton discovered gravity.",
    ])
    assert len(results) == 2

    for result in results:
        assert result.addresses, "Each text should produce addresses"

    # Ingest emits the pure id: a store hit owned by another id never
    # replaces hash(canonical) (the id query and every other pod compute).
    einstein_ids = [a.concept for a in results[0].addresses]
    assert _hash_entity_id("", "albert_einstein") in einstein_ids, einstein_ids
    assert "abc123def456" not in einstein_ids, einstein_ids


def test_resolve_fuzzy_batch_on_memory_store(store):
    """Test resolve_fuzzy_batch on MemoryEntityStore."""
    _register_entity(store, "abc123def456", ["albert_einstein", "einstein"], "person")

    comp = get_comparator("person")
    results = store.resolve_fuzzy_batch([
        ("einstein", "person", comp),
        ("nonexistent_entity", "person", comp),
    ])
    assert results[0] == "abc123def456"
    assert results[1] is None


def test_type_independent_hash():
    """Entity hash is the same regardless of detected type.

    This ensures ingest (which might detect 'concept') and query (which
    might detect 'object') produce the same entity ID for the same
    canonical form.
    """
    h1 = _hash_entity_id("concept", "anarchism")
    h2 = _hash_entity_id("object", "anarchism")
    assert h1 == h2
    assert re.match(r"^[a-f0-9]{12}$", h1)


class TestSurfaceFormSimplification:
    """Surface forms store only canonical forms — no conjugations, no
    component expansion for common nouns."""

    def test_single_token_no_plural(self, encoder):
        """Single-token entity should produce only the canonical form."""
        forms = encoder._surface_forms_with_aliases("pump", entity_type="object")
        assert forms == ["pump"], f"Expected ['pump'], got {forms}"

    def test_proper_name_gets_components(self, encoder):
        """Person entities expand into individual name components (when enabled)."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases("albert_einstein", entity_type="person")
        assert "albert_einstein" in forms
        assert "albert" in forms
        assert "einstein" in forms
        # No plural variants
        assert "albert_einsteins" not in forms

    def test_common_noun_no_components(self, encoder):
        """Object entities do NOT expand into components."""
        forms = encoder._surface_forms_with_aliases("control_system", entity_type="object")
        assert forms == ["control_system"], f"Expected ['control_system'], got {forms}"

    def test_three_part_person_name_shortening(self, encoder):
        """Three-part person names get first_last shortening + components (when enabled)."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "martin_luther_king", entity_type="person",
            )
        assert "martin_luther_king" in forms
        assert "martin_king" in forms  # name shortening
        assert "martin" in forms
        assert "luther" in forms
        assert "king" in forms

    def test_three_part_object_no_expansion(self, encoder):
        """Three-part object names get no shortening or components."""
        forms = encoder._surface_forms_with_aliases(
            "central_processing_unit", entity_type="object",
        )
        assert forms == ["central_processing_unit"]

    def test_organization_gets_components(self, encoder):
        """Organization entities expand like person names (when enabled)."""
        with patch.dict(os.environ, {"NAM_ENTITY_COMPONENT_EXPANSION": "1"}):
            forms = encoder._surface_forms_with_aliases(
                "european_union", entity_type="organization",
            )
        assert "european_union" in forms
        assert "european" in forms
        assert "union" in forms

    def test_location_no_components(self, encoder):
        """Location entities do NOT expand components (avoid 'new' from 'new_york')."""
        forms = encoder._surface_forms_with_aliases("new_york", entity_type="location")
        assert forms == ["new_york"]

    def test_no_synonym_expansion(self, encoder):
        """Synonym variants should not be stored as surface forms."""
        forms = encoder._surface_forms_with_aliases("bicycle", entity_type="object")
        assert forms == ["bicycle"]
