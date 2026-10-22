"""Tests for LmdbEntityStore — reads from LMDB, queues writes for async session bucket publish."""

import pytest
from unittest.mock import MagicMock


class FakeLmdbCache:
    """Minimal mock of LmdbEntityCache for testing."""

    def __init__(self, data=None):
        self._data = data or {}

    def get(self, key):
        return self._data.get(key)

    def batch_get(self, keys):
        return {k: self._data.get(k) for k in keys}


@pytest.fixture
def store():
    from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
    cache = FakeLmdbCache({
        "surf::PERSON::Alice": b"ent-001",
        "surf::Alice": b"ent-001",
        "surf::ORG::Acme Corp": b"ent-002",
    })
    return LmdbEntityStore(cache)


class TestResolveSurfaceForm:
    def test_typed_hit(self, store):
        assert store.resolve_surface_form("Alice", "PERSON") == "ent-001"

    def test_untyped_hit(self, store):
        assert store.resolve_surface_form("Alice") == "ent-001"

    def test_miss(self, store):
        assert store.resolve_surface_form("Bob") is None

    def test_typed_miss(self, store):
        assert store.resolve_surface_form("Alice", "ORG") is None

    def test_org_typed_hit(self, store):
        assert store.resolve_surface_form("Acme Corp", "ORG") == "ent-002"


class TestResolveSurfaceFormsBatch:
    def test_batch_mixed(self, store):
        queries = [("Alice", "PERSON"), ("Bob", None), ("Acme Corp", "ORG")]
        result = store.resolve_surface_forms_batch(queries)
        assert result[("Alice", "PERSON")] == "ent-001"
        assert result[("Bob", None)] is None
        assert result[("Acme Corp", "ORG")] == "ent-002"


class TestCreateEntity:
    def test_queues_writes(self, store):
        store.create_entity("ent-003", ["Bob", "Robert"], {"entity_type": "PERSON"})
        writes = store.drain_pending_lmdb_writes()
        keys = [k for k, v in writes]
        # ent:: doc for entity resolution
        assert "ent::ent-003" in keys
        # Typed key for both surface forms
        assert "surf::PERSON::Bob" in keys
        assert "surf::PERSON::Robert" in keys
        # Unscoped key only for primary form
        assert "surf::Bob" in keys
        assert "surf::Robert" not in keys
        # surf:: values are entity id
        for k, v in writes:
            if k.startswith("surf::"):
                assert v == b"ent-003"

    def test_multi_token_secondary_gets_untyped_key(self, store):
        """Multi-token secondary surface forms get untyped keys for cross-type resolution."""
        store.create_entity(
            "ent-100",
            ["president_abraham_lincoln", "abraham_lincoln", "lincoln"],
            {"entity_type": "titled_entity"},
        )
        writes = store.drain_pending_lmdb_writes()
        keys = [k for k, v in writes]
        # Primary always gets untyped
        assert "surf::president_abraham_lincoln" in keys
        # Multi-token secondary also gets untyped
        assert "surf::abraham_lincoln" in keys
        # Single-token secondary does NOT get untyped
        assert "surf::lincoln" not in keys
        # All get typed
        assert "surf::titled_entity::president_abraham_lincoln" in keys
        assert "surf::titled_entity::abraham_lincoln" in keys
        assert "surf::titled_entity::lincoln" in keys

    def test_ent_doc_contains_surface_forms(self, store):
        import json
        store.create_entity("ent-010", ["Alice", "alice"], {"entity_type": "PERSON"})
        writes = store.drain_pending_lmdb_writes()
        ent_writes = [(k, v) for k, v in writes if k.startswith("ent::")]
        assert len(ent_writes) == 1
        k, v = ent_writes[0]
        assert k == "ent::ent-010"
        doc = json.loads(v)
        assert doc["surface_forms"] == ["Alice", "alice"]
        assert doc["metadata"]["entity_type"] == "PERSON"
        assert "created_at" in doc

    def test_drain_clears(self, store):
        store.create_entity("ent-004", ["Carol"], {"entity_type": "PERSON"})
        writes1 = store.drain_pending_lmdb_writes()
        assert len(writes1) > 0
        writes2 = store.drain_pending_lmdb_writes()
        assert len(writes2) == 0

    def test_no_entity_type(self, store):
        store.create_entity("ent-005", ["Widget"], {})
        writes = store.drain_pending_lmdb_writes()
        keys = [k for k, v in writes]
        # ent:: doc + unscoped key for primary form
        assert "ent::ent-005" in keys
        assert "surf::Widget" in keys
        assert len(keys) == 2

    def test_batch_create(self, store):
        entities = [
            ("ent-006", ["Dave"], {"entity_type": "PERSON"}),
            ("ent-007", ["Eve"], {"entity_type": "PERSON"}),
        ]
        store.create_entities_batch(entities)
        writes = store.drain_pending_lmdb_writes()
        keys = [k for k, v in writes]
        assert "ent::ent-006" in keys
        assert "ent::ent-007" in keys
        assert "surf::PERSON::Dave" in keys
        assert "surf::PERSON::Eve" in keys


class TestGetEntity:
    def test_always_none(self, store):
        assert store.get_entity("ent-001") is None


class TestFuzzyResolution:
    """resolve_fuzzy and resolve_fuzzy_batch use surf:: lookups."""

    def test_resolve_fuzzy_exact_typed_hit(self, store):
        """Exact canonical match via typed surf:: key."""
        assert store.resolve_fuzzy("Alice", "PERSON", None) == "ent-001"

    def test_resolve_fuzzy_exact_untyped_hit(self, store):
        """Exact canonical match via untyped surf:: key."""
        assert store.resolve_fuzzy("Alice", "OTHER_TYPE", None) == "ent-001"

    def test_resolve_fuzzy_miss(self, store):
        assert store.resolve_fuzzy("unknown_person", "PERSON", None) is None

    def test_resolve_fuzzy_batch_exact_match(self, store):
        """Batch: exact canonical form found via surf:: lookup."""
        result = store.resolve_fuzzy_batch([("Alice", "PERSON", None)])
        assert result == {0: "ent-001"}

    def test_resolve_fuzzy_batch_empty(self):
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        s = LmdbEntityStore(FakeLmdbCache())
        assert s.resolve_fuzzy_batch([]) == {}

    def test_resolve_fuzzy_batch_miss(self, store):
        result = store.resolve_fuzzy_batch([("unknown", "PERSON", None)])
        assert result == {}

    def test_store_candidate_keys_noop(self, store):
        store.store_candidate_keys("ent-001", "Alice", ["k1"])

    def test_lookup_candidate_keys_empty(self, store):
        assert store.lookup_candidate_keys(["k1"]) == []

    def test_lookup_candidate_keys_by_key_empty(self, store):
        assert store.lookup_candidate_keys_by_key(["k1"]) == {}


class TestTitleStrippedResolution:
    """Title-stripped surface form resolution for titled entities."""

    @pytest.fixture
    def titled_store(self):
        """Store where Lincoln article created 'abraham_lincoln' entity."""
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        cache = FakeLmdbCache({
            # Lincoln article created entity 5d33753699af
            "surf::full_name::abraham_lincoln": b"5d33753699af",
            "surf::abraham_lincoln": b"5d33753699af",
            "surf::full_name::abraham": b"5d33753699af",
            "surf::full_name::lincoln": b"5d33753699af",
        })
        return LmdbEntityStore(cache)

    def test_titled_resolves_via_stripped_form(self, titled_store):
        """'president_abraham_lincoln' strips to 'abraham_lincoln' and finds existing entity."""
        result = titled_store.resolve_fuzzy(
            "president_abraham_lincoln", "titled_entity", None,
        )
        assert result == "5d33753699af"

    def test_titled_batch_resolves_via_stripped(self, titled_store):
        """Batch: titled form resolves via stripped surf:: key."""
        result = titled_store.resolve_fuzzy_batch([
            ("president_abraham_lincoln", "titled_entity", None),
        ])
        assert result == {0: "5d33753699af"}

    def test_untitled_resolves_exact(self, titled_store):
        """'abraham_lincoln' resolves directly via exact surf:: match."""
        result = titled_store.resolve_fuzzy(
            "abraham_lincoln", "full_name", None,
        )
        assert result == "5d33753699af"

    def test_cross_type_via_untyped_key(self, titled_store):
        """Different entity_type still matches via untyped surf:: key."""
        result = titled_store.resolve_fuzzy(
            "abraham_lincoln", "person", None,
        )
        # No surf::person::abraham_lincoln, but surf::abraham_lincoln exists
        assert result == "5d33753699af"

    def test_dr_stripped(self):
        """'dr_martin_luther_king' strips to 'martin_luther_king'."""
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        cache = FakeLmdbCache({
            "surf::full_name::martin_luther_king": b"aabbccdd",
            "surf::martin_luther_king": b"aabbccdd",
        })
        store = LmdbEntityStore(cache)
        result = store.resolve_fuzzy("dr_martin_luther_king", "titled_entity", None)
        assert result == "aabbccdd"

    def test_multi_title_stripped(self):
        """'vice_president_gore' strips both 'vice' and 'president'."""
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        cache = FakeLmdbCache({
            "surf::person::gore": b"goregoregore",
        })
        store = LmdbEntityStore(cache)
        result = store.resolve_fuzzy("vice_president_gore", "person", None)
        assert result == "goregoregore"

    def test_no_title_no_strip(self, titled_store):
        """'albert_einstein' has no title prefix — no stripping, miss if not in store."""
        result = titled_store.resolve_fuzzy("albert_einstein", "full_name", None)
        assert result is None

    def test_martin_luther_not_stripped(self):
        """'martin_luther' — 'martin' is NOT a title prefix, no stripping."""
        from nam_heads.impls.entity.default.storage.lmdb_store import _title_stripped
        assert _title_stripped("martin_luther") is None

    def test_martin_luther_king_not_stripped(self):
        """'martin_luther_king' — 'martin' is NOT a title, no stripping even though 'king' is."""
        from nam_heads.impls.entity.default.storage.lmdb_store import _title_stripped
        assert _title_stripped("martin_luther_king") is None

    def test_batch_mixed_titled_and_untitled(self, titled_store):
        """Batch with both titled and untitled candidates."""
        result = titled_store.resolve_fuzzy_batch([
            ("president_abraham_lincoln", "titled_entity", None),  # qi=0: title-stripped hit
            ("abraham_lincoln", "full_name", None),                # qi=1: exact hit
            ("unknown_person", "person", None),                    # qi=2: miss
        ])
        assert result[0] == "5d33753699af"
        assert result[1] == "5d33753699af"
        assert 2 not in result

    def test_reverse_order_titled_first(self):
        """Booth article processed first, Lincoln article resolves to existing entity.

        create_entity now stores untyped surf:: for multi-token surface forms,
        so 'abraham_lincoln' gets an untyped key even when it's not the primary form.
        """
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        # Booth article created entity 984f273b91fe with surface forms:
        # primary="president_abraham_lincoln", secondary="abraham_lincoln" etc.
        # create_entity stores untyped for primary + multi-token secondaries.
        cache = FakeLmdbCache({
            "surf::titled_entity::president_abraham_lincoln": b"984f273b91fe",
            "surf::president_abraham_lincoln": b"984f273b91fe",  # untyped primary
            "surf::titled_entity::abraham_lincoln": b"984f273b91fe",
            "surf::abraham_lincoln": b"984f273b91fe",  # untyped multi-token secondary
            "surf::titled_entity::lincoln": b"984f273b91fe",
        })
        store = LmdbEntityStore(cache)
        # Lincoln article processes "abraham_lincoln" with type "full_name"
        result = store.resolve_fuzzy("abraham_lincoln", "full_name", None)
        # surf::full_name::abraham_lincoln misses, but surf::abraham_lincoln hits
        assert result == "984f273b91fe"


class TestThreadSafety:
    def test_concurrent_creates(self, store):
        import threading

        def create_batch(prefix, count):
            for i in range(count):
                store.create_entity(
                    f"ent-{prefix}-{i}",
                    [f"name-{prefix}-{i}"],
                    {"entity_type": "PERSON"},
                )

        threads = [
            threading.Thread(target=create_batch, args=("a", 100)),
            threading.Thread(target=create_batch, args=("b", 100)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        writes = store.drain_pending_lmdb_writes()
        # Each entity creates 3 writes (ent:: doc + typed surf:: + unscoped primary surf::)
        assert len(writes) == 600


class TestRawDocuments:
    """get_raw_batch / put_raw_batch (the concept learner's cpt:: state)."""

    def test_get_reads_lmdb_and_omits_misses(self):
        from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore
        store = LmdbEntityStore(FakeLmdbCache({"cpt::dark_rum": b'{"learned":true}'}))
        assert store.get_raw_batch(["cpt::dark_rum", "cpt::other"]) == {"cpt::dark_rum": b'{"learned":true}'}
        assert store.get_raw_batch([]) == {}

    def test_put_is_queued_with_the_entity_writes(self, store):
        store.create_entity("ent-9", ["dark_rum"], {"entity_type": "compound_noun"})
        store.put_raw_batch([("cpt::dark_rum", b"{}")])
        writes = store.drain_pending_lmdb_writes()
        assert ("cpt::dark_rum", b"{}") in writes
        assert any(k.startswith("ent::") for k, _ in writes)
        assert store.drain_pending_lmdb_writes() == []
