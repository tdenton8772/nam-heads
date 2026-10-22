"""
Tests for the CouchbaseKV entity store using a mock KV client.
Validates the store implements the EntityStore ABC correctly
without needing a real Couchbase instance.
"""
import json
import pytest


class MockKV:
    """In-memory mock of CouchbaseKV with get/set/batch API."""

    def __init__(self):
        self._data = {}  # key -> (value_bytes, cas)
        self._cas_counter = 0

    def get(self, key):
        if key in self._data:
            return self._data[key]
        return None, None

    def set(self, key, value, cas=0, expiration=0):
        self._cas_counter += 1
        self._data[key] = (value, self._cas_counter)
        return self._cas_counter

    def pipeline_get(self, keys):
        return {k: self.get(k) for k in keys}

    def pipeline_set(self, items):
        return [(k, self.set(k, v)) for k, v in items]


@pytest.fixture
def store():
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    return CouchbaseKVEntityStore(MockKV())


def test_resolve_surface_form_empty(store):
    assert store.resolve_surface_form("nonexistent") is None


def test_create_and_resolve(store):
    store.create_entity(
        entity_id="person:einstein",
        surface_forms=["einstein", "albert_einstein"],
        metadata={"entity_type": "person"},
    )
    # Type-scoped resolution
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:einstein"
    assert store.resolve_surface_form("albert_einstein", entity_type="person") == "person:einstein"
    # Unscoped does NOT resolve typed entities (no unscoped keys written)
    assert store.resolve_surface_form("einstein") is None
    assert store.resolve_surface_form("albert_einstein") is None
    assert store.resolve_surface_form("feynman") is None


def test_type_scoped_isolation(store):
    """Different entity types with shared surface forms stay isolated."""
    store.create_entity(
        entity_id="person:albert_einstein",
        surface_forms=["albert_einstein", "einstein"],
        metadata={"entity_type": "person"},
    )
    store.create_entity(
        entity_id="concept:bose_einstein_condensation",
        surface_forms=["bose_einstein_condensation", "einstein_condensation"],
        metadata={"entity_type": "concept"},
    )
    # Type-scoped: each type resolves to its own entity
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:albert_einstein"
    assert store.resolve_surface_form("einstein_condensation", entity_type="concept") == "concept:bose_einstein_condensation"
    # Person lookup for "einstein_condensation" returns None (no cross-type pollution)
    assert store.resolve_surface_form("einstein_condensation", entity_type="person") is None


def test_first_writer_wins_within_type(store):
    """First entity registered with a surface form wins within the same type."""
    store.create_entity(
        entity_id="person:albert_einstein",
        surface_forms=["albert_einstein", "einstein"],
        metadata={"entity_type": "person"},
    )
    store.create_entity(
        entity_id="person:hans_albert_einstein",
        surface_forms=["hans_albert_einstein", "einstein"],
        metadata={"entity_type": "person"},
    )
    # "einstein" still resolves to the first-registered person entity
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:albert_einstein"
    # Full forms resolve correctly
    assert store.resolve_surface_form("albert_einstein", entity_type="person") == "person:albert_einstein"
    assert store.resolve_surface_form("hans_albert_einstein", entity_type="person") == "person:hans_albert_einstein"


def test_get_entity(store):
    store.create_entity(
        entity_id="object:bike",
        surface_forms=["bike"],
        metadata={"entity_type": "object", "source": "test"},
    )
    entity = store.get_entity("object:bike")
    assert entity is not None
    assert entity["surface_forms"] == ["bike"]
    assert entity["metadata"]["entity_type"] == "object"


def test_get_entity_missing(store):
    assert store.get_entity("nonexistent") is None


def test_add_surface_form(store):
    store.create_entity(
        entity_id="person:fleming",
        surface_forms=["fleming"],
        metadata={"entity_type": "person"},
    )
    store.add_surface_form("person:fleming", "alexander_fleming", max_forms=10)
    # Added surface form resolves via type-scoped lookup
    assert store.resolve_surface_form("alexander_fleming", entity_type="person") == "person:fleming"
    entity = store.get_entity("person:fleming")
    assert "alexander_fleming" in entity["surface_forms"]


def test_add_surface_form_max_cap(store):
    store.create_entity(
        entity_id="object:x",
        surface_forms=["a", "b"],
        metadata={},
    )
    store.add_surface_form("object:x", "c", max_forms=2)
    entity = store.get_entity("object:x")
    # First form "a" should have been popped
    assert len(entity["surface_forms"]) <= 2
    assert "c" in entity["surface_forms"]


def test_candidate_keys_store_and_lookup(store):
    store.store_candidate_keys("person:einstein", "einstein", ["ph:ANSTN", "sorted:einstein"])
    store.store_candidate_keys("person:feynman", "feynman", ["ph:FNMN", "sorted:feynman"])

    results = store.lookup_candidate_keys(["ph:ANSTN"])
    assert len(results) == 1
    assert results[0] == ("person:einstein", "einstein")

    results = store.lookup_candidate_keys(["ph:FNMN"])
    assert len(results) == 1
    assert results[0] == ("person:feynman", "feynman")

    # Lookup with no matching keys
    results = store.lookup_candidate_keys(["ph:NONEXIST"])
    assert len(results) == 0


def test_candidate_keys_append(store):
    """Multiple entities sharing the same candidate key."""
    store.store_candidate_keys("person:einstein", "einstein", ["ph:ANSTN"])
    store.store_candidate_keys("person:einstein_boris", "einstein_boris", ["ph:ANSTN"])

    results = store.lookup_candidate_keys(["ph:ANSTN"])
    entity_ids = [r[0] for r in results]
    assert "person:einstein" in entity_ids
    assert "person:einstein_boris" in entity_ids


def test_fuzzy_resolve(store):
    """Test the base class resolve_fuzzy via stored candidate keys."""
    from nam_heads.impls.entity.default.comparators import get_comparator

    # Register entity with candidate keys
    store.create_entity(
        entity_id="person:einstein",
        surface_forms=["einstein", "albert einstein"],
        metadata={"entity_type": "person"},
    )
    comparator = get_comparator("person")
    for sf in ["einstein", "albert einstein"]:
        keys = comparator.generate_keys(sf)
        if keys:
            store.store_candidate_keys("person:einstein", sf, keys)

    # Exact surface match first (type-scoped)
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:einstein"

    # Fuzzy resolve (should find via candidate keys)
    result = store.resolve_fuzzy("einstein", "person", comparator)
    assert result == "person:einstein"


def test_no_entity_type_in_metadata(store):
    """Entities without entity_type in metadata still work via unscoped keys."""
    store.create_entity(
        entity_id="thing:widget",
        surface_forms=["widget"],
        metadata={"source": "test"},
    )
    # Resolves via unscoped (primary form)
    assert store.resolve_surface_form("widget") == "thing:widget"


def test_einstein_disambiguation(store):
    """Bose-Einstein and Albert Einstein must be separate person entities.

    Simulates real ingest: Bose-Einstein article processed first, creates
    person:bose_einstein with component surface forms. Albert Einstein
    article processed second. Both share 'einstein' as a component but
    must remain distinct entities.
    """
    from nam_heads.impls.entity.default.comparators import get_comparator

    # --- Bose-Einstein article processed first ---
    store.create_entity(
        entity_id="person:bose_einstein",
        surface_forms=["bose_einstein", "bose", "einstein"],
        metadata={"entity_type": "person"},
    )
    comparator = get_comparator("person")
    for sf in ["bose_einstein", "bose", "einstein"]:
        keys = comparator.generate_keys(sf)
        if keys:
            store.store_candidate_keys("person:bose_einstein", sf, keys)

    # --- Albert Einstein article processed second ---
    store.create_entity(
        entity_id="person:albert_einstein",
        surface_forms=["albert_einstein", "albert", "einstein"],
        metadata={"entity_type": "person"},
    )
    for sf in ["albert_einstein", "albert", "einstein"]:
        keys = comparator.generate_keys(sf)
        if keys:
            store.store_candidate_keys("person:albert_einstein", sf, keys)

    # Both entities exist
    assert store.get_entity("person:bose_einstein") is not None
    assert store.get_entity("person:albert_einstein") is not None

    # Full forms resolve to their own entity (type-scoped)
    assert store.resolve_surface_form("bose_einstein", entity_type="person") == "person:bose_einstein"
    assert store.resolve_surface_form("albert_einstein", entity_type="person") == "person:albert_einstein"

    # "einstein" component: first-writer-wins within person type
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:bose_einstein"

    # Unscoped returns None for typed entities (no unscoped keys written)
    assert store.resolve_surface_form("bose_einstein") is None
    assert store.resolve_surface_form("albert_einstein") is None
    assert store.resolve_surface_form("einstein") is None

    # Cross-type isolation: object entity doesn't pollute person resolution
    store.create_entity(
        entity_id="object:albert_einstein",
        surface_forms=["albert_einstein"],
        metadata={"entity_type": "object"},
    )
    # Person resolution still returns the person entity
    assert store.resolve_surface_form("albert_einstein", entity_type="person") == "person:albert_einstein"
    # Object resolution returns the object entity
    assert store.resolve_surface_form("albert_einstein", entity_type="object") == "object:albert_einstein"

    # Fuzzy: "albert_einstein" must NOT resolve to "bose_einstein"
    fuzzy_result = store.resolve_fuzzy("albert_einstein", "person", comparator)
    assert fuzzy_result != "person:bose_einstein", \
        "albert_einstein must not fuzzy-match to bose_einstein"


def test_hans_albert_einstein_separate(store):
    """Hans Albert Einstein (son) must not merge with Albert Einstein (father)."""
    from nam_heads.impls.entity.default.comparators import get_comparator

    store.create_entity(
        entity_id="person:albert_einstein",
        surface_forms=["albert_einstein", "albert", "einstein"],
        metadata={"entity_type": "person"},
    )
    comparator = get_comparator("person")
    for sf in ["albert_einstein", "albert", "einstein"]:
        keys = comparator.generate_keys(sf)
        if keys:
            store.store_candidate_keys("person:albert_einstein", sf, keys)

    # Hans Albert Einstein should NOT fuzzy-match to Albert Einstein
    fuzzy_result = store.resolve_fuzzy("hans_albert_einstein", "person", comparator)
    assert fuzzy_result is None or fuzzy_result != "person:albert_einstein", \
        "hans_albert_einstein must not fuzzy-match to albert_einstein (different person)"


def test_created_at_in_entity(store):
    """Entity docs include created_at timestamp."""
    store.create_entity(
        entity_id="person:test",
        surface_forms=["test"],
        metadata={"entity_type": "person"},
    )
    entity = store.get_entity("person:test")
    assert entity is not None
    assert "created_at" in entity
    assert isinstance(entity["created_at"], float)
    assert entity["created_at"] > 0


def test_created_at_in_batch(store):
    """Batch-created entities include created_at timestamp."""
    store.create_entities_batch([
        ("person:a", ["alice"], {"entity_type": "person"}),
        ("person:b", ["bob"], {"entity_type": "person"}),
    ])
    for eid in ("person:a", "person:b"):
        entity = store.get_entity(eid)
        assert entity is not None
        assert "created_at" in entity
        assert isinstance(entity["created_at"], float)


def test_no_unscoped_keys_written_for_typed_entities(store):
    """Typed entities must NOT write unscoped surf:: keys."""
    kv = store._kv
    store.create_entity(
        entity_id="person:tesla",
        surface_forms=["tesla", "nikola_tesla"],
        metadata={"entity_type": "person"},
    )
    # Scoped keys exist
    assert kv._data.get("surf::person::tesla") is not None
    assert kv._data.get("surf::person::nikola_tesla") is not None
    # Unscoped keys must NOT exist
    assert "surf::tesla" not in kv._data
    assert "surf::nikola_tesla" not in kv._data


def test_no_unscoped_keys_written_batch(store):
    """Batch create with typed entities must NOT write unscoped surf:: keys."""
    kv = store._kv
    store.create_entities_batch([
        ("person:curie", ["curie", "marie_curie"], {"entity_type": "person"}),
        ("place:paris", ["paris"], {"entity_type": "place"}),
    ])
    # Scoped keys exist
    assert kv._data.get("surf::person::curie") is not None
    assert kv._data.get("surf::person::marie_curie") is not None
    assert kv._data.get("surf::place::paris") is not None
    # Unscoped keys must NOT exist
    assert "surf::curie" not in kv._data
    assert "surf::marie_curie" not in kv._data
    assert "surf::paris" not in kv._data


def test_untyped_entity_uses_unscoped_keys(store):
    """Entities without entity_type still use unscoped surf:: keys."""
    kv = store._kv
    store.create_entity(
        entity_id="thing:gadget",
        surface_forms=["gadget"],
        metadata={"source": "test"},
    )
    assert kv._data.get("surf::gadget") is not None
    assert store.resolve_surface_form("gadget") == "thing:gadget"


def test_batch_resolve_scoped_only(store):
    """Batch resolve returns None for scoped miss (no unscoped fallback)."""
    store.create_entity(
        entity_id="person:darwin",
        surface_forms=["darwin", "charles_darwin"],
        metadata={"entity_type": "person"},
    )
    results = store.resolve_surface_forms_batch([
        ("darwin", "person"),          # hit
        ("charles_darwin", "person"),  # hit
        ("darwin", "concept"),         # miss - wrong type
        ("darwin", None),              # miss - no unscoped key
    ])
    assert results[("darwin", "person")] == "person:darwin"
    assert results[("charles_darwin", "person")] == "person:darwin"
    assert results[("darwin", "concept")] is None
    assert results[("darwin", None)] is None


def test_resolve_fuzzy_batch(store):
    """Batch fuzzy resolution finds multiple entities in one call."""
    from nam_heads.impls.entity.default.comparators import get_comparator

    store.create_entity(
        entity_id="person:einstein",
        surface_forms=["einstein", "albert_einstein"],
        metadata={"entity_type": "person"},
    )
    store.create_entity(
        entity_id="person:newton",
        surface_forms=["newton", "isaac_newton"],
        metadata={"entity_type": "person"},
    )

    comparator = get_comparator("person")
    for eid, sforms in [("person:einstein", ["einstein", "albert_einstein"]),
                         ("person:newton", ["newton", "isaac_newton"])]:
        for sf in sforms:
            keys = comparator.generate_keys(sf)
            if keys:
                store.store_candidate_keys(eid, sf, keys)

    results = store.resolve_fuzzy_batch([
        ("einstein", "person", comparator),
        ("newton", "person", comparator),
        ("nonexistent_xyz", "person", comparator),
    ])
    assert results[0] == "person:einstein"
    assert results[1] == "person:newton"
    assert results[2] is None


# ---- kv_on_cache_miss tests ----

def test_kv_on_miss_default_true():
    """Default kv_on_cache_miss is True."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    store = CouchbaseKVEntityStore(MockKV())
    assert store._kv_on_miss is True


def test_kv_on_miss_false_constructor():
    """Constructor accepts kv_on_cache_miss=False."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    store = CouchbaseKVEntityStore(MockKV(), kv_on_cache_miss=False)
    assert store._kv_on_miss is False


def test_resolve_surface_form_skips_kv_when_kv_on_miss_false():
    """resolve_surface_form() returns None for uncached keys when kv_on_cache_miss=False."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    # Pre-populate KV with a surf:: key
    kv.set("surf::person::einstein", b"person:einstein")

    store = CouchbaseKVEntityStore(kv, kv_on_cache_miss=False)
    # Key exists in KV but store should not look it up
    assert store.resolve_surface_form("einstein", entity_type="person") is None


def test_resolve_surface_form_returns_cached_when_kv_on_miss_false():
    """resolve_surface_form() returns L1-cached values when kv_on_cache_miss=False."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    store = CouchbaseKVEntityStore(kv, kv_on_cache_miss=False)

    # Populate L1 cache via create_entity (which updates _surf_cache)
    store.create_entity(
        entity_id="person:einstein",
        surface_forms=["einstein"],
        metadata={"entity_type": "person"},
    )
    # L1 cache was populated by create_entity
    assert store.resolve_surface_form("einstein", entity_type="person") == "person:einstein"


def test_resolve_batch_skips_kv_when_kv_on_miss_false():
    """resolve_surface_forms_batch() skips L3 KV when kv_on_cache_miss=False."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    kv.set("surf::person::newton", b"person:newton")

    store = CouchbaseKVEntityStore(kv, kv_on_cache_miss=False)
    results = store.resolve_surface_forms_batch([("newton", "person")])
    # Key exists in KV but should not be fetched
    assert results[("newton", "person")] is None


def test_resolve_batch_returns_cached_when_kv_on_miss_false():
    """resolve_surface_forms_batch() returns L1-cached values when kv_on_cache_miss=False."""
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    store = CouchbaseKVEntityStore(kv, kv_on_cache_miss=False)

    # Populate L1 cache via create_entity
    store.create_entity(
        entity_id="person:curie",
        surface_forms=["curie"],
        metadata={"entity_type": "person"},
    )
    results = store.resolve_surface_forms_batch([("curie", "person")])
    assert results[("curie", "person")] == "person:curie"


class _FakeL2:
    def __init__(self, d):
        self.d = d

    def batch_get(self, keys):
        return {k: self.d[k] for k in keys if k in self.d}


def test_raw_get_reads_lmdb_then_kv():
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    kv.set("cpt::b", b"from-kv")
    kv.set("cpt::a", b"stale-kv")
    s = CouchbaseKVEntityStore(kv, lmdb_cache=_FakeL2({"cpt::a": b"from-lmdb"}))
    assert s.get_raw_batch(["cpt::a", "cpt::b", "cpt::c"]) == {"cpt::a": b"from-lmdb", "cpt::b": b"from-kv"}


def test_raw_get_without_read_through_stays_on_lmdb():
    from nam_heads.impls.entity.default.storage.couchbase_kv_store import CouchbaseKVEntityStore
    kv = MockKV()
    kv.set("cpt::b", b"from-kv")
    s = CouchbaseKVEntityStore(kv, lmdb_cache=_FakeL2({}), kv_on_cache_miss=False)
    assert s.get_raw_batch(["cpt::b"]) == {}


def test_raw_put_writes_kv_and_queues_lmdb_sync(store):
    store.put_raw_batch([("cpt::dark_rum", b'{"learned":false}')])
    assert store._kv.get("cpt::dark_rum")[0] == b'{"learned":false}'
    assert ("cpt::dark_rum", b'{"learned":false}') in store.drain_pending_lmdb_writes()
    store.put_raw_batch([])
    assert store.drain_pending_lmdb_writes() == []
