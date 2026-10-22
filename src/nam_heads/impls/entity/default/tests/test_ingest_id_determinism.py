"""Entity ids are a pure function of the mention, whatever the store holds.

The rule: for every lookup key, the value stored is exactly what the shared
create logic produces on a miss -- so the id ingest emits is always
hash(canonical), regardless of pod, LMDB sync lag, or ingest vs query. A store
hit that names another id (an alias written by a different canonical) must not
change the ingest id; query keeps such a hit only as an alternate candidate.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ..head import EntityEncoder, _hash_entity_id, _ingest_entity_id
from ..storage.lmdb_store import LmdbEntityStore, _lookup_keys


class FakeLmdb:
    """Dict-backed stand-in for LmdbEntityCache (get / batch_get / available)."""

    available = True

    def __init__(self):
        self.d = {}

    def get(self, key):
        return self.d.get(key)

    def batch_get(self, keys):
        return {k: self.d[k] for k in keys if k in self.d}

    def sync(self, writes):
        """Relay drained writes into LMDB in order (last writer wins)."""
        for k, v in writes:
            self.d[k] = v


def _encoder(store):
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.2
    enc.enable_fuzzy = True
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = store
    return enc


LINCOLN = _hash_entity_id("", "abraham_lincoln")
PRESIDENT = _hash_entity_id("", "president_abraham_lincoln")
RECORD = "Abraham Lincoln was born in Kentucky."


def _ids(result):
    return {a.metadata.get("normalized"): a.concept for a in result.addresses}


def _encode(enc, text, path):
    if path == "batch":
        return enc.encode_batch_texts([text])[0]
    return enc._encode_internal(text, mode="ingest")


def _sync_president_aliases(lmdb, enc):
    """Another pod created "President Abraham Lincoln"; its aliases
    (including abraham_lincoln) reach this pod's LMDB."""
    forms = enc._surface_forms_with_aliases(
        "president_abraham_lincoln", entity_type="person")
    assert "abraham_lincoln" in forms
    writer = LmdbEntityStore(FakeLmdb())
    writer.create_entity(PRESIDENT, forms, {"entity_type": "person"})
    lmdb.sync(writer.drain_pending_lmdb_writes())


# -- helper ------------------------------------------------------------------

class TestIngestEntityId:
    def test_miss(self):
        assert _ingest_entity_id("person", "abraham_lincoln", None) == (LINCOLN, False)

    def test_matching_hit_exists(self):
        assert _ingest_entity_id("person", "abraham_lincoln", LINCOLN) == (LINCOLN, True)

    def test_foreign_hit_does_not_change_id(self):
        assert _ingest_entity_id("person", "abraham_lincoln", PRESIDENT) == (LINCOLN, False)

    def test_type_not_in_id(self):
        assert _ingest_entity_id("full_name", "x_y", None)[0] == \
            _ingest_entity_id("titled_entity", "x_y", None)[0]


# -- (a) the Lincoln scenario ------------------------------------------------

@pytest.mark.parametrize("path", ["batch", "single"])
def test_alias_from_titled_record_does_not_capture_id(path):
    lmdb = FakeLmdb()
    store = LmdbEntityStore(lmdb)
    enc = _encoder(store)

    before = _ids(_encode(enc, RECORD, path))
    assert before.get("abraham_lincoln") == LINCOLN

    _sync_president_aliases(lmdb, enc)
    assert lmdb.d["surf::abraham_lincoln"] == PRESIDENT.encode()

    # A fresh encoder (another pod / restarted process) sees the alias.
    after = _ids(_encode(_encoder(LmdbEntityStore(lmdb)), RECORD, path))
    assert after.get("abraham_lincoln") == LINCOLN
    assert PRESIDENT not in after.values()


@pytest.mark.parametrize("path", ["batch", "single"])
def test_foreign_hit_still_creates_own_entry(path):
    """A mismatching hit is a miss for bookkeeping: the canonical's own
    surf:: and ent:: entries are queued so the store converges."""
    lmdb = FakeLmdb()
    enc = _encoder(LmdbEntityStore(lmdb))
    _sync_president_aliases(lmdb, enc)
    _encode(enc, RECORD, path)
    writes = dict(enc.store.drain_pending_lmdb_writes())
    assert f"ent::{LINCOLN}" in writes
    assert writes.get("surf::person::abraham_lincoln") == LINCOLN.encode() or \
        writes.get("surf::abraham_lincoln") == LINCOLN.encode()


@pytest.mark.parametrize("path", ["batch", "single"])
def test_matching_hit_is_exists_not_create(path):
    """A hit equal to hash(canonical) counts as existing: no second create
    after the ent:: record is ensured once."""
    lmdb = FakeLmdb()
    enc = _encoder(LmdbEntityStore(lmdb))
    _encode(enc, RECORD, path)
    lmdb.sync(enc.store.drain_pending_lmdb_writes())
    enc2 = _encoder(LmdbEntityStore(lmdb))
    _encode(enc2, RECORD, path)
    first = enc2.store.drain_pending_lmdb_writes()
    if path == "single":
        assert first == []  # every mention hit its own id
    else:
        # batch ensures the ent:: record once per process lifetime
        assert all(k.startswith(("ent::", "surf::")) for k, _ in first)
    _encode(enc2, RECORD, path)
    assert enc2.store.drain_pending_lmdb_writes() == []


# -- (b) empty vs populated store -------------------------------------------

TEXTS = [
    "President Abraham Lincoln. President Abraham Lincoln led the Union.",
    "Abraham Lincoln was born in Kentucky.",
    "The Beatles. The Beatles were a band from Liverpool.",
    "John Lennon of the Beatles wrote songs with Paul McCartney.",
    "Queen Elizabeth II visited the United States in 1957.",
    "Elizabeth II was the Queen of the United Kingdom.",
]


@pytest.mark.parametrize("path", ["batch", "single"])
def test_empty_vs_populated_store_identical(path):
    empty = FakeLmdb()
    enc_a = _encoder(LmdbEntityStore(empty))
    ids_a = [[a.concept for a in _encode(enc_a, t, path).addresses] for t in TEXTS]

    populated = FakeLmdb()
    populated.sync(enc_a.store.drain_pending_lmdb_writes())
    assert populated.d, "first pass should have queued writes"
    enc_b = _encoder(LmdbEntityStore(populated))
    ids_b = [[a.concept for a in _encode(enc_b, t, path).addresses] for t in TEXTS]
    assert ids_a == ids_b


def test_batch_and_single_paths_agree():
    lmdb = FakeLmdb()
    enc = _encoder(LmdbEntityStore(lmdb))
    _sync_president_aliases(lmdb, enc)
    b = _ids(enc.encode_batch_texts([RECORD])[0])
    s = _ids(enc._encode_internal(RECORD, mode="ingest"))
    assert b["abraham_lincoln"] == s["abraham_lincoln"] == LINCOLN


# -- (c) deterministic tie-break ---------------------------------------------

def test_lookup_key_precedence():
    assert _lookup_keys("president_abraham_lincoln", "person") == [
        "surf::person::president_abraham_lincoln",
        "surf::president_abraham_lincoln",
        "surf::person::abraham_lincoln",
        "surf::abraham_lincoln",
    ]
    assert _lookup_keys("lincoln", None) == ["surf::lincoln"]


def test_resolve_fuzzy_batch_precedence_in_process():
    lmdb = FakeLmdb()
    lmdb.d = {
        "surf::president_abraham_lincoln": b"untyped_exact",
        "surf::person::abraham_lincoln": b"typed_strip",
        "surf::abraham_lincoln": b"untyped_strip",
        "surf::person::kentucky": b"typed_k",
        "surf::kentucky": b"untyped_k",
    }
    store = LmdbEntityStore(lmdb)
    qs = [("president_abraham_lincoln", "person", None),
          ("abraham_lincoln", "titled_entity", None),
          ("kentucky", "person", None),
          ("nothing_here", "person", None)]
    res = store.resolve_fuzzy_batch(qs)
    assert res == {0: "untyped_exact", 1: "untyped_strip", 2: "typed_k"}
    # Single-query path agrees with the batch path.
    for qi, (c, t, _) in enumerate(qs):
        assert store.resolve_fuzzy(c, t, None) == res.get(qi)


_TIE_SCRIPT = textwrap.dedent("""
    import json
    from nam_heads.impls.entity.default.storage.lmdb_store import LmdbEntityStore

    class L:
        available = True
        def __init__(self, d): self.d = d
        def get(self, k): return self.d.get(k)
        def batch_get(self, keys): return {k: self.d[k] for k in keys if k in self.d}

    d = {}
    qs = []
    for i in range(40):
        name = f"president_name_{i}"
        # conflicting hits on every key a query looks up
        d[f"surf::person::{name}"] = f"typed_exact_{i}".encode()
        d[f"surf::{name}"] = f"untyped_exact_{i}".encode()
        d[f"surf::person::name_{i}"] = f"typed_strip_{i}".encode()
        d[f"surf::name_{i}"] = f"untyped_strip_{i}".encode()
        qs.append((name, "person", None))
        # untyped vs stripped-typed only
        qs.append((f"name_{i}", "titled_entity", None))
    res = LmdbEntityStore(L(d)).resolve_fuzzy_batch(qs)
    print(json.dumps(sorted(res.items())))
""")


def test_tie_break_independent_of_hash_seed():
    root = Path(__file__).resolve().parents[5]  # .../nam
    outs = set()
    for seed in range(7):
        env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=str(root))
        out = subprocess.run([sys.executable, "-c", _TIE_SCRIPT], env=env,
                             capture_output=True, text=True, check=True).stdout
        outs.add(out)
    assert len(outs) == 1, outs
    import json
    res = dict(json.loads(outs.pop()))
    assert res[0] == "typed_exact_0"          # typed exact beats everything
    assert res[1] == "untyped_strip_0"        # name_0 untyped exact
    assert all(v.startswith(("typed_exact_", "untyped_strip_")) for v in res.values())


# -- (d) query mode includes hash(canonical) --------------------------------

def _query_store():
    lmdb = FakeLmdb()
    # surf:: for the mention points at another canonical's id
    lmdb.d["surf::abraham_lincoln"] = PRESIDENT.encode()
    lmdb.d["surf::person::abraham_lincoln"] = PRESIDENT.encode()
    return LmdbEntityStore(lmdb)


def test_query_mode_includes_hash_of_canonical():
    enc = _encoder(_query_store())
    res = enc.encode_query("Abraham Lincoln")
    by_norm = {a.metadata.get("normalized"): a for a in res.addresses}
    addr = by_norm["abraham_lincoln"]
    assert addr.concept == LINCOLN
    assert addr.metadata.get("alt_entity_ids") == [PRESIDENT]
    assert not addr.metadata.get("unresolved")


def test_query_mode_matching_hit_has_no_alternate():
    lmdb = FakeLmdb()
    lmdb.d["surf::abraham_lincoln"] = LINCOLN.encode()
    enc = _encoder(LmdbEntityStore(lmdb))
    addr = {a.metadata.get("normalized"): a
            for a in enc.encode_query("Abraham Lincoln").addresses}["abraham_lincoln"]
    assert addr.concept == LINCOLN
    assert "alt_entity_ids" not in addr.metadata


def test_query_mode_miss_unchanged():
    enc = _encoder(LmdbEntityStore(FakeLmdb()))
    addr = {a.metadata.get("normalized"): a
            for a in enc.encode_query("Abraham Lincoln").addresses}["abraham_lincoln"]
    assert addr.concept == LINCOLN
    assert addr.metadata.get("unresolved") is True


def test_resolve_query_entities_primary_and_alternate():
    enc = _encoder(_query_store())
    items = enc.resolve_query_entities("Who was Abraham Lincoln?")
    lincoln = [it for it in items if it["normalized"] == "abraham_lincoln"]
    assert lincoln, items
    assert lincoln[0]["entity_id"] == LINCOLN
    assert lincoln[0]["alt_entity_ids"] == [PRESIDENT]
