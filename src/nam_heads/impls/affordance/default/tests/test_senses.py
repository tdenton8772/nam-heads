"""Verb senses (senses.py): the table, the axis encoder, and the head under NAM_AFFORDANCE_SENSES / _PARK."""
import json

import pytest

from src.encoders import axis_encoder as AE
from nam_heads.impls.affordance.default import senses as S

SEED = {
    "version": "test",
    "senses": {"1": ["repair", 3], "2": ["run", 4], "3": ["run", 7], "4": ["describe", 9], "5": ["fix", 3]},
    "word": {"repair": 1, "describe": 4, "fix": 5},
    "frame": {"run|for": 3},
    "dominant": {"run": 2},
    "not_verb": ["cod"],
    "cells": {"3": {"family": 0, "members": ["repair", "fix"], "neighbours": [4, 7, 9]},
              "4": {"family": 1, "members": ["run"], "neighbours": [7]},
              "7": {"family": 1, "members": ["run"], "neighbours": [4]},
              "9": {"family": 2, "members": ["describe"], "neighbours": []}},
    "counts": {}, "examples": {}, "wordnet": {}, "next_cell": 10,
}


@pytest.fixture
def seed_env(tmp_path, monkeypatch):
    p = tmp_path / "sense_seed.json"
    p.write_text(json.dumps(SEED))
    monkeypatch.setenv("NAM_AFFORDANCE_SENSE_SEED", str(p))
    monkeypatch.setenv("NAM_AFFORDANCE_SENSES", "1")
    monkeypatch.setattr(S, "_TABLE", None)
    monkeypatch.setattr(AE, "_BUILTIN", {})
    return p


def test_resolve_order_frame_word_dominant_not_verb_learned_unknown(seed_env):
    t = S.table()
    assert t.resolve("run", "for") == (S.CELL, 7)            # the frame decides
    assert t.resolve("run", "_") == (S.CELL, 4)              # else the dominant sense
    assert t.resolve("repair") == (S.CELL, 3)
    assert t.resolve("cod") == (S.SKIP, None)                # a teacher non-verb
    assert t.resolve("livestream") == (S.UNKNOWN, None)
    assert t.resolve("livestream", learned={"kind": "new", "cell": 12}) == (S.CELL, 12)
    assert t.resolve("blarg", learned={"kind": "not_verb"}) == (S.SKIP, None)
    assert t.resolve("blarg", learned={"kind": "leave_out"}) == (S.SKIP, None)


def test_values_codes_and_neighbours(seed_env):
    enc = AE.builtin_axis_encoder("affordance")
    assert isinstance(enc, S.SenseTableAxisEncoder) and enc.artifact_id.startswith("sense-table-")
    codes = enc.encode_values(["repair@3", "fix@3", "__null__", "describe", "zzz"])
    assert codes[0] == f"lca:3:{S.fine_of('repair')}" and codes[0] != codes[1] and codes[1].startswith("lca:3:")
    assert codes[2] == "lca:0:0" and codes[3].startswith("lca:9:")
    assert int(codes[4].split(":")[1]) >= S.UNKNOWN_CELL_BASE      # never a seed cell, never NULL
    assert enc.nearest_coarse_codes(3, k=2) == [4, 7] and enc.nearest_coarse_codes(99, k=2) == []
    assert S.parse_value("a@b") is None and S.parse_value("run@4") == ("run", 4)
    assert enc.encode_values(["repair@3"]) == enc.encode_values(["repair@3"])     # deterministic


def test_the_charcnn_stays_without_the_switch(monkeypatch):
    monkeypatch.setenv("NAM_AFFORDANCE_SENSES", "0")
    monkeypatch.setattr(AE, "_BUILTIN", {})
    assert isinstance(AE.builtin_axis_encoder("affordance"), AE.CharCNNAxisEncoder)


def test_frame_after_reads_the_next_token():
    class T:
        def __init__(self, text):
            self.text = text
    doc = [T("She"), T("ran"), T("for"), T("office")]
    assert S.frame_after(doc, 1) == "for" and S.frame_after(doc, 3) == "_" and S.frame_after(doc, 0) == "_"


# ---- the head ----------------------------------------------------------------------------------------

def _head(monkeypatch, park=True):
    from nam_heads.impls.affordance.default.head import AffordanceEncoder
    h = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    assert h.parking is False                      # the host grants parking (slots.create_head), never the head
    h.parking = park
    return h


def test_head_emits_word_at_cell_values(seed_env, monkeypatch):
    h = _head(monkeypatch)
    addrs = h.encode("The technician repaired the pump after the failure.").addresses
    assert [a.concept for a in addrs] == ["repair@3"]


def test_head_uses_the_frame_and_the_dominant_sense(seed_env, monkeypatch):
    h = _head(monkeypatch)
    assert [a.concept for a in h.encode("She ran for office in 1990.").addresses] == ["run@7"]
    assert [a.concept for a in h.encode("She ran the company for years.").addresses] == ["run@4"]


def test_unknown_verb_parks_the_record_at_ingest(seed_env, monkeypatch):
    h = _head(monkeypatch)
    r = h.encode("The band livestreamed the concert to millions of fans.")
    assert r.addresses == []
    parked = r.metadata["parked"]
    assert [p["value"] for p in parked] and parked[0]["token"].lower().startswith("livestream")
    assert "concert" in parked[0]["sentence"]


def test_unknown_verb_is_left_out_at_query_and_without_parking(seed_env, monkeypatch):
    h = _head(monkeypatch)
    q = h.encode_query("which band livestreamed the concert")
    assert "parked" not in q.metadata and all("livestream" not in a.concept for a in q.addresses)
    h2 = _head(monkeypatch, park=False)
    r = h2.encode("The band livestreamed the concert to millions of fans.")
    assert "parked" not in r.metadata


def test_a_learned_sense_is_used_and_cached_positively_only(seed_env, monkeypatch):
    h = _head(monkeypatch)
    calls = []

    class FakeKV:
        docs = {}

        def pipeline_get(self, keys):
            calls.append(list(keys))
            return {k: (json.dumps(self.docs[k]).encode(), 1) for k in keys if k in self.docs}
    store = h.category_store
    monkeypatch.setattr(store, "_kv", FakeKV(), raising=False)
    if not hasattr(store, "_key"):
        monkeypatch.setattr(store, "_key", lambda name: f"vcat::{name}", raising=False)
    text = "The band livestreamed the concert to millions of fans."
    r = h.encode(text)
    assert "parked" in r.metadata                                  # not learned yet: parks
    value = r.metadata["parked"][0]["value"]                       # the base as the head sees it
    FakeKV.docs[store._key("sense:" + value)] = {"kind": "new", "cell": 12}    # the session decides it
    assert [a.concept for a in h.encode(text).addresses] == [f"{value}@12"]
    n = len(calls)
    h.encode(text)
    assert len(calls) == n                                         # cached once known


def test_a_junk_repaired_base_falls_back_to_a_base_the_table_knows(seed_env, monkeypatch):
    """A base the lexicon learned from noise ("repaired" -> "repaire") must not park the record when the
    token has a candidate the sense table knows."""
    h = _head(monkeypatch)
    from nam_heads.impls.affordance.default import senses as S2
    tbl = S2.table()
    assert h._known_base("repaired", "repaire", tbl) == "repair"
    assert h._known_base("livestreamed", "livestreame", tbl) is None
