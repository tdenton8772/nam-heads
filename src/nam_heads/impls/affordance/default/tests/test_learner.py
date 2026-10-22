"""The affordance learner part: candidates, marking, items, answer -> learned entry, and the part as a
nam_heads learner (the SDK's conformance check)."""
import json

import pytest

from nam_heads.sdk import testing as T
from nam_heads.sdk.store import MemoryStore, StoreSpec

from nam_heads.impls.affordance.default import learner as A
from nam_heads.impls.affordance.default.senses import SenseTable

SEED = {
    "senses": {"1": ["stream", 3], "2": ["broadcast", 3], "3": ["air", 4], "4": ["brand", 5], "5": ["repair", 6]},
    "word": {"stream": 1, "broadcast": 2, "air": 3, "brand": 4, "repair": 5},
    "frame": {}, "dominant": {}, "not_verb": [],
    "cells": {"3": {"members": ["broadcast", "stream", "air"], "neighbours": [4]},
              "4": {"members": ["air"], "neighbours": [3]}, "5": {"members": ["brand"], "neighbours": []},
              "6": {"members": ["repair"], "neighbours": []}},
    "counts": {"stream": 50, "broadcast": 40, "air": 10, "brand": 9, "repair": 30},
    "examples": {"stream": "...the match was ⟦streamed⟧ live...", "broadcast": "", "air": "", "brand": ""},
    "wordnet": {"telecast": [["broadcast", "synset"], ["air", "synset"], ["zzz", "sister"]]},
    "next_cell": 7,
}


@pytest.fixture
def table():
    return SenseTable(SEED)


def test_candidates_from_wordnet_only_known_words(table):
    cs = A.candidates("telecast", table)
    assert {c["concept"] for c in cs} == {"broadcast", "air"} and all(c["kind"] == "synset" for c in cs)
    b = [c for c in cs if c["concept"] == "broadcast"][0]
    assert b["members"][0] in ("broadcast", "stream") and "telecast" not in b["members"]


def test_candidates_for_an_unknown_word_use_stems_and_spellings(table):
    assert A.candidates("livestream", table) == []                 # no stem, no close spelling (as in training)
    cs = A.candidates("restream", table)
    assert "stream" in {c["concept"] for c in cs} and cs[0]["kind"] in ("stem", "spelling")
    cs2 = A.candidates("rebrand", table)
    assert "brand" in {c["concept"] for c in cs2}
    assert len(A.candidates("qqqq", table)) == 0


def test_mark_picks_the_occurrence_matching_the_sentence():
    text = "They livestreamed games. Later the band livestreamed the concert to fans."
    m = A.mark(text, "livestreamed", "Later the band livestreamed the concert to fans .")
    assert "band ⟦livestreamed⟧ the concert" in m and m.count("⟦") == 1
    assert A.mark("nothing here", "x") is None


def test_build_item_falls_back_to_the_head_sentence(table):
    it = A.build_item("1|livestreame", "livestreame", {"token": "livestreamed", "sentence": "The band livestreamed it ."},
                      {"title": "T", "text": "text without the word"}, table)
    assert it["record"]["text"] == "The band ⟦livestreamed⟧ it ." and it["word"] == "livestreame"
    assert A.build_item("1|x", "x", {"token": "zz"}, {"text": "no"}, table) is None


def test_entries_from_answers(table):
    cands = A.candidates("restream", table)
    names = [c["concept"] for c in cands]
    assert "stream" in names
    ans = lambda wo, concept=None: {"occurrence_verb": True, "word_outcome": wo, "base": "livestream", "concept": concept}
    e = A.entry_from_answer("restream", ans("join", "stream"), [4.0] * len(cands), cands, table)
    assert e["kind"] == "join" and e["cell"] == 3
    e = A.entry_from_answer("restream", ans("not_a_verb_word"), None, cands, table)
    assert e["kind"] == "not_verb"
    exp = [3.0 if n == "stream" else 0.5 for n in names]
    e = A.entry_from_answer("restream", ans("new"), exp, cands, table)
    assert e["kind"] == "new" and e["cell"] == 3 and e["near"] == "stream"   # very close: the same cell
    exp = [1.2 if n == "stream" else 0.1 for n in names]
    e = A.entry_from_answer("restream", ans("new"), exp, cands, table)
    assert e["kind"] == "new" and e["cell"] == A.new_cell("restream") and e["neighbours"] == [3]
    assert A.entry_from_answer("x", None, None, cands, table)["kind"] == "leave_out"
    e = A.entry_from_answer("restream", ans("join", "unknownverb"), [0.0] * len(cands), cands, table)
    assert e["kind"] == "new"                                   # a join to a verb nobody knows becomes new
    assert A.new_cell("livestream") >= A.NEW_CELL_BASE and A.store_key("x") == "vcat::sense:x"


# ---- the part as a nam_heads learner ----------------------------------------------------------------

@pytest.fixture
def learner(table, monkeypatch):
    from nam_heads.impls.affordance.default import senses
    monkeypatch.setattr(senses, "table", lambda: table)
    return A.AffordanceLearner("affordance", MemoryStore(StoreSpec(name="affordance")))


PASSAGE = {"key": "Band_narrative", "title": "Band",
           "text": "The band started small. Later the band restreamed the concert to fans."}


def test_the_part_passes_the_sdk_conformance_check(learner):
    cases = [{"value": "restream", "occurrence": {"token": "restreamed",
                                                  "sentence": "Later the band restreamed the concert ."},
              "record": PASSAGE},
             {"value": "telecast", "occurrence": {"token": "telecast", "sentence": "They telecast it ."},
              "record": {"key": "k", "title": "", "text": ""}}]
    assert T.check_learner(learner, cases) == []


def test_items_carry_the_whole_record_and_its_title(learner):
    it = learner.build_item("7|restream", "restream",
                            {"token": "restreamed", "sentence": "Later the band restreamed the concert ."},
                            PASSAGE, {})
    assert it["item"] == "7|restream" and it["record"]["title"] == "Band"
    assert it["record"]["text"] == ("The band started small. Later the band ⟦restreamed⟧ the concert to fans.")


def test_pending_skips_seed_verbs_and_returns_stored_decisions(learner):
    learner.store.put_many(learner.store_writes({"restream": {"kind": "not_verb"}}))
    todo, decided = learner.pending(["restream", "stream", "telecast"])
    assert todo == ["telecast"] and decided == {"restream": {"kind": "not_verb"}}
    assert learner.decision_key("x") == "vcat::sense:x"


def test_decisions_from_server_answers_and_malformed_ones(learner):
    it = learner.build_item("1|restream", "restream", {"token": "restreamed"}, PASSAGE, {})
    names = [c["concept"] for c in it["candidates"]]
    good = {"item": "1|restream", "answer": {"word_outcome": "join", "concept": "stream", "base": "restream"},
            "expected_scores": [4.0] * len(names)}
    d = learner.decision("restream", it, good, {})
    assert d["kind"] == "join" and d["cell"] == 3 and d["model"] == "head-learner-affordance-r5"
    for bad in (None, "x", {}, {"answer": {"occurrence_verb": True}}, {"answer": "join", "expected_scores": 3}):
        assert learner.decision("restream", it, bad, {})["kind"] == "leave_out"


def test_gpu_workload_comes_from_impl_meta_and_the_host_overrides_it(table, monkeypatch):
    from nam_heads.impls.affordance.default import senses
    monkeypatch.setattr(senses, "table", lambda: table)
    st = MemoryStore(StoreSpec(name="affordance"))
    gpu = A.AffordanceLearner("affordance", st).gpu_workload()
    assert gpu["image"] == "tdenton8772/nam-head-learner:v0.0.1" and gpu["port"] == 50051
    over = A.AffordanceLearner("affordance", st, {"gpu": {"model_id": "head-learner-affordance-r6"}})
    assert over.gpu_workload()["model_id"] == "head-learner-affordance-r6" and over.model == "head-learner-affordance-r6"
    assert json.loads(over.store_writes({"v": {"kind": "new", "cell": 3}})[0][1]) == {"kind": "new", "cell": 3}
