"""The affordance head's learner part (nam_heads.sdk.learner, declared in head.py's IMPL_META;
docs/design-head-learning.md section 7): turn a parked verb into a learner item, and the learner's answer
into a learned sense entry for the head's store.

Items have the shape the model was trained on (scripts/lexicon/lexicon_prompt.py; nam-data/lexicon
testset_v1/build_testset.py made the training candidates the same way):
  candidates  a word WordNet knows: its synset / antonym / sister / hypernym neighbours that the sense table
              knows; any other word: the known verbs among its stems (predicates.surface_candidates) and its
              close spellings. At most 8; each with its manner cell's members and one corpus example.
  record      the record's title and text (the host's view of the main record: __title__, __payload__)
              with the occurrence marked ⟦like this⟧; the head's sentence alone when the text lacks it.

Answer -> entry (stored as vcat::sense:<value>, read by the head, heads/affordance/senses.py):
  not_a_verb_word                 {"kind": "not_verb"}
  join <a candidate / known verb> {"kind": "join", "cell": that verb's cell}
  new, best candidate score >= 2.5 (the manner cells' own threshold)  {"kind": "new", "cell": that cell}
  new otherwise                   {"kind": "new", "cell": a new cell id of its own}
  no usable answer                {"kind": "leave_out"} (its records are written without the value)
"""
from __future__ import annotations

import difflib
import re
import time
import zlib
from typing import Dict, List, Optional

from nam_heads.sdk.learner import HeadLearner

from . import predicates as P
from .senses import LEARNED_PREFIX, SenseTable

MAX_CANDIDATES = 8
CELL_JOIN_SCORE = 2.5
NEW_CELL_BASE = 500_000              # learned cells of their own (seed cells are 1..N, unknown 900,000+)


def store_key(value: str) -> str:
    """The session key of a learned entry (the affordance store's vcat:: prefix)."""
    return f"vcat::{LEARNED_PREFIX}{value}"


def new_cell(word: str) -> int:
    return NEW_CELL_BASE + zlib.crc32(word.encode("utf-8")) % 399_989


def _cell_of(table: SenseTable, word: str, learned: Dict[str, dict]) -> Optional[int]:
    c = table.seed_cell(word)
    if c is not None:
        return c
    d = learned.get(word) or {}
    return int(d["cell"]) if d.get("cell") else None


def candidates(word: str, table: SenseTable, learned: Optional[Dict[str, dict]] = None) -> List[dict]:
    learned = learned or {}
    known = lambda w: table.seed_cell(w) is not None or bool((learned.get(w) or {}).get("cell"))
    out, seen = [], set()

    def add(w: str, kind: str):
        if w and w != word and w not in seen and known(w) and len(out) < MAX_CANDIDATES:
            seen.add(w)
            cell = _cell_of(table, w, learned)
            members = [m for m in (table.cells.get(cell, {}).get("members") or [w]) if m != word]
            if w not in members:
                members = [w] + members
            out.append({"concept": w, "members": members[:6], "kind": kind,
                        "example": table.examples.get(w, "")})
    if word in table.wordnet:
        for w, kind in table.wordnet[word]:
            add(w, kind)
    else:
        for c in P.surface_candidates(word):
            add(c, "stem")
        pool = [w for w, n in table.counts.items() if n > 2]
        for c in difflib.get_close_matches(word, pool, n=4, cutoff=0.8):
            add(c, "spelling")
    # the training items listed candidates in shuffled order; a fixed, word-seeded order here
    out.sort(key=lambda c: zlib.crc32((word + "|" + c["concept"]).encode()))
    return out


def mark(text: str, token: str, sentence: str = "") -> Optional[str]:
    """``text`` with one occurrence of ``token`` marked ⟦token⟧: the one whose three words on each side best
    match the words around the token in ``sentence`` (the head's token-joined sentence), else the first."""
    ms = list(re.finditer(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", text))
    if not ms:
        return None
    best = ms[0]
    if len(ms) > 1 and sentence:
        sw = [w.lower() for w in re.findall(r"\w+", sentence)]
        ti = next((k for k, w in enumerate(sw) if w == token.lower()), None)
        if ti is not None:
            want = sw[max(0, ti - 3):ti] + sw[ti + 1:ti + 4]

            def score(m):
                before = [w.lower() for w in re.findall(r"\w+", text[:m.start()])][-3:]
                after = [w.lower() for w in re.findall(r"\w+", text[m.end():])][:3]
                return sum(1 for w in before + after if w in want)
            best = max(ms, key=score)
    return text[:best.start()] + "\u27e6" + token + "\u27e7" + text[best.end():]


def build_item(item_id: str, value: str, occ: dict, record: dict, table: SenseTable,
               learned: Optional[Dict[str, dict]] = None) -> Optional[dict]:
    """A learner item for ``value`` occurring in ``record`` (the host's ``{"key", "title", "text"}``), or
    None if the occurrence is in neither the record text nor the head's sentence."""
    record = record or {}
    text = record.get("text") or ""
    token = (occ or {}).get("token") or value
    marked = mark(text, token, (occ or {}).get("sentence", "")) if text else None
    if marked is None and occ and occ.get("sentence"):
        marked = mark(occ["sentence"], token)            # the sentence alone, as the head saw it
    if marked is None:
        return None
    return {"item": item_id, "record": {"title": record.get("title") or "", "text": marked},
            "token": token, "word": value, "candidates": candidates(value, table, learned)}


def entry_from_answer(value: str, answer: Optional[dict], expected: Optional[list], cands: List[dict],
                      table: SenseTable, learned: Optional[Dict[str, dict]] = None,
                      model: str = "") -> dict:
    """The learned store entry for ``value`` from the learner's answer."""
    learned = learned or {}
    base = {"value": value, "decided_at": round(time.time(), 1), "model": model,
            "scores": dict(zip([c["concept"] for c in cands], expected or []))}
    if not answer:
        return dict(base, kind="leave_out", reason="no answer")
    if answer["word_outcome"] == "not_a_verb_word":
        return dict(base, kind="not_verb")
    if answer["word_outcome"] == "join" and answer.get("concept"):
        cell = _cell_of(table, answer["concept"], learned)
        if cell is not None:
            return dict(base, kind="join", cell=cell, concept=answer["concept"], base=answer.get("base"))
    best, best_s = None, -1.0
    for c, s in zip(cands, expected or []):
        if s is not None and s > best_s:
            best, best_s = c["concept"], float(s)
    if best is not None and best_s >= CELL_JOIN_SCORE:
        return dict(base, kind="new", cell=_cell_of(table, best, learned), near=best, base=answer.get("base"))
    neighbours = sorted({_cell_of(table, c["concept"], learned) for c, s in zip(cands, expected or [])
                         if s is not None and s >= 1.0} - {None})
    return dict(base, kind="new", cell=new_cell(value), neighbours=neighbours, base=answer.get("base"))


class AffordanceLearner(HeadLearner):
    """The learner part: a parked verb -> a learned sense entry (``vcat::sense:<value>``), decided by the
    graded learner on the GPU (head.py IMPL_META "learner" gpu; the chart's headLearning.learners.affordance
    overrides the image and model id)."""

    def _initialize(self) -> None:
        from .senses import table
        self.table = table()
        self.model = (self.gpu_workload() or {}).get("model_id", "")

    def decision_key(self, value: str) -> str:
        return store_key(value)

    def pending(self, values):
        todo, decided = super().pending(values)
        # a verb the seed table knows needs no decision (an earlier session may have parked it before
        # the table had it)
        return [v for v in todo if self.table.seed_cell(v) is None], decided

    def build_item(self, item_id, value, occurrence, record, decided):
        return build_item(item_id, value, occurrence, record, self.table, dict(decided))

    def decision(self, value, item, answer, decided):
        a = answer if isinstance(answer, dict) else {}
        inner = a.get("answer")
        if not isinstance(inner, dict) or not inner.get("word_outcome"):
            inner = None
        expected = a.get("expected_scores")
        if not isinstance(expected, list):
            expected = None
        return entry_from_answer(value, inner, expected, item.get("candidates") or [], self.table,
                                 dict(decided), model=self.model)
