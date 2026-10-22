"""Verb senses for the affordance axis (docs/design-learned-verb-senses.md, docs/design-head-learning.md).

``NAM_AFFORDANCE_SENSES=1`` switches the affordance head and its axis encoder from the synonym fold and the
CharCNN code to the sense table: the seed graph compiled by scripts/lexicon/build_affordance_seed.py
(data/sense_seed.json) plus what the learning session writes to the store (``vcat::sense:<word>``).

- A verb resolves to a sense: ``word|frame`` entry (the frame is the token after the verb when it is a
  particle or preposition, the same rule the seed graph used), else the word entry, else the dominant sense
  of a word with several, else a learned entry. A teacher non-verb or a learned "not a verb" / "leave out"
  is skipped; anything else is UNKNOWN, and the record parks for the learning queue.
- The emitted value is ``<word>@<cell>`` (the meaning, not just the word: head_codes is keyed by value), and
  its code is ``lca:<cell>:<fine>``: the coarse code is the sense's manner cell (1..N; 0 stays the axis's
  NULL), so a query matches the cell and its precompiled nearest cells; fine is a stable hash of the word.
- Values without "@" (the copular "classify" / "describe") are coded from the word's seed sense.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.encoders.axis_encoder import NULL_CODE, NULL_VALUE, AxisEncoder

FRAME_WORDS = frozenset(
    "up down out off in on over away back about around through into onto upon across along by for from "
    "with to at of as after against apart aside forward together under behind".split())
LEARNED_PREFIX = "sense:"            # store doc name (vcat::sense:<word>), written by the learning session
UNKNOWN_CELL_BASE = 900_000          # cells for values the table does not know (never a seed cell)
SEED_PATH = Path(__file__).parent / "data" / "sense_seed.json"

CELL, SKIP, UNKNOWN = "cell", "skip", "unknown"


def enabled() -> bool:
    return os.getenv("NAM_AFFORDANCE_SENSES", "0") == "1"


def fine_of(word: str) -> int:
    """The fine coordinate: a stable hash of the word (never used for retrieval, kept distinct per word)."""
    return zlib.crc32(word.encode("utf-8")) % 999_983 + 1


def value_of(word: str, cell: int) -> str:
    return f"{word}@{int(cell)}"


def parse_value(value: str) -> Optional[Tuple[str, int]]:
    w, sep, c = value.rpartition("@")
    if not sep or not w or not c.isdecimal():
        return None
    return w, int(c)


def frame_after(doc, i: int) -> str:
    """The frame of the verb at token ``i``: the next token if it is a particle or preposition, else '_'."""
    if i + 1 < len(doc):
        nxt = doc[i + 1].text.lower()
        if nxt in FRAME_WORDS:
            return nxt
    return "_"


class SenseTable:
    """The compiled seed (immutable) + resolution against it and learned store docs."""

    def __init__(self, seed: dict, version: str = ""):
        self.version = version or seed.get("version", "")
        senses = seed["senses"]
        self.sense_cell = {int(k): int(v[1]) for k, v in senses.items()}
        self.sense_word = {int(k): v[0] for k, v in senses.items()}
        self.word = {w: int(s) for w, s in seed["word"].items()}
        self.frame = {k: int(s) for k, s in seed["frame"].items()}
        self.dominant = {w: int(s) for w, s in seed["dominant"].items()}
        self.not_verb = frozenset(seed.get("not_verb", ()))
        self.cells = {int(k): v for k, v in seed["cells"].items()}
        self.next_cell = int(seed.get("next_cell", 1 + max(self.cells, default=0)))
        self.counts = seed.get("counts", {})
        self.examples = seed.get("examples", {})
        self.wordnet = seed.get("wordnet", {})

    def knows(self, word: str) -> bool:
        return word in self.word or word in self.dominant or word in self.not_verb

    def seed_cell(self, word: str, frame: str = "_") -> Optional[int]:
        sid = self.frame.get(f"{word}|{frame}")
        if sid is None:
            sid = self.word.get(word)
        if sid is None:
            sid = self.dominant.get(word)
        return None if sid is None else self.sense_cell[sid]

    def resolve(self, word: str, frame: str = "_", learned: Optional[dict] = None) -> Tuple[str, Optional[int]]:
        """(CELL, cell) | (SKIP, None) | (UNKNOWN, None) for a verb ``word`` in ``frame``. ``learned`` is the
        word's store doc (kind join/new with a cell, or not_verb / leave_out), if any."""
        cell = self.seed_cell(word, frame)
        if cell is not None:
            return CELL, cell
        if word in self.not_verb:
            return SKIP, None
        if learned:
            if learned.get("kind") in ("not_verb", "leave_out"):
                return SKIP, None
            if learned.get("cell"):
                return CELL, int(learned["cell"])
        return UNKNOWN, None

    def neighbours(self, cell: int, k: int) -> List[int]:
        c = self.cells.get(int(cell))
        return list(c["neighbours"][:k]) if c else []

    def code_for(self, value: str) -> str:
        if value == NULL_VALUE:
            return NULL_CODE
        pv = parse_value(value)
        if pv is not None:
            word, cell = pv
            return f"lca:{cell}:{fine_of(word)}"
        cell = self.seed_cell(value)
        if cell is None:
            cell = UNKNOWN_CELL_BASE + zlib.crc32(value.encode("utf-8")) % 99_991
        return f"lca:{cell}:{fine_of(value)}"


_TABLE: Optional[SenseTable] = None
_LOCK = threading.Lock()


def seed_path() -> Path:
    return Path(os.getenv("NAM_AFFORDANCE_SENSE_SEED", "") or SEED_PATH)


def table() -> SenseTable:
    """The process's sense table (loaded once)."""
    global _TABLE
    with _LOCK:
        if _TABLE is None:
            p = seed_path()
            raw = p.read_bytes()
            _TABLE = SenseTable(json.loads(raw), version=hashlib.sha256(raw).hexdigest()[:16])
        return _TABLE


class SenseTableAxisEncoder(AxisEncoder):
    """The affordance axis's encoder under NAM_AFFORDANCE_SENSES=1: codes from the sense table."""

    axis = "affordance"

    def __init__(self, tbl: Optional[SenseTable] = None):
        self.table = tbl or table()
        self.artifact_id = f"sense-table-{self.table.version}"

    def encode_values(self, values: List[str]) -> List[str]:
        return [self.table.code_for(v) for v in values]

    def nearest_coarse_codes(self, code: int, k: int = 3) -> List[int]:
        return self.table.neighbours(int(code), k)

    def get_metrics(self) -> Dict:
        return {"axis": self.axis, "artifact_id": self.artifact_id, "cells": len(self.table.cells)}
