"""
Rule-based Named Entity Recognizer.

Three layers (first match wins per token span):
1. Date/time regex patterns
2. Gazetteer matching (forward-maximum multi-token)
3. Pattern grouping (consecutive PROPN sequences)

Produces IOB tags: B for first token, I for continuation, O for outside.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# -- Date/time patterns --------------------------------------------------

_MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
)
_MONTH_SET = frozenset(_MONTHS)

_DAYS = (
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "mon", "tue", "tues", "wed", "thu",
    "thurs", "fri", "sat", "sun",
)
_DAY_SET = frozenset(_DAYS)

# Ordinal suffixes
_ORDINAL_RE = re.compile(r"^\d{1,2}(?:st|nd|rd|th)$", re.IGNORECASE)

# Year patterns
_YEAR_RE = re.compile(r"^(?:1[0-9]{3}|20[0-9]{2})s?$")

# Century patterns
_CENTURY_RE = re.compile(r"^\d{1,2}(?:st|nd|rd|th)$", re.IGNORECASE)

# Clock time
_TIME_RE = re.compile(
    r"^\d{1,2}:\d{2}(?::\d{2})?(?:\s*(?:am|pm|AM|PM|a\.m\.|p\.m\.))?$"
)

# Corporate suffixes for ORG detection
_CORP_SUFFIXES = frozenset({
    "inc", "inc.", "corp", "corp.", "ltd", "ltd.", "llc", "plc",
    "co", "co.", "group", "foundation", "institute", "university",
    "college", "association", "committee", "commission", "council",
    "department", "ministry", "agency", "bureau", "authority",
    "bank", "airlines", "airways", "motors", "industries",
    "technologies", "systems", "solutions", "services", "partners",
    "holdings", "enterprises", "international", "corporation",
    "company",
})

# Title prefixes that extend PERSON spans
_TITLE_PREFIXES = frozenset({
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "dr", "dr.",
    "prof", "prof.", "sir", "dame", "lord", "lady",
    "president", "senator", "governor", "mayor",
    "gen", "gen.", "col", "col.", "capt", "capt.",
    "sgt", "sgt.", "lt", "lt.", "rev", "rev.",
    "king", "queen", "prince", "princess", "emperor", "empress",
    "pope", "saint", "st", "st.",
})


class NERRecognizer:
    """Rule-based NER with gazetteers + patterns."""

    def __init__(self, data_dir: str):
        self._gazetteers: Dict[str, Set[str]] = {}
        self._load_gazetteers(data_dir)

    def _load_gazetteers(self, data_dir: str) -> None:
        gaz_dir = os.path.join(data_dir, "gazetteers")
        if not os.path.isdir(gaz_dir):
            return

        _file_label_map = {
            "person_first.txt": "_first_names",
            "person_last.txt": "_last_names",
            "countries.txt": "GPE",
            "cities.txt": "GPE",
            "organizations.txt": "ORG",
            "nationalities.txt": "NORP",
        }

        for filename, key in _file_label_map.items():
            path = os.path.join(gaz_dir, filename)
            if os.path.exists(path):
                with open(path) as f:
                    entries = {line.strip().lower() for line in f if line.strip()}
                if key.startswith("_"):
                    # Internal lookup (not a label)
                    self._gazetteers[key] = entries
                elif key in self._gazetteers:
                    self._gazetteers[key].update(entries)
                else:
                    self._gazetteers[key] = entries

        logger.debug(
            "ner.loaded gazetteers: %s",
            {k: len(v) for k, v in self._gazetteers.items()},
        )

    def recognize(
        self, tokens: list, sentences: list
    ) -> List[dict]:
        """
        Recognize named entities in tokens.

        Mutates token .ent_type and .ent_iob fields in-place.

        Returns list of entity dicts:
            {"text": str, "label": str, "start": int, "end": int}
        """
        n = len(tokens)
        # Track which tokens are already claimed by an entity
        claimed = [False] * n
        entities: List[dict] = []

        # Layer 1: Date/time patterns
        self._date_time_pass(tokens, claimed, entities)

        # Layer 2: Gazetteer matching
        self._gazetteer_pass(tokens, claimed, entities)

        # Layer 3: Pattern grouping (consecutive PROPN)
        self._pattern_pass(tokens, claimed, entities)

        # Sort entities by start position
        entities.sort(key=lambda e: e["start"])

        # Set IOB tags on tokens
        for tok in tokens:
            if not tok.ent_type:
                tok.ent_iob = "O"

        return entities

    # -- Layer 1: Date/time -----------------------------------------------

    def _date_time_pass(
        self, tokens: list, claimed: list, entities: list
    ) -> None:
        n = len(tokens)
        i = 0
        while i < n:
            if claimed[i]:
                i += 1
                continue

            span = self._try_date_time(tokens, i)
            if span:
                start, end, label = span
                self._mark_entity(tokens, claimed, entities, start, end, label)
                i = end
            else:
                i += 1

    def _try_date_time(
        self, tokens: list, i: int
    ) -> Optional[Tuple[int, int, str]]:
        """Try to match a date/time pattern starting at index i."""
        n = len(tokens)
        lower = tokens[i].text.lower()

        # "January 15, 2024" or "January 2024" or "January 15"
        if lower in _MONTH_SET:
            end = i + 1
            # Optional day number
            if end < n and (tokens[end].like_num or _ORDINAL_RE.match(tokens[end].text)):
                end += 1
                # Skip comma
                if end < n and tokens[end].text == ",":
                    end += 1
            # Optional year
            if end < n and _YEAR_RE.match(tokens[end].text):
                end += 1
            if end > i + 1:
                return (i, end, "DATE")
            # Bare month name alone
            return (i, i + 1, "DATE")

        # Day names
        if lower in _DAY_SET:
            return (i, i + 1, "DATE")

        # "the 1990s", "1990s", "2024"
        if _YEAR_RE.match(tokens[i].text):
            return (i, i + 1, "DATE")

        # "14th century"
        if (
            _CENTURY_RE.match(tokens[i].text)
            and i + 1 < n
            and tokens[i + 1].text.lower() == "century"
        ):
            return (i, i + 2, "DATE")

        # Clock time "3:00", "3:00 pm"
        if _TIME_RE.match(tokens[i].text):
            return (i, i + 1, "TIME")

        # "am"/"pm" after a number
        if (
            lower in ("am", "pm", "a.m.", "p.m.", "a.m", "p.m")
            and i > 0
            and tokens[i - 1].like_num
        ):
            # Already claimed by the number; extend if not claimed
            return None

        return None

    # -- Layer 2: Gazetteer matching --------------------------------------

    def _merged_gazetteer(self) -> Dict[str, str]:
        """span -> label over every public gazetteer, first label (dict order)
        winning, as the per-label loop did. Rebuilt only when the gazetteers
        change (they are loaded once; tests may replace them)."""
        fp = tuple((k, len(v)) for k, v in self._gazetteers.items())
        if getattr(self, "_gaz_fp", None) != fp:
            merged: Dict[str, str] = {}
            for label, entries in self._gazetteers.items():
                if label.startswith("_"):
                    continue
                for entry in entries:
                    merged.setdefault(entry, label)
            self._gaz_merged, self._gaz_fp = merged, fp
            # First word of every entry: a span can only equal an entry whose
            # first word equals the span's first word.
            self._gaz_first_words = {e.split(" ", 1)[0] for e in merged}
        return self._gaz_merged

    def _gazetteer_pass(
        self, tokens: list, claimed: list, entities: list
    ) -> None:
        """Forward-maximum multi-token gazetteer matching.

        One dict lookup per candidate span (the merged gazetteer) instead of a
        scan over every label, and token texts gathered once: this pass was
        ~half of the rule backend's parse time.
        """
        n = len(tokens)
        merged = self._merged_gazetteer()
        first_words = self._gaz_first_words
        texts = [t.text for t in tokens]
        first_names = self._gazetteers.get("_first_names", set())
        last_names = self._gazetteers.get("_last_names", set())
        i = 0
        while i < n:
            if claimed[i]:
                i += 1
                continue

            best_match = None  # (end, label)

            # Try longest match first (up to 5 tokens). Every span starting
            # here begins with this token's lower-cased text (lowering a
            # token alone equals lowering it before a space), so when no
            # entry starts with that word only the single-token checks can
            # match. A token containing a space skips the shortcut.
            max_len = min(5, n - i)
            if " " not in texts[i] and texts[i].lower() not in first_words:
                max_len = min(max_len, 1)
            for length in range(max_len, 0, -1):
                end = i + length
                span_text = " ".join(texts[i:end]).lower()

                label = merged.get(span_text)
                if label is not None:
                    best_match = (end, label)
                    break

                # Single-token PERSON check: first name + last name
                if length == 1:
                    word_lower = texts[i].lower()

                    if (
                        word_lower in first_names
                        and tokens[i].pos == "PROPN"
                    ):
                        # Check if followed by a last name
                        if i + 1 < n and tokens[i + 1].text.lower() in last_names:
                            # Two-token person name
                            best_match = (i + 2, "PERSON")
                            break

            if best_match:
                end, label = best_match
                self._mark_entity(tokens, claimed, entities, i, end, label)
                i = end
            else:
                i += 1

    # -- Layer 3: Pattern grouping ----------------------------------------

    def _pattern_pass(
        self, tokens: list, claimed: list, entities: list
    ) -> None:
        """Group consecutive PROPN tokens into entities."""
        n = len(tokens)
        i = 0
        while i < n:
            if claimed[i] or tokens[i].pos != "PROPN":
                i += 1
                continue

            # Check for title prefix
            has_title = False
            if tokens[i].text.lower() in _TITLE_PREFIXES:
                has_title = True

            # Collect consecutive PROPN tokens
            start = i
            j = i + 1
            while j < n and not claimed[j] and tokens[j].pos == "PROPN":
                j += 1

            span_len = j - start

            if span_len == 0:
                i += 1
                continue

            # Determine label based on span properties
            label = self._classify_propn_span(tokens, start, j, has_title)

            if span_len >= 1:
                self._mark_entity(tokens, claimed, entities, start, j, label)

            i = j

    def _classify_propn_span(
        self, tokens: list, start: int, end: int, has_title: bool
    ) -> str:
        """Classify a PROPN span as PERSON, ORG, or GPE."""
        span_len = end - start

        # Check for corporate suffixes -> ORG
        last_lower = tokens[end - 1].text.lower()
        if last_lower in _CORP_SUFFIXES:
            return "ORG"

        # Has title prefix -> PERSON
        if has_title:
            return "PERSON"

        # 2-3 tokens without corporate suffix -> PERSON
        if span_len <= 3:
            return "PERSON"

        # 4+ tokens -> ORG
        return "ORG"

    # -- Helpers ----------------------------------------------------------

    def _mark_entity(
        self,
        tokens: list,
        claimed: list,
        entities: list,
        start: int,
        end: int,
        label: str,
    ) -> None:
        """Mark tokens with entity type and IOB tags."""
        text = " ".join(t.text for t in tokens[start:end])
        entities.append({
            "text": text,
            "label": label,
            "start": start,
            "end": end,
        })

        for k in range(start, end):
            claimed[k] = True
            tokens[k].ent_type = label
            tokens[k].ent_iob = "B" if k == start else "I"
