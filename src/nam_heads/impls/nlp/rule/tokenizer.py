"""
Regex-based tokenizer and sentence splitter.

Produces RawToken objects that are progressively enriched by later
pipeline stages (POS tagger, lemmatizer, NER, dep parser).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


class RawToken:
    """Mutable token accumulator used throughout the pipeline."""

    __slots__ = (
        "text", "lemma", "pos", "tag", "dep", "ent_type", "ent_iob",
        "head_idx", "idx", "is_stop", "is_punct", "is_space", "like_num",
        "morph_data",
    )

    def __init__(self, text: str, idx: int):
        self.text = text
        self.lemma = text.lower()
        self.pos = ""
        self.tag = ""
        self.dep = ""
        self.ent_type = ""
        self.ent_iob = "O"
        self.head_idx = idx
        self.idx = idx
        self.is_stop = False
        self.is_punct = False
        self.is_space = False
        self.like_num = False
        self.morph_data: Dict[str, List[str]] = {}


# -- contractions --------------------------------------------------------

_CONTRACTIONS: Dict[str, List[str]] = {
    "n't":  ["n't"],
    "'re":  ["'re"],
    "'ve":  ["'ve"],
    "'ll":  ["'ll"],
    "'d":   ["'d"],
    "'m":   ["'m"],
    "'s":   ["'s"],
}

# Every suffix above contains an apostrophe (lower() never changes it), so a
# span without one cannot match any of them. Derived, so adding a suffix
# without an apostrophe turns the shortcut off rather than breaking it.
_CONTRACTIONS_NEED_APOSTROPHE = all("'" in s for s in _CONTRACTIONS)

# Full-word contractions that need splitting
_WORD_CONTRACTIONS: Dict[str, List[str]] = {
    "cannot":   ["can", "not"],
    "gonna":    ["going", "to"],
    "wanna":    ["want", "to"],
    "gotta":    ["got", "to"],
    "lemme":    ["let", "me"],
    "gimme":    ["give", "me"],
}

# -- abbreviation guard for sentence splitting ---------------------------

_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "blvd",
    "dept", "est", "fig", "gen", "gov", "inc", "corp", "ltd", "co",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct",
    "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "vs", "etc", "approx", "dept", "assn", "bros", "no", "vol",
    "rev", "sgt", "cpl", "pvt", "capt", "maj", "col", "lt",
    "cmdr", "adm", "pvt", "drs", "rep", "sen",
    "u.s", "u.k", "e.g", "i.e", "a.m", "p.m",
})

# -- number detection ----------------------------------------------------

_NUM_RE = re.compile(
    r"^[+-]?"
    r"(?:\d[\d,]*\.?\d*|\.\d+)"
    r"(?:[eE][+-]?\d+)?$"
)

_NUM_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
    "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion", "trillion",
    "first", "second", "third", "fourth", "fifth",
})

# -- main word regex -----------------------------------------------------
# Matches words (with hyphens/apostrophes), or single non-word/non-space chars
_TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")


class Tokenizer:
    """Regex tokenizer with contraction splitting and sentence detection."""

    def tokenize(self, text: str) -> Tuple[List[RawToken], List[dict]]:
        """
        Tokenize text into RawToken list and sentence span dicts.

        Returns:
            (tokens, sentences) where sentences is a list of
            {"start": int, "end": int, "text": str} dicts with token indices.
        """
        raw_pieces = self._split_raw(text)
        tokens = self._build_tokens(raw_pieces)
        sentences = self._split_sentences(tokens)
        return tokens, sentences

    # -- internals -------------------------------------------------------

    def _split_raw(self, text: str) -> List[str]:
        """Regex tokenize, then split contractions."""
        spans = _TOKEN_RE.findall(text)
        pieces: List[str] = []
        for span in spans:
            lower = span.lower()

            # whole-word contractions
            if lower in _WORD_CONTRACTIONS:
                pieces.extend(_WORD_CONTRACTIONS[lower])
                continue

            # apostrophe contractions
            if _CONTRACTIONS_NEED_APOSTROPHE and "'" not in lower:
                pieces.append(span)
                continue
            split = False
            for suffix, replacement in _CONTRACTIONS.items():
                if lower.endswith(suffix) and len(span) > len(suffix):
                    prefix = span[: -len(suffix)]
                    pieces.append(prefix)
                    pieces.extend(replacement)
                    split = True
                    break
            if not split:
                pieces.append(span)

        return pieces

    def _build_tokens(self, pieces: List[str]) -> List[RawToken]:
        """Create RawToken objects with basic flags set."""
        tokens: List[RawToken] = []
        for i, piece in enumerate(pieces):
            tok = RawToken(piece, i)
            tok.is_punct = len(piece) == 1 and not piece.isalnum()
            tok.is_space = piece.isspace()
            tok.like_num = bool(_NUM_RE.match(piece)) or piece.lower() in _NUM_WORDS
            tokens.append(tok)
        return tokens

    def _split_sentences(
        self, tokens: List[RawToken]
    ) -> List[dict]:
        """
        Split tokens into sentence spans.

        Rules:
        - Sentence boundary at `.` `!` `?` followed by a capitalized word
          or end of tokens.
        - Abbreviation guard: don't split after known abbreviations.
        """
        if not tokens:
            return []

        sentence_starts: List[int] = [0]

        for i, tok in enumerate(tokens):
            if tok.text not in ".!?":
                continue

            # Check abbreviation guard
            if tok.text == "." and i > 0:
                prev_lower = tokens[i - 1].text.lower().rstrip(".")
                if prev_lower in _ABBREVIATIONS:
                    continue

            # Look ahead for a capitalized word (skip punctuation/space)
            j = i + 1
            while j < len(tokens) and (tokens[j].is_punct or tokens[j].is_space):
                j += 1

            if j < len(tokens) and tokens[j].text[0:1].isupper():
                sentence_starts.append(j)
            elif j >= len(tokens):
                # End of text after sentence-ending punctuation -- no new sentence
                pass

        # Build sentence dicts
        sentences: List[dict] = []
        for si, start in enumerate(sentence_starts):
            end = sentence_starts[si + 1] if si + 1 < len(sentence_starts) else len(tokens)
            sent_text = " ".join(t.text for t in tokens[start:end])
            sentences.append({"start": start, "end": end, "text": sent_text})

        return sentences
