"""
Rule-based lemmatizer.

Two stages:
1. Exception dictionary (~3k entries): irregular verbs, nouns, adjectives
2. POS-conditioned suffix stripping: -ing/-ed/-es/-s with doubled-consonant
   and y-rules for verbs; -ies/-es/-s for nouns; -er/-est for adjectives.

Output always lowercased.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_VOWELS = frozenset("aeiou")

# Doubled consonants that should be un-doubled after stripping -ing/-ed
# e.g. "running" -> strip "ning" -> "run" not "runn"
_DOUBLE_CONSONANTS = frozenset({
    "bb", "cc", "dd", "ff", "gg", "ll", "mm", "nn", "pp", "rr", "ss", "tt", "zz",
})


class Lemmatizer:
    """Rule-based lemmatizer with exception dictionary + suffix rules."""

    def __init__(self, data_dir: str):
        self._exceptions: Dict[str, str] = {}
        path = os.path.join(data_dir, "lemma_exceptions.json")
        if os.path.exists(path):
            with open(path) as f:
                self._exceptions = json.load(f)
            logger.debug("lemmatizer.loaded exceptions=%d", len(self._exceptions))

    def lemmatize(self, tokens: list) -> None:
        """Lemmatize tokens in-place. Mutates each token's .lemma field."""
        for tok in tokens:
            tok.lemma = self._lemmatize_one(tok.text.lower(), tok.pos)

    def _lemmatize_one(self, word: str, pos: str) -> str:
        # Stage 1: exception dictionary
        if word in self._exceptions:
            return self._exceptions[word]

        # Stage 2: POS-conditioned suffix stripping
        if pos == "VERB":
            return self._lemmatize_verb(word)
        elif pos == "NOUN":
            return self._lemmatize_noun(word)
        elif pos == "ADJ":
            return self._lemmatize_adj(word)

        return word

    def _lemmatize_verb(self, word: str) -> str:
        """Strip verb inflections: -ing, -ed, -es, -s."""
        # -ing
        if word.endswith("ying") and len(word) > 4:
            # "dying" -> "die", "lying" -> "lie"
            candidate = word[:-4] + "ie"
            if candidate in self._exceptions or len(candidate) >= 3:
                return candidate

        if word.endswith("ing") and len(word) > 4:
            stem = word[:-3]
            # doubled consonant: "running" -> "run"
            if len(stem) >= 2 and stem[-2:] in _DOUBLE_CONSONANTS:
                return stem[:-1]
            # Stems ending in -er/-ew/-ow: vowel is part of the word, not a dropped silent-e
            # "viewing" -> "view", "following" -> "follow", "pioneering" -> "pioneer"
            if stem.endswith(("er", "ew", "ow")):
                return stem
            # "making" -> "make" (silent-e restoration for VCe pattern)
            # Only when penultimate is vowel: "making" -> "mak" -> "make"
            # Not for "founding" -> "found" (penultimate 'n' is consonant)
            if (len(stem) >= 2
                    and stem[-1] not in _VOWELS
                    and stem[-1] != "y"
                    and stem[-2] in _VOWELS):
                return stem + "e"
            return stem

        # -ed
        if word.endswith("ied") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("ed") and len(word) > 3:
            stem = word[:-2]
            if not stem:
                return word
            # doubled consonant: "stopped" -> "stop"
            if len(stem) >= 2 and stem[-2:] in _DOUBLE_CONSONANTS:
                return stem[:-1]
            # "loved" -> "love" (already ends in e after stripping d)
            if word.endswith("ed") and not word.endswith("eed"):
                if stem[-1] == "e":
                    return stem
                # Stems ending in -er/-ew/-ow: vowel is part of the word, not a dropped silent-e
                # "viewed" -> "view", "followed" -> "follow", "pioneered" -> "pioneer"
                if stem.endswith(("er", "ew", "ow")):
                    return stem
                # silent-e restoration for VCe pattern: "hoped" -> "hope"
                # Only when penultimate is vowel (VCe base form)
                # "founded" -> "found" (no e), "turned" -> "turn" (no e)
                if (len(stem) >= 2
                        and stem[-1] not in _VOWELS
                        and stem[-1] != "y"
                        and stem[-2] in _VOWELS):
                    return stem + "e"
            return stem

        # -es
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"
        if word.endswith("es") and len(word) > 3:
            if word.endswith(("ches", "shes", "sses", "xes", "zes")):
                return word[:-2]
            return word[:-1]  # "makes" -> "make" handled by -s below won't reach here

        # -s (but not -ss)
        if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
            return word[:-1]

        return word

    def _lemmatize_noun(self, word: str) -> str:
        """Strip noun inflections: -ies, -es, -s."""
        if word.endswith("ies") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("es") and len(word) > 3:
            if word.endswith(("ches", "shes", "sses", "xes", "zes")):
                return word[:-2]

        if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
            return word[:-1]

        return word

    def _lemmatize_adj(self, word: str) -> str:
        """Strip adjective inflections: -er, -est."""
        if word.endswith("iest") and len(word) > 5:
            return word[:-4] + "y"

        if word.endswith("est") and len(word) > 4:
            stem = word[:-3]
            if len(stem) >= 2 and stem[-2:] in _DOUBLE_CONSONANTS:
                return stem[:-1]
            if stem[-1] not in _VOWELS and len(stem) >= 2:
                return stem + "e"
            return stem

        if word.endswith("ier") and len(word) > 4:
            return word[:-3] + "y"

        if word.endswith("er") and len(word) > 3:
            stem = word[:-2]
            if len(stem) >= 2 and stem[-2:] in _DOUBLE_CONSONANTS:
                return stem[:-1]
            if stem[-1] not in _VOWELS and len(stem) >= 2:
                return stem + "e"
            return stem

        return word
