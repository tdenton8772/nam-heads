"""
Rule-based POS tagger.

Six-tier cascade (each tier only tags untagged tokens):
1. Punctuation / space / number  (from tokenizer flags)
2. Function words                (closed-class lookup)
3. Dictionary lookup             (Brown corpus most-frequent POS)
4. Capitalization heuristic      (non-sentence-initial caps -> PROPN)
5. Suffix rules                  (longest-match-first)
6. Context disambiguation        (second pass: DET->NOUN, after "to"->VERB, etc.)

Also assigns fine-grained `tag` (VBN, VBG, VBZ, etc.) for attribute head.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class POSTagger:
    """Rule-based POS tagger with six-tier cascade."""

    def __init__(self, data_dir: str):
        self._pos_dict: Dict[str, str] = {}
        self._function_words: Dict[str, str] = {}
        self._suffix_rules: List[dict] = []
        self._stopwords: Set[str] = set()

        self._load_data(data_dir)

    def _load_data(self, data_dir: str) -> None:
        # POS dictionary (Brown corpus derived)
        path = os.path.join(data_dir, "pos_dictionary.json")
        if os.path.exists(path):
            with open(path) as f:
                self._pos_dict = json.load(f)
            logger.debug("pos_tagger.loaded pos_dict entries=%d", len(self._pos_dict))

        # Function words
        path = os.path.join(data_dir, "function_words.json")
        if os.path.exists(path):
            with open(path) as f:
                self._function_words = json.load(f)

        # Suffix rules (sorted by length descending for longest-match-first)
        path = os.path.join(data_dir, "suffixes.json")
        if os.path.exists(path):
            with open(path) as f:
                self._suffix_rules = json.load(f)
            self._suffix_rules.sort(key=lambda r: len(r["suffix"]), reverse=True)

        # Stopwords
        path = os.path.join(data_dir, "stopwords.json")
        if os.path.exists(path):
            with open(path) as f:
                self._stopwords = set(json.load(f))

    def tag(self, tokens: list, sentences: list) -> None:
        """
        Tag tokens in-place. Mutates each token's .pos, .tag, .is_stop fields.

        Args:
            tokens: list of RawToken (from tokenizer)
            sentences: list of sentence dicts with "start" key
        """
        if not tokens:
            return

        sentence_starts = {s["start"] for s in sentences}

        # --- Tier 1: punctuation / space / number ---
        for tok in tokens:
            if tok.is_punct:
                tok.pos = "PUNCT"
            elif tok.is_space:
                tok.pos = "SPACE"
            elif tok.like_num:
                tok.pos = "NUM"

        # --- Tier 2: function words ---
        for tok in tokens:
            if tok.pos:
                continue
            lower = tok.text.lower()
            if lower in self._function_words:
                tok.pos = self._function_words[lower]

        # --- Tier 3: dictionary lookup ---
        for tok in tokens:
            if tok.pos:
                continue
            lower = tok.text.lower()
            if lower in self._pos_dict:
                tok.pos = self._pos_dict[lower]

        # --- Tier 4: capitalization heuristic ---
        for tok in tokens:
            if tok.pos:
                continue
            if tok.text and tok.text[0].isupper():
                # Only apply if NOT at sentence start
                if tok.idx not in sentence_starts:
                    tok.pos = "PROPN"

        # --- Tier 5: suffix rules ---
        for tok in tokens:
            if tok.pos:
                continue
            lower = tok.text.lower()
            for rule in self._suffix_rules:
                if lower.endswith(rule["suffix"]) and len(lower) > len(rule["suffix"]):
                    tok.pos = rule["pos"]
                    break

        # --- Tier 6: context disambiguation (second pass) ---
        self._context_pass(tokens, sentence_starts)

        # --- Fine-grained tags ---
        self._assign_tags(tokens)

        # --- Stopword flags ---
        for tok in tokens:
            tok.is_stop = tok.text.lower() in self._stopwords

        # --- Default: anything still untagged -> NOUN ---
        for tok in tokens:
            if not tok.pos:
                tok.pos = "NOUN"

    def _context_pass(self, tokens: list, sentence_starts: set) -> None:
        """Second pass: reclassify based on context."""
        n = len(tokens)
        for i, tok in enumerate(tokens):
            lower = tok.text.lower()

            # After DET, reclassify VERB -> NOUN (e.g. "the run", "a call")
            # But NOT if the word ends in -ed/-en (likely participle: "the broken window")
            if (
                i > 0
                and tokens[i - 1].pos == "DET"
                and tok.pos == "VERB"
                and not lower.endswith(("ed", "en", "ing"))
            ):
                tok.pos = "NOUN"

            # After "to", reclassify NOUN -> VERB if it could be a verb
            if (
                i > 0
                and tokens[i - 1].text.lower() == "to"
                and tokens[i - 1].pos == "PART"
                and tok.pos == "NOUN"
                and lower in self._pos_dict
                and self._pos_dict.get(lower) == "VERB"
            ):
                tok.pos = "VERB"

            # After AUX + potential -ed/-en word, reclassify to VERB
            if (
                i > 0
                and tokens[i - 1].pos == "AUX"
                and tok.pos in ("ADJ", "NOUN")
                and (lower.endswith("ed") or lower.endswith("en"))
            ):
                tok.pos = "VERB"

            # "that" after VERB -> SCONJ (complementizer)
            if (
                lower == "that"
                and i > 0
                and tokens[i - 1].pos in ("VERB", "ADJ")
            ):
                tok.pos = "SCONJ"

            # Sentence-initial capitalized unknown word at tier 4 was skipped;
            # Try dictionary first, fallback to PROPN if truly unknown
            if not tok.pos and tok.idx in sentence_starts and tok.text[0:1].isupper():
                dict_pos = self._pos_dict.get(lower)
                if dict_pos:
                    tok.pos = dict_pos
                else:
                    tok.pos = "PROPN"

    def _assign_tags(self, tokens: list) -> None:
        """Assign fine-grained tags (VBN, VBG, VBZ, etc.)."""
        n = len(tokens)
        for i, tok in enumerate(tokens):
            if tok.pos == "VERB":
                lower = tok.text.lower()
                if lower.endswith("ing"):
                    tok.tag = "VBG"
                elif lower.endswith("ed") or lower.endswith("en"):
                    # Check if preceded by AUX "be" form -> VBN (past participle)
                    if i > 0 and tokens[i - 1].pos == "AUX":
                        aux_lower = tokens[i - 1].text.lower()
                        if aux_lower in (
                            "be", "am", "is", "are", "was", "were",
                            "been", "being",
                        ):
                            tok.tag = "VBN"
                        else:
                            tok.tag = "VBD" if lower.endswith("ed") else "VBN"
                    else:
                        tok.tag = "VBD" if lower.endswith("ed") else "VBN"
                elif lower.endswith("s") and not lower.endswith("ss"):
                    tok.tag = "VBZ"
                else:
                    tok.tag = "VB"
            elif tok.pos == "NOUN":
                tok.tag = "NN"
            elif tok.pos == "PROPN":
                tok.tag = "NNP"
            elif tok.pos == "ADJ":
                lower = tok.text.lower()
                if lower.endswith("er"):
                    tok.tag = "JJR"
                elif lower.endswith("est"):
                    tok.tag = "JJS"
                else:
                    tok.tag = "JJ"
            elif tok.pos == "ADV":
                tok.tag = "RB"
            elif tok.pos == "NUM":
                tok.tag = "CD"
            elif tok.pos:
                # Default: use POS as tag
                tok.tag = tok.pos
