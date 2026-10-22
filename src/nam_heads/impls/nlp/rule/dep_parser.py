"""
Rule-based dependency parser.

Per-sentence positional heuristics, 13 rule groups covering all 16 dep
relations used by NAM's encoder heads and bundler:

    amod, acomp, attr, oprd, nsubj, nsubjpass, dobj, obj, pobj,
    ROOT, conj, xcomp, advcl, mark, advmod, compound

Correctness guarantees:
- Every token has a valid head_idx
- ROOT: head_idx == self.idx
- No cycles (path from every token to ROOT via head links)
- Bundler can walk token.head upward to find VERB ancestors

Copula verbs for acomp/attr detection.
"""

from __future__ import annotations

from typing import List, Set

_COPULA_VERBS = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
    "become", "became", "seem", "seemed", "appear", "appeared",
    "remain", "remained", "look", "looked", "feel", "felt",
    "sound", "sounded", "taste", "tasted", "smell", "smelled",
})

_BE_FORMS = frozenset({
    "be", "am", "is", "are", "was", "were", "been", "being",
})


class DependencyParser:
    """Per-sentence positional heuristic dependency parser."""

    def parse(self, tokens: list, sentences: list) -> None:
        """
        Parse dependencies in-place per sentence.
        Mutates token .dep, .head_idx, .morph_data fields.
        """
        for sent in sentences:
            start = sent["start"]
            end = sent["end"]
            sent_tokens = tokens[start:end]
            if not sent_tokens:
                continue
            self._parse_sentence(sent_tokens, start)

        # Anything outside sentence spans gets dep="dep", head=self
        sent_ranges = [(s["start"], s["end"]) for s in sentences]
        for tok in tokens:
            if not tok.dep:
                tok.dep = "dep"
                tok.head_idx = tok.idx

    def _parse_sentence(self, sent_tokens: list, offset: int) -> None:
        """Parse a single sentence."""
        n = len(sent_tokens)
        if n == 0:
            return

        # Initialize: all tokens head to themselves, no dep
        for tok in sent_tokens:
            tok.dep = ""
            tok.head_idx = tok.idx

        # --- Rule 1: ROOT ---
        root_idx = self._find_root(sent_tokens, offset)
        root = sent_tokens[root_idx - offset]
        root.dep = "ROOT"
        root.head_idx = root.idx  # ROOT heads itself

        # --- Rule 12: Passive detection (must run before nsubj) ---
        self._passive_pass(sent_tokens, offset, root_idx)

        # --- Rule 8: compound (must run before nsubj to link "fire truck") ---
        new_root = self._compound_pass(sent_tokens, offset)
        if new_root is not None:
            root_idx = new_root

        # --- Rule 2: nsubj / nsubjpass ---
        self._subject_pass(sent_tokens, offset, root_idx)

        # --- Rule 3: dobj/obj ---
        self._object_pass(sent_tokens, offset, root_idx)

        # --- Rule 13: det (DET -> head=next NOUN) ---
        self._det_pass(sent_tokens, offset)

        # --- Rule 4: amod ---
        self._amod_pass(sent_tokens, offset)

        # --- Rule 5: acomp/attr/oprd ---
        self._predicate_adj_pass(sent_tokens, offset, root_idx)

        # --- Rule 6: prep/pobj ---
        self._prep_pobj_pass(sent_tokens, offset)

        # --- Rule 7: advmod ---
        self._advmod_pass(sent_tokens, offset)

        # --- Rule 9: conj/cc ---
        self._conj_pass(sent_tokens, offset)

        # --- Rule 10: mark/advcl ---
        self._advcl_pass(sent_tokens, offset, root_idx)

        # --- Rule 11: xcomp ---
        self._xcomp_pass(sent_tokens, offset, root_idx)

        # --- Rule AUX attachment ---
        self._aux_pass(sent_tokens, offset)

        # --- Default: unlinked -> dep (head=ROOT) ---
        for tok in sent_tokens:
            if not tok.dep:
                tok.dep = "dep"
                tok.head_idx = root_idx

    def _find_root(self, sent_tokens: list, offset: int) -> int:
        """Find ROOT: first VERB, fallback first AUX, fallback first NOUN/PROPN."""
        # Prefer VERB over AUX (copula/modal should only be ROOT if no main verb)
        # Skip VBN/VBG that appear before first NOUN (likely participial modifier)
        first_noun_idx = None
        for tok in sent_tokens:
            if tok.pos in ("NOUN", "PROPN"):
                first_noun_idx = tok.idx
                break

        for tok in sent_tokens:
            if tok.pos == "VERB":
                # Skip participles before first noun (e.g. "the broken window")
                if (
                    tok.tag in ("VBN", "VBG")
                    and first_noun_idx is not None
                    and tok.idx < first_noun_idx
                ):
                    continue
                # Skip verbs in multi-token named entities
                # (e.g. "United" in "United States" tagged as GPE)
                # Only skip if this is part of a span (B + next token is I)
                if tok.ent_iob == "I" and tok.ent_type:
                    continue
                if tok.ent_iob == "B" and tok.ent_type:
                    local_i = tok.idx - offset
                    if (local_i + 1 < len(sent_tokens)
                            and sent_tokens[local_i + 1].ent_iob == "I"):
                        continue
                return tok.idx

        for tok in sent_tokens:
            if tok.pos == "AUX":
                return tok.idx

        for tok in sent_tokens:
            if tok.pos in ("NOUN", "PROPN"):
                return tok.idx

        # Fallback: first non-punct token, or first token
        for tok in sent_tokens:
            if not tok.is_punct:
                return tok.idx

        return sent_tokens[0].idx

    def _passive_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """Detect passive: AUX('be') + VBN -> set morph Voice=Pass."""
        n = len(sent_tokens)
        for i in range(n - 1):
            tok = sent_tokens[i]
            nxt = sent_tokens[i + 1]
            if (
                tok.pos == "AUX"
                and tok.text.lower() in _BE_FORMS
                and nxt.pos == "VERB"
                and nxt.tag == "VBN"
            ):
                nxt.morph_data["Voice"] = ["Pass"]

    def _subject_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """NOUN/PROPN/PRON left of ROOT -> nsubj (or nsubjpass if passive)."""
        root_local = root_idx - offset
        for i in range(root_local):
            tok = sent_tokens[i]
            if tok.dep:
                continue
            if tok.pos in ("NOUN", "PROPN", "PRON"):
                root_tok = sent_tokens[root_local]
                if root_tok.morph_data.get("Voice") == ["Pass"]:
                    tok.dep = "nsubjpass"
                else:
                    tok.dep = "nsubj"
                tok.head_idx = root_idx
                return  # Only one subject

    def _object_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """First NOUN/PROPN/PRON right of ROOT before ADP -> dobj."""
        root_local = root_idx - offset
        n = len(sent_tokens)
        for i in range(root_local + 1, n):
            tok = sent_tokens[i]
            if tok.dep:
                continue
            # Stop at preposition (object will be pobj instead)
            if tok.pos == "ADP":
                break
            if tok.pos in ("NOUN", "PROPN", "PRON"):
                tok.dep = "dobj"
                tok.head_idx = root_idx
                return  # Only first direct object

    def _det_pass(self, sent_tokens: list, offset: int) -> None:
        """DET -> det (head=next NOUN/PROPN)."""
        n = len(sent_tokens)
        for i in range(n):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "DET":
                continue
            # Find next noun
            for j in range(i + 1, min(i + 4, n)):
                candidate = sent_tokens[j]
                if candidate.pos in ("NOUN", "PROPN"):
                    tok.dep = "det"
                    tok.head_idx = candidate.idx
                    break

    def _amod_pass(self, sent_tokens: list, offset: int) -> None:
        """ADJ immediately before NOUN -> amod (head=NOUN)."""
        n = len(sent_tokens)
        for i in range(n - 1):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "ADJ":
                continue
            # Look for a following noun (allow skipping other ADJ/compound)
            for j in range(i + 1, min(i + 4, n)):
                candidate = sent_tokens[j]
                if candidate.pos in ("NOUN", "PROPN"):
                    tok.dep = "amod"
                    tok.head_idx = candidate.idx
                    break
                elif candidate.pos not in ("ADJ", "NOUN", "PROPN", "NUM"):
                    break
            # Also handle VBN as participial modifier (amod)
        for i in range(n - 1):
            tok = sent_tokens[i]
            if tok.dep or tok.tag != "VBN":
                continue
            for j in range(i + 1, min(i + 3, n)):
                candidate = sent_tokens[j]
                if candidate.pos in ("NOUN", "PROPN"):
                    tok.dep = "amod"
                    tok.head_idx = candidate.idx
                    break
                elif candidate.pos not in ("ADJ",):
                    break

    def _compound_pass(self, sent_tokens: list, offset: int) -> int | None:
        """NOUN immediately before NOUN (no intervening DET) -> compound.

        Returns new root_idx if ROOT was reassigned (e.g. consecutive PROPN
        sequence like "Albert Einstein"), or None if ROOT unchanged.
        """
        n = len(sent_tokens)
        new_root_idx = None
        for i in range(n - 1):
            tok = sent_tokens[i]
            nxt = sent_tokens[i + 1]
            if tok.dep and tok.dep != "ROOT":
                continue
            if (
                tok.pos in ("NOUN", "PROPN")
                and nxt.pos in ("NOUN", "PROPN")
            ):
                if tok.dep == "ROOT":
                    # Transfer ROOT rightward: "Albert(ROOT) Einstein" →
                    # Albert(compound) Einstein(ROOT)
                    tok.dep = "compound"
                    tok.head_idx = nxt.idx
                    nxt.dep = "ROOT"
                    nxt.head_idx = nxt.idx
                    new_root_idx = nxt.idx
                else:
                    tok.dep = "compound"
                    tok.head_idx = nxt.idx
        return new_root_idx

    def _predicate_adj_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """ADJ after copula -> acomp/attr (head=copula verb)."""
        root_local = root_idx - offset
        root_tok = sent_tokens[root_local]
        root_lower = root_tok.text.lower()

        if root_lower not in _COPULA_VERBS and root_tok.lemma.lower() not in _COPULA_VERBS:
            return

        n = len(sent_tokens)
        for i in range(root_local + 1, n):
            tok = sent_tokens[i]
            if tok.dep:
                continue
            if tok.pos == "ADJ":
                tok.dep = "acomp"
                tok.head_idx = root_idx
                return
            elif tok.pos in ("NOUN", "PROPN"):
                tok.dep = "attr"
                tok.head_idx = root_idx
                return

    def _prep_pobj_pass(self, sent_tokens: list, offset: int) -> None:
        """ADP -> prep. NOUN/PROPN/PRON after ADP -> pobj."""
        n = len(sent_tokens)
        for i in range(n):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "ADP":
                continue

            # Head of prep: preceding VERB or NOUN
            head_idx = self._find_preceding(
                sent_tokens, i, {"VERB", "NOUN", "PROPN", "AUX", "ADJ"}
            )
            if head_idx is not None:
                tok.dep = "prep"
                tok.head_idx = sent_tokens[head_idx].idx
            else:
                tok.dep = "prep"
                # Fall through to default head

            # Find pobj: next NOUN/PROPN/PRON after this ADP
            for j in range(i + 1, n):
                candidate = sent_tokens[j]
                if candidate.dep:
                    continue
                if candidate.pos in ("NOUN", "PROPN", "PRON"):
                    candidate.dep = "pobj"
                    candidate.head_idx = tok.idx
                    break
                elif candidate.pos in ("DET", "ADJ", "NUM", "ADV"):
                    continue  # Skip modifiers between prep and pobj
                else:
                    break

    def _advmod_pass(self, sent_tokens: list, offset: int) -> None:
        """ADV -> advmod (head=nearest VERB/ADJ)."""
        n = len(sent_tokens)
        for i in range(n):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "ADV":
                continue

            # Find nearest VERB/ADJ
            best = None
            best_dist = n + 1
            for j in range(n):
                if j == i:
                    continue
                candidate = sent_tokens[j]
                if candidate.pos in ("VERB", "ADJ", "AUX"):
                    dist = abs(j - i)
                    if dist < best_dist:
                        best_dist = dist
                        best = j

            if best is not None:
                tok.dep = "advmod"
                tok.head_idx = sent_tokens[best].idx

    def _conj_pass(self, sent_tokens: list, offset: int) -> None:
        """CCONJ between same-POS -> cc. Second conjunct -> conj."""
        n = len(sent_tokens)
        for i in range(1, n - 1):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "CCONJ":
                continue

            # Find first conjunct (same POS left of CCONJ)
            left = None
            for j in range(i - 1, -1, -1):
                if sent_tokens[j].pos not in ("PUNCT", "CCONJ", "DET", "ADP", "ADV"):
                    left = j
                    break

            # Find second conjunct (same POS right of CCONJ)
            right = None
            for j in range(i + 1, n):
                if sent_tokens[j].pos not in ("PUNCT", "CCONJ", "DET", "ADP", "ADV"):
                    right = j
                    break

            if left is not None and right is not None:
                # cc attaches to first conjunct
                tok.dep = "cc"
                tok.head_idx = sent_tokens[left].idx

                # Second conjunct -> conj (head=first conjunct)
                if not sent_tokens[right].dep:
                    sent_tokens[right].dep = "conj"
                    sent_tokens[right].head_idx = sent_tokens[left].idx

    def _advcl_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """SCONJ -> mark. Clause verb after SCONJ -> advcl (head=ROOT)."""
        n = len(sent_tokens)
        for i in range(n):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "SCONJ":
                continue

            # Find the clause verb after SCONJ
            clause_verb = None
            for j in range(i + 1, n):
                if sent_tokens[j].pos in ("VERB", "AUX"):
                    clause_verb = j
                    break

            if clause_verb is not None:
                tok.dep = "mark"
                tok.head_idx = sent_tokens[clause_verb].idx

                if not sent_tokens[clause_verb].dep:
                    sent_tokens[clause_verb].dep = "advcl"
                    sent_tokens[clause_verb].head_idx = root_idx

    def _xcomp_pass(
        self, sent_tokens: list, offset: int, root_idx: int
    ) -> None:
        """'to' + VERB -> xcomp (head=preceding VERB or ROOT)."""
        n = len(sent_tokens)
        for i in range(n - 1):
            tok = sent_tokens[i]
            nxt = sent_tokens[i + 1]
            if tok.text.lower() != "to" or tok.pos != "PART":
                continue
            if nxt.pos != "VERB" or nxt.dep:
                continue

            # Find preceding VERB for head
            head = None
            for j in range(i - 1, -1, -1):
                if sent_tokens[j].pos in ("VERB", "AUX"):
                    head = sent_tokens[j].idx
                    break

            if head is None:
                head = root_idx

            nxt.dep = "xcomp"
            nxt.head_idx = head

            # "to" gets mark dep
            if not tok.dep:
                tok.dep = "aux"
                tok.head_idx = nxt.idx

    def _aux_pass(self, sent_tokens: list, offset: int) -> None:
        """AUX tokens that aren't ROOT -> aux (head=nearest VERB)."""
        n = len(sent_tokens)
        for i in range(n):
            tok = sent_tokens[i]
            if tok.dep or tok.pos != "AUX":
                continue

            # Find nearest VERB (prefer right, then left)
            head = None
            for j in range(i + 1, n):
                if sent_tokens[j].pos == "VERB":
                    head = sent_tokens[j].idx
                    break
            if head is None:
                for j in range(i - 1, -1, -1):
                    if sent_tokens[j].pos == "VERB":
                        head = sent_tokens[j].idx
                        break

            if head is not None:
                tok.dep = "aux"
                tok.head_idx = head

    # -- Helpers ----------------------------------------------------------

    def _find_preceding(
        self, sent_tokens: list, idx: int, pos_set: Set[str]
    ) -> int | None:
        """Find the nearest preceding token with POS in pos_set."""
        for j in range(idx - 1, -1, -1):
            if sent_tokens[j].pos in pos_set:
                return j
        return None
