"""Which text of a parsed record the gaius head embeds.

The head never parses: it reads the shared ParsedDocument (rule backend) that
the NLP stage put in ``context["spacy_doc"]`` and takes its sentence spans.

- The DCP adapter prepends the record's title as a fragment
  (``Albert_Einstein.``). It is the record's name, so it is kept (with
  spaces) and leads the record's text; a list page's abstract is often a bare
  list of names, and its title (``List of Polydor Records artists``) is then
  the only text that says what the record is.
- A real sentence has at least MIN_WORDS words and is not an editorial
  pointer ("may refer to", "see also", ...).
- The record is typed from its title and its lead -- the first real sentence,
  the one that says what the subject is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

MIN_WORDS = 3

POINTER_RE = re.compile(
    r"^\W*(see also\b|this article is about\b|for other uses\b|not to be confused with\b|"
    r".{0,80}\b(may|can|could) (also )?refer to\b)",
    re.I)

_NO_SPACE_BEFORE = re.compile(r"\s+([.,;:!?%)\]}])")
_NO_SPACE_AFTER = re.compile(r"([(\[{$])\s+")
_CLITIC = re.compile(r"\s+('s|'re|'ve|'ll|'d|'m|n't)\b", re.I)


def detokenize(words: Sequence[str]) -> str:
    """Rule-backend tokens back to running text ("physicist ." -> "physicist.")."""
    s = " ".join(w for w in words if w)
    s = _NO_SPACE_BEFORE.sub(r"\1", s)
    s = _NO_SPACE_AFTER.sub(r"\1", s)
    s = _CLITIC.sub(r"\1", s)
    return s.strip()


def _words(tokens) -> List[str]:
    return [t.text for t in tokens if not getattr(t, "is_punct", False) and (t.text or "").strip()]


def title_of(tokens) -> Optional[str]:
    """The DCP-prepended title, if this sentence is that fragment: its first
    token is the underscore-joined page name (running text never produces
    such a token)."""
    toks = list(tokens)
    if not toks:
        return None
    first = toks[0].text or ""
    if "_" not in first or " " in first or len(_words(toks)) > 1:
        return None
    return first.replace("_", " ").strip(" .")


_BE = "be"
_PHRASE_POS = {"DET", "ADJ", "NOUN", "PROPN", "ADV", "NUM", "VERB", "PART"}
_NOUN_POS = {"NOUN", "PROPN"}
MAX_PHRASE = 8
# Nouns that name a kind of kind: in "a species of moth" the thing is a moth.
LIGHT_NOUNS = frozenset({"species", "subspecies", "genus", "kind", "type", "sort", "form",
                         "variety", "breed", "one"})


@dataclass(frozen=True)
class Phrase:
    text: str    # the phrase as running text ("a species of moth")
    head: str    # its head noun ("moth")


def _np(toks, start: int, limit: int):
    out = []
    for t in toks[start:start + limit]:
        if t.pos_ not in _PHRASE_POS:
            break
        out.append(t)
    nouns = [j for j, t in enumerate(out) if t.pos_ in _NOUN_POS]
    return out[:nouns[-1] + 1] if nouns else []


def defining_phrase(tokens) -> Optional[Phrase]:
    """What a lead sentence says its subject IS, from the parse.

    The noun phrase after the first form of "be" ("was a German-born
    theoretical *physicist*"), cut after its last noun and continued through
    one "of" ("a species of *moth*"; a LIGHT_NOUNS head hands the head over to
    the complement), at most MAX_PHRASE tokens; else the subject phrase before
    the first verb ("The *Battle* of Waterloo was fought"). None when neither
    has a noun.
    """
    toks = [t for t in tokens if not getattr(t, "is_punct", False)]
    lemmas = [(t.lemma_ or t.text or "").lower() for t in toks]
    for i, lem in enumerate(lemmas):
        if lem != _BE:
            continue
        np_ = _np(toks, i + 1, MAX_PHRASE)
        if not np_:
            break
        head = np_[-1]
        end = i + 1 + len(np_)
        if end < len(toks) and lemmas[end] == "of" and len(np_) < MAX_PHRASE - 1:
            comp = _np(toks, end + 1, MAX_PHRASE - len(np_) - 1)
            if comp:
                if (head.lemma_ or head.text or "").lower() in LIGHT_NOUNS:
                    head = comp[-1]
                np_ = np_ + [toks[end]] + comp
        return Phrase(detokenize([t.text for t in np_]), head.text)
    subj = []
    for t, lem in zip(toks, lemmas):
        if lem == _BE or t.pos_ in {"VERB", "AUX"}:
            break
        subj.append(t)
    subj = subj[:MAX_PHRASE]
    nouns = [t for t in subj if t.pos_ == "NOUN"]
    if not nouns:
        return None
    first_adp = next((j for j, t in enumerate(subj) if t.pos_ == "ADP"), len(subj))
    before = [t for t in subj[:first_adp] if t.pos_ == "NOUN"]
    return Phrase(detokenize([t.text for t in subj]), (before or nouns)[-1].text)


def is_real(tokens) -> bool:
    toks = list(tokens)
    if len(_words(toks)) < MIN_WORDS:
        return False
    return not POINTER_RE.match(detokenize([t.text for t in toks[:30]]))


@dataclass
class RecordText:
    title: Optional[str] = None
    # (sentence index in doc.sents, text) of every real sentence, in order
    sentences: List[Tuple[int, str]] = field(default_factory=list)
    # sentence index -> defining_phrase of that sentence (None: no phrase)
    phrases: Dict[int, Optional[Phrase]] = field(default_factory=dict)

    @property
    def lead(self) -> Optional[Tuple[int, str]]:
        return self.sentences[0] if self.sentences else None

    def record_query(self) -> str:
        """Title and lead ("" if the record has neither)."""
        lead = self.lead[1] if self.lead else ""
        if not self.title:
            return lead
        if lead.lower().startswith(self.title.lower()):
            return lead
        return f"{self.title}. {lead}".strip() if lead else self.title

    def query_text(self) -> str:
        """The text the record is typed from: the lead's defining phrase,
        else the title and lead ("" when it has neither)."""
        ph = self.phrases.get(self.lead[0]) if self.lead else None
        return ph.text if ph else self.record_query()


def record_text(doc) -> RecordText:
    out = RecordText()
    for i, span in enumerate(doc.sents):
        toks = list(span)
        if i == 0:
            t = title_of(toks)
            if t:
                out.title = t
                continue
        if is_real(toks):
            out.sentences.append((i, detokenize([t.text for t in toks])))
            out.phrases[i] = defining_phrase(toks)
    return out
