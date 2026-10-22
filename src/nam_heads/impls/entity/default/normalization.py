# src/encoders/heads/entity/normalization.py

"""
Entity mention detection and normalization.

Two deterministic passes over one ParsedDocument (rule NLP):

1. Name spans (``name_spans``): runs of capitalised tokens, joined by name
   connectors ("Battle of Lostwithiel", "Leonardo da Vinci", "Bunnies &
   Burrows"), abbreviation dots ("Great Wyrley F.C.", "U.S. Army") and one
   trailing number or regnal numeral ("Apollo 15", "Charles I"). Read from
   capitalisation alone, so a name survives a POS tag the tagger got wrong:
   it looks a word up in its dictionary BEFORE it checks case
   (nlp/rule/pos_tagger.py tier 3 before tier 4), so "Young" comes out
   ADJ and "United" VERB, and the noun chunker cuts "Brigham Young" and
   "United States Mint" apart.
2. Noun chunks, cleaned: articles, demonstratives and punctuation skipped,
   cut at the first preposition, the "s" a curly apostrophe leaves behind
   ("Earth ’ s exploration") dropped.

A chunk inside a name span gives way to the span; a chunk that is a name span
plus lower-case words ("entrepreneur Yves Rocher") is replaced by the span.

NO entity typing
NO resolution
"""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
import re
import unicodedata


@dataclass
class EntityCandidate:
    text: str
    normalized: str
    confidence: float
    pos_pattern: str = "UNK"
    token_count: int = 1
    is_propn: bool = False
    has_digit: bool = False
    has_hyphen: bool = False
    # Rule-NLP NER labels over the mention's content tokens ("O" = none);
    # None when the candidate did not come from the parse (temporal_filter).
    ner_labels: Optional[FrozenSet[str]] = None
    # A name as written: a name span, the record title, or a mention whose
    # content words are all capitalised with at least one of them not opening
    # a sentence ("the Moon", "LDS Church"). A name is never a codebook
    # rejection or a compound-filter drop, and a name ending in a capitalised
    # word keeps its form in the entity id ("Angolan Armed Forces" is not
    # singularised).
    is_name: bool = False
    # First token of the mention in the parse (-1: not from the parse). Only
    # orders the candidates of one record.
    start: int = -1
    # The chunk introduced the mention with an article or demonstrative
    # ("the decisive engagement", "a unique blend"). Not part of the POS
    # pattern; the compound filter reads it for lower-case descriptions.
    has_article: bool = False


# Closed-class / stable sets
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those"}
_PREPOSITIONS = {"of", "on", "by", "for", "with", "to", "in", "at", "from"}

_LOW_SALIENCE_HEADS = {
    "cause", "reason", "issue", "problem", "effect", "result", "event", "thing"
}

_CONTEXTUAL_NOUNS = {"line", "floor", "level", "area", "section"}

# Apostrophe tokens. The rule tokenizer splits "Earth’s" into
# "Earth" "’" "s"; the stray "s" opened the chunk "s exploration".
_APOSTROPHES = frozenset({"’", "'", "‘"})

# Lower-case words that join capitalised words inside one name.
_NAME_CONNECTORS = frozenset({
    "of", "de", "da", "del", "di", "du", "la", "le", "van", "von", "der", "&", "y",
})
# Capitalised only because they open a sentence: never the first word of a name.
_SENTENCE_FUNCTION_WORDS = frozenset({
    "The", "A", "An", "In", "On", "At", "By", "For", "From", "Of", "To", "With",
    "And", "Or", "But", "As", "It", "Its", "His", "Her", "Their", "This", "That",
    "These", "Those", "He", "She", "They", "We", "You", "I", "If", "When", "While",
    "After", "Before", "During", "Despite", "Although", "Over", "Under", "Since",
    "Through", "Throughout", "Today", "Interestingly", "Notably", "Historically",
    "Meanwhile", "Usually", "Sometimes", "Once", "Not", "Some", "Many", "Most",
    "Each", "Every", "Both", "Such", "There", "Here", "What", "Who", "Why", "How",
    "Where", "Which", "Unfortunately", "Simply", "Together", "Later",
})
# Tags of a sentence-initial word that cannot open a name.
_NAME_START_EXCLUDED_POS = frozenset({
    "VERB", "ADV", "ADP", "SCONJ", "PRON", "DET", "AUX", "CCONJ", "PART", "INTJ",
})
# Number words the tagger marks NUM that do open names: "First English Civil War".
_ORDINAL_NAME_WORDS = frozenset({
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth",
    "Ninth", "Tenth", "Two", "Three", "Four", "Five",
})
_TEMPORAL_ENT = frozenset({"DATE", "TIME"})
_ROMAN_TOKEN = re.compile(
    r"^(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII|XIII|XIV|XV|XVI|XVII|XVIII|XIX|XX)$")
_YEAR_TOKEN = re.compile(r"^(?:1\d{3}|20\d{2})s?$")
_INITIALS = re.compile(r"^(?:[A-Z]\.){2,}$")


# ------------------------
# Public API
# ------------------------

def extract_candidates(text: str, debug: bool = False, spacy_doc=None,
                       skip_tokens: int = 0) -> List[EntityCandidate]:
    """Entity candidates of a parsed text, in text order.

    ``skip_tokens``: candidates starting before this token are left out (the
    record title the ingest adapter puts in front of the text; the head adds
    the title itself as one candidate)."""
    if spacy_doc is not None and hasattr(spacy_doc, "sentences"):
        return _candidates_from_chunks(spacy_doc, spacy_doc.noun_chunks, debug=debug,
                                       skip_tokens=skip_tokens)

    return _regex_fallback(text)


# ------------------------
# Name spans
# ------------------------

def _sentence_starts(doc) -> Set[int]:
    return {s.start_token for s in getattr(doc, "sentences", ()) or ()}


def _is_cap(tok) -> bool:
    s = tok.text
    return (bool(s) and s[0].isupper() and not tok.is_punct
            and s not in _APOSTROPHES)


def _temporal(tok) -> bool:
    return (getattr(tok, "ent_type_", "") or "") in _TEMPORAL_ENT


def _opens_name(tok, sent_starts: Set[int]) -> bool:
    """Can *tok* be the first word of a name span?"""
    if not _is_cap(tok) or tok.pos_ == "PRON" or tok.text == "I" or _temporal(tok):
        return False
    if tok.pos_ == "NUM" and tok.text not in _ORDINAL_NAME_WORDS:
        return False
    if tok.i in sent_starts and (tok.text in _SENTENCE_FUNCTION_WORDS
                                 or tok.pos_ in _NAME_START_EXCLUDED_POS):
        return False
    return True


def _continues_name(tok, sent_starts: Set[int], after_abbrev: bool) -> bool:
    """Can capitalised *tok* extend the name span it follows?"""
    if not _is_cap(tok) or tok.pos_ == "PRON" or _temporal(tok):
        return False
    if after_abbrev and tok.text in _SENTENCE_FUNCTION_WORDS:
        return False                  # "Esteghlal F.C. In 2005 ..."
    if tok.i in sent_starts:
        # Only an abbreviation dot ("F.C.", "St.") lets a name run on past
        # what the sentence splitter took for a sentence end.
        if not after_abbrev:
            return False
        if tok.text in _SENTENCE_FUNCTION_WORDS or tok.pos_ in _NAME_START_EXCLUDED_POS:
            return False
    return True


def _abbreviation(s: str) -> bool:
    """A token a following period belongs to: an initial ("F", "C", "U") or
    a Title-case short form ("St", "Jr", "Dr"). Not "II" or "UK": the period
    after them ends the sentence."""
    if len(s) == 1:
        return s.isupper()
    return len(s) == 2 and s[0].isupper() and s[1].islower()


def _trailing_number(toks, j: int) -> bool:
    """Token j ends a name as its number: "Apollo 15", "Charles I"."""
    if j >= len(toks):
        return False
    s = toks[j].text
    if s.isdigit():
        return len(s) <= 4 and not _YEAR_TOKEN.match(s)
    if _ROMAN_TOKEN.match(s):
        # "I" is the pronoun when a verb follows ("... Smith I was") or
        # precedes ("I Think I Love You").
        if s == "I" and ((j + 1 < len(toks) and toks[j + 1].pos_ in ("VERB", "AUX")
                          and toks[j + 1].text not in ("'s", "’s"))
                         or toks[j - 1].pos_ == "VERB"):
            return False
        return True
    return False


def name_spans(doc) -> List[Tuple[int, int, str]]:
    """Name spans of a parsed document: (start, end, compact text).

    ``end`` is exclusive. The compact text joins abbreviation dots to their
    letter ("Great Wyrley F.C."); a span's tokens are doc[start:end]. A span
    has at least two tokens, or is a run of initials ("U.S."). Deterministic:
    a function of the token texts, tags, NER labels and sentence starts.
    """
    toks = list(doc)
    n = len(toks)
    sent_starts = _sentence_starts(doc)
    out: List[Tuple[int, int, str]] = []
    i = 0
    while i < n:
        t = toks[i]
        if not _opens_name(t, sent_starts):
            i += 1
            continue
        # A sentence-initial common word joined to a name only by a connector
        # ("Residents of Brussels") is not part of it.
        if (t.i in sent_starts and t.pos_ != "PROPN" and i + 1 < n
                and toks[i + 1].text.lower() in _NAME_CONNECTORS):
            i += 1
            continue
        j = i + 1
        parts = [t.text]
        after_abbrev = False
        while j < n:
            u = toks[j]
            prev = toks[j - 1].text
            if u.text == ".":
                if _abbreviation(prev) and j + 1 < n and _is_cap(toks[j + 1]):
                    parts[-1] += "."
                    j += 1
                    after_abbrev = True
                    continue
                if len(prev) == 1 and prev.isupper() and "." in parts[-1]:
                    parts[-1] += "."          # "F.C." closing a name
                    j += 1
                break
            if _continues_name(u, sent_starts, after_abbrev):
                if after_abbrev and len(u.text) <= 2 and parts[-1].endswith(".") \
                        and len(parts[-1]) <= 3:
                    parts[-1] += u.text       # "F." + "C" -> "F.C"
                else:
                    parts.append(u.text)
                after_abbrev = False
                j += 1
                continue
            low = u.text.lower()
            if low in _NAME_CONNECTORS and j + 1 < n:
                k = j + 1
                if toks[k].text.lower() == "the" and k + 1 < n:
                    k += 1
                nxt = toks[k]
                if _is_cap(nxt) and nxt.pos_ != "PRON" and not _temporal(nxt) \
                        and nxt.i not in sent_starts:
                    parts.extend(x.text for x in toks[j:k])
                    j = k
                    continue
            if low == "the" and j + 1 < n and toks[j - 1].pos_ == "PROPN":
                nxt = toks[j + 1]
                # "Edward the Confessor", "Alexander the Great"
                if _is_cap(nxt) and nxt.pos_ in ("PROPN", "NOUN", "ADJ") \
                        and nxt.i not in sent_starts:
                    parts.append("the")
                    j += 1
                    continue
            break
        if _trailing_number(toks, j):
            parts.append(toks[j].text)
            j += 1
        while len(parts) > 1 and parts[-1].lower() in _NAME_CONNECTORS | {"the"}:
            parts.pop()
            j -= 1
        if j - i >= 2 and (len(parts) >= 2 or _INITIALS.match(parts[0])):
            out.append((i, j, " ".join(parts)))
            i = j
        else:
            i += 1
    return out


# Lower-case particles inside a name ("Académie des Beaux-Arts", "Muhammad
# bin Tughlaq"): a word leading into a name is never one of these.
_NAME_PARTICLES = _NAME_CONNECTORS | frozenset({
    "des", "dos", "das", "do", "della", "delle", "degli", "dei", "den", "ten", "ter",
    "het", "zu", "zum", "zur", "bin", "ibn", "bint", "ben", "al", "el", "af", "av",
})


def _leads_into_name(lead) -> bool:
    """``lead`` (the chunk's tokens before a name) ends in a lower-case word
    that describes the name rather than belonging to it ("southern Sweden",
    "genus Huia", "French composer Pierre-Alexandre Monsigny"). Not a name
    particle ("van Gogh", "Muhammad bin Tughlaq"), and after a capitalised
    word only a noun ("English monk Bede"; "New York" leads into nothing)."""
    if not lead:
        return False
    last = lead[-1]
    w = last.text
    if not w or w[:1].isupper() or w.lower() in _NAME_PARTICLES:
        return False
    if len(lead) > 1 and lead[-2].text[:1].isupper():
        return last.pos_ == "NOUN"
    return True


def _all_capitalised(toks, sent_starts: Set[int]) -> bool:
    """Every content word capitalised (connectors aside) and at least one of
    them not opening a sentence: a name as written, not a sentence-initial
    capital."""
    words = [t for t in toks if not t.is_punct and t.text.lower() not in _NAME_CONNECTORS
             and t.text.lower() != "the" and not t.text.isdigit()]
    if not words or not all(_is_cap(t) for t in words):
        return False
    return any(t.i not in sent_starts for t in words)


def _span_candidate(doc, start: int, end: int, compact: str,
                    sent_starts: Set[int]) -> Optional[EntityCandidate]:
    toks = list(doc[start:end])
    # The surface keeps every token, so it matches the parse token by token
    # ("Great Wyrley F . C ."); the id is read from the compact form.
    text = " ".join(t.text for t in toks)
    normalized = normalize_mention(compact)
    if not normalized:
        return None
    pos = "_".join("PROPN" if _is_cap(t) else t.pos_ for t in toks if not t.is_punct)
    return EntityCandidate(
        text=text, normalized=normalized, confidence=0.6,
        pos_pattern=pos, token_count=sum(1 for t in toks if not t.is_punct),
        is_propn=True, has_digit=any(c.isdigit() for c in compact),
        has_hyphen="-" in compact,
        ner_labels=frozenset((getattr(t, "ent_type_", "") or "O")
                             for t in toks if not t.is_punct) or None,
        is_name=True, start=start,
    )


def _candidates_from_chunks(doc, chunks, debug: bool = False,
                            skip_tokens: int = 0) -> List[EntityCandidate]:
    """Build EntityCandidates from name spans and pre-extracted noun chunks."""
    sent_starts = _sentence_starts(doc)
    spans = name_spans(doc)
    cover: Dict[int, int] = {}
    for si, (s, e, _c) in enumerate(spans):
        for k in range(s, e):
            cover[k] = si
    out: List[EntityCandidate] = []
    seen: Set[str] = set()
    emitted: Set[int] = set()

    def add(cand: Optional[EntityCandidate]) -> bool:
        if cand is None or not cand.normalized or cand.normalized in seen:
            return False
        if cand.start >= 0 and cand.start < skip_tokens:
            return False
        seen.add(cand.normalized)
        out.append(cand)
        return True

    def emit_span(si: int) -> None:
        if si in emitted:
            return
        emitted.add(si)
        s, e, compact = spans[si]
        add(_span_candidate(doc, s, e, compact, sent_starts))

    for chunk in chunks:
        root_token = doc[chunk.root_idx]
        if root_token.pos_ == "PRON":
            continue

        if root_token.lemma_.lower() in _LOW_SALIENCE_HEADS:
            continue

        clean = _clean_span(doc, chunk.start_token, chunk.end_token)
        if not clean:
            continue

        idx = [t.i for t in clean]
        span_ids = sorted({cover[k] for k in idx if k in cover})
        if span_ids:
            if len(span_ids) == 1 and all(k in cover for k in idx):
                si = span_ids[0]
                s, e, _c = spans[si]
                emit_span(si)
                if idx[0] == s and idx[-1] == e - 1:
                    continue                    # the chunk IS the name
                # Strictly inside a name: only a proper noun right after "of"
                # that runs to the end of the name stands on its own
                # ("Lostwithiel" in "Battle of Lostwithiel"; not "Joseph W"
                # in "The Life of Joseph W. McVey").
                p1 = doc[idx[0] - 1].text.lower() if idx[0] > 0 else ""
                p2 = doc[idx[0] - 2].text.lower() if idx[0] > 1 else ""
                last = max(k for k in range(s, e) if not doc[k].is_punct)
                if not (root_token.pos_ == "PROPN" and (p1 == "of" or (p1 == "the" and p2 == "of"))
                        and idx[-1] == last):
                    continue
            else:
                for si in span_ids:
                    emit_span(si)
                extra = [doc[k] for k in idx if k not in cover]
                if all(not x.text[:1].isupper() for x in extra):
                    # a name plus lower-case words ("entrepreneur Yves Rocher",
                    # "Andhra Pradesh stretches"): the name alone
                    continue
                first = spans[span_ids[0]][0]
                lead = [doc[k] for k in idx if k < first]
                if len(lead) == len(extra) and lead[-1].i == first - 1 and _leads_into_name(lead):
                    # words leading into the name, the last one lower case
                    # ("French composer Pierre-Alexandre Monsigny"): the name alone
                    continue

        raw_tokens = [t.text for t in clean]
        raw = " ".join(raw_tokens).strip()
        if len(raw) < 2:
            continue

        normalized = normalize_mention(raw)
        if normalized in seen:
            continue

        score = _score_candidate(
            raw=raw, head=root_token, doc=doc,
            start=chunk.start_token, end=chunk.end_token,
            token_count=len(clean),
        )
        if score < 0.15:
            if debug:
                print(f"  reject '{raw}' score={score:.2f}")
            continue

        if debug:
            print(f"  chunk '{raw}' -> '{normalized}' score={score:.2f}")

        # Chunk metadata for codebook classification, read from the content
        # tokens: "the town" is the noun "town" (NOUN, one token), not a
        # two-token DET_NOUN compound.
        pos_pattern = "_".join(t.pos_ for t in clean)
        token_count = len(clean)
        is_propn = root_token.pos_ == "PROPN"
        has_digit = any(c.isdigit() for c in raw)
        has_hyphen = "-" in raw

        # The proper-noun run that ends the chunk, if words lead into it
        # ("president Abraham Lincoln", "southern Sweden", "English monk
        # Bede"): the name alone is the candidate, as for a name span. Never
        # after a capitalised word or a name connector: "New York" does not
        # yield "York", nor "van Gogh" "Gogh".
        propn_start = None
        if token_count > 1:
            for ti, t in enumerate(clean):
                if t.pos_ == "PROPN" and propn_start is None:
                    propn_start = ti
                elif t.pos_ != "PROPN" and propn_start is not None:
                    propn_start = None  # non-contiguous, reset
        name_only = bool(propn_start) and _leads_into_name(clean[:propn_start])

        if not name_only:
            add(EntityCandidate(
                text=raw, normalized=normalized, confidence=score,
                pos_pattern=pos_pattern, token_count=token_count,
                is_propn=is_propn, has_digit=has_digit, has_hyphen=has_hyphen,
                ner_labels=_span_ner_labels(doc, chunk.start_token, chunk.end_token),
                is_name=_all_capitalised(clean, sent_starts), start=clean[0].i,
                has_article=_has_article(doc, chunk.start_token, chunk.end_token),
            ))

        if name_only:
            tail = clean[propn_start:]
            propn_raw = " ".join(t.text for t in tail)
            propn_norm = normalize_mention(propn_raw)
            if propn_norm and len(propn_raw) >= 2:
                propn_score = min(score + 0.1, 0.9)
                if add(EntityCandidate(
                    text=propn_raw, normalized=propn_norm,
                    confidence=propn_score,
                    pos_pattern="_".join(t.pos_ for t in tail),
                    token_count=len(tail), is_propn=True,
                    has_digit=any(c.isdigit() for c in propn_raw),
                    has_hyphen="-" in propn_raw,
                    ner_labels=frozenset((getattr(t, "ent_type_", "") or "O") for t in tail),
                    is_name=_all_capitalised(tail, sent_starts), start=tail[0].i,
                )) and debug:
                    print(f"  propn_tail '{propn_raw}' -> '{propn_norm}' score={propn_score:.2f}")

    # Name spans no chunk touched ("United States Mint" whose chunk was "Mint"
    # is emitted above; a name the chunker never reached is emitted here).
    for si in range(len(spans)):
        emit_span(si)
    # Text order: the head keeps the first max_entities_per_record.
    out.sort(key=lambda c: c.start)
    return out


# ------------------------
# Span cleaning
# ------------------------

def _clean_span(doc, start: int, end: int) -> list:
    """The span's content tokens: determiners and punctuation skipped,
    cut at the first preposition, the "s" of a split apostrophe dropped."""
    toks = []
    for t in doc[start:end]:
        low = t.text.lower()
        if low in _DETERMINERS:
            continue
        if low in _PREPOSITIONS:
            break
        if t.is_punct:
            continue
        if low == "s" and t.i > 0 and doc[t.i - 1].text in _APOSTROPHES:
            continue
        toks.append(t)
    return toks


def _has_article(doc, start: int, end: int) -> bool:
    """An article or demonstrative among the span's tokens before its first
    preposition (the tokens _clean_span reads)."""
    for t in doc[start:end]:
        low = t.text.lower()
        if low in _PREPOSITIONS:
            return False
        if low in _DETERMINERS:
            return True
    return False


def _clean_span_tokens(doc, start: int, end: int) -> List[str]:
    """Remove determiners and structural tokens from a span."""
    return [t.text for t in _clean_span(doc, start, end)]


def _span_ner_labels(doc, start: int, end: int) -> Optional[FrozenSet[str]]:
    """NER labels over the span's content tokens ("O" for an unlabelled
    token) -- the same tokens the candidate text is built from."""
    toks = _clean_span(doc, start, end)
    if not toks:
        return None
    return frozenset((getattr(t, "ent_type_", "") or "O") for t in toks)


# ------------------------
# Scoring
# ------------------------

def _score_candidate(raw: str, head, doc, start: int, end: int,
                     token_count: Optional[int] = None) -> float:
    """``token_count``: the mention's content tokens (articles and
    punctuation out); defaults to the whole chunk doc[start:end]."""
    score = 0.0

    if head.pos_ == "PROPN":
        score += 0.45
    else:
        score += 0.25

    if token_count is None:
        token_count = len(doc[start:end])
    if token_count > 1:
        score += 0.15

    if re.search(r"\d", raw) or "-" in raw:
        score += 0.25

    head_lemma = head.lemma_.lower()
    if head_lemma in _CONTEXTUAL_NOUNS:
        if re.search(r"\d", raw):
            score -= 0.05
        else:
            score -= 0.20

    score = max(0.0, min(score, 1.0))
    return score


# ------------------------
# Normalization
# ------------------------

def normalize_mention(text: str) -> str:
    """Canonical snake_case normalization."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"[\s\-]+", "_", text).strip("_")
    return text


# ------------------------
# Regex fallback (no parsed doc available)
# ------------------------

def _regex_fallback(text: str) -> List[EntityCandidate]:
    """Conservative fallback when no ParsedDocument is provided."""
    results: List[EntityCandidate] = []
    seen: set = set()

    for m in re.finditer(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", text):
        name = f"{m.group(1)} {m.group(2)}"
        norm = normalize_mention(name)
        if norm in seen:
            continue
        seen.add(norm)
        results.append(EntityCandidate(text=name, normalized=norm, confidence=0.6))

    for match in re.findall(r"\b[A-Za-z\-]*\d+[A-Za-z\-]*\b", text):
        results.append(
            EntityCandidate(
                text=match,
                normalized=normalize_mention(match),
                confidence=0.6,
            )
        )

    if not results:
        for tok in re.findall(r"\b[a-zA-Z]{4,}\b", text):
            norm = normalize_mention(tok)
            if norm in seen:
                continue
            seen.add(norm)
            results.append(EntityCandidate(text=tok, normalized=norm, confidence=0.35))

    return results
