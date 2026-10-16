"""Sentence features for the CCO ontology codebook.

One function, used by both training and the head, turns a parsed sentence (the
shared ParsedDocument from the production NLP backend) into a fixed-width,
L2-normalised feature vector. No parser is loaded here: the sentence must
already be parsed.

Why these features (lessons from the sketch head, docs/plans/ontology-bfo-cco.md):
- Length invariance: features are weighted lexical cues hashed into a fixed
  vector and L2-normalised, so a one-clause sentence and a long one land on the
  same scale. The sketch head's "any token" booleans saturated on long text and
  were zero on short text.
- What a sentence says a thing IS dominates: the predicate noun of a copular
  clause ("Tennessee is a ... *state*", "Einstein was a German *physicist*") and
  its modifiers carry the most weight; the appositive head ("Nashville, the
  *capital* ...") likewise.
- Subject pronouns ("he"/"she" vs "it") separate persons from everything else
  without trusting the rule NER, which labels most short proper nouns PERSON.
- NER labels are one weak cue among many, never the decider.
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence

import numpy as np

FEATURE_VERSION = "cco-sentence-v2"
DIM = 1 << 12  # 4096 hashed buckets

# Cue weights. Stated here (and hashed into FEATURE_SPEC) so an artifact
# trained with other weights refuses to load.
WEIGHTS = {
    "pred": 4.0,      # predicate noun of a copular clause
    "predmod": 1.5,   # adjective / compound modifying the predicate noun
    "appos": 3.0,     # appositive head noun
    "subjpron": 3.0,  # subject pronoun class (he/she, it, they)
    "subjhead": 2.0,  # common-noun head of the subject ("The *Battle* of ...")
    "noun": 1.0,      # every other noun lemma
    "verb": 1.0,      # verb lemmas
    "ner": 0.5,       # NER label of an entity in the sentence
    "num": 0.5,       # a number / year appears
    "title": 4.0,     # list-style title fragment (TITLE_KINDS)
}

MIN_TOKENS = 3  # shorter sentences (the bare "Title." fragment) carry no cues

# Editorial pointers and notes, not content: a sentence that opens this way
# carries no cues (wiki corpora leave them in leads; see the
# gotcha-wiki-corpus-pointer-pages memory note).
POINTER_RE = re.compile(
    r"^\W*(see also\b|this article is about\b|this article\b|for other uses\b|"
    r"not to be confused with\b|(this is |the following is )?an? (incomplete )?list of\b|"
    r"(this is |the following is )?a list of\b|.{0,80}\b(may|can|could) (also )?refer to\b)",
    re.I)

# The DCP adapter prepends the record's title ("List_of_Polydor_Records_artists.")
# as a fragment the head otherwise ignores. A list or index page's abstract is
# often the bare list ("John Adair / B. R. Ambedkar / ..."), so its title is the
# only text saying what the record is: such a fragment carries one cue,
# title:<kind>, and the codebook learns its class from the labels.
TITLE_KINDS = (
    ("list", re.compile(r"^Lists?_of_", re.I)),
    ("timeline", re.compile(r"^Timeline_of_", re.I)),
    ("index", re.compile(r"^Index_of_", re.I)),
    ("outline", re.compile(r"^Outline_of_", re.I)),
    ("glossary", re.compile(r"^Glossary_of_", re.I)),
    ("bibliography", re.compile(r"^Bibliography_of_", re.I)),
    ("listings", re.compile(r"^National_Register_of_Historic_Places_listings_in_", re.I)),
    ("discography", re.compile(r"_discography\W*$", re.I)),
    ("filmography", re.compile(r"_filmography\W*$", re.I)),
)

_BE = {"be"}
_PERSON_PRON = {"he", "she", "him", "her", "his", "hers"}
_THING_PRON = {"it", "its"}
_GROUP_PRON = {"they", "them", "their"}
_NOUN_POS = {"NOUN", "PROPN"}
_PHRASE_POS = {"DET", "ADJ", "NOUN", "PROPN", "ADV", "NUM", "VERB"}


FEATURE_SPEC = hashlib.sha256(
    (FEATURE_VERSION + repr(DIM) + repr(sorted(WEIGHTS.items())) + repr(MIN_TOKENS) + POINTER_RE.pattern
     + repr([(k, r.pattern) for k, r in TITLE_KINDS])).encode()
).hexdigest()


# 8,192 entries (~2 MB) keep ~82% of hits; a 262,144-entry cache cost ~78 MB
# per process for ~4% more speed (measured 2026-10-04) and pushed the
# ontology container toward its memory limit.
BUCKET_CACHE_SIZE = 8192


@lru_cache(maxsize=BUCKET_CACHE_SIZE)
def _bucket(key: str):
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little")
    return v % DIM, (1.0 if (v >> 63) & 1 else -1.0)


def cues(tokens: Sequence, ents: Iterable = ()) -> List[tuple]:
    """(cue key, weight) pairs for one sentence.

    ``tokens``: the sentence's tokens (``.lemma_``, ``.pos_``, ``.dep_``,
    ``.like_num``). ``ents``: entity spans with ``.label_``.
    """
    toks = [t for t in tokens if not getattr(t, "is_punct", False)]
    out: List[tuple] = []
    lemmas = [(t.lemma_ or t.text or "").lower() for t in toks]

    # Predicate noun: the head of the noun phrase that follows the first form
    # of "be" ("is a landlocked *state*", "was a German-born theoretical
    # *physicist*"). The phrase runs over determiners, adjectives, nouns and
    # adverbs, up to 8 tokens; its head is the last common noun (else the last
    # proper noun). Position-based, because the rule parser labels the
    # predicate dobj/attr/ROOT inconsistently.
    pred_idx: Optional[int] = None
    phrase: List[int] = []
    for i, lem in enumerate(lemmas):
        if lem in _BE:
            for j in range(i + 1, min(i + 9, len(toks))):
                if toks[j].pos_ not in _PHRASE_POS:
                    break
                phrase.append(j)
            break
    nouns = [j for j in phrase if toks[j].pos_ == "NOUN"] or \
            [j for j in phrase if toks[j].pos_ == "PROPN"]
    if nouns:
        pred_idx = nouns[-1]
        out.append(("pred:" + lemmas[pred_idx], WEIGHTS["pred"]))
        for m in phrase:
            if m != pred_idx and toks[m].pos_ in {"ADJ", "NOUN", "PROPN", "VERB"}:
                out.append(("predmod:" + lemmas[m], WEIGHTS["predmod"]))

    # Subject head: the first common noun before the first "be"/verb, i.e. the
    # subject phrase's head ("The *Battle* of Waterloo was fought"). Common
    # nouns only: a proper name would just be memorised.
    for i, t in enumerate(toks):
        if lemmas[i] in _BE or t.pos_ in {"VERB", "AUX"}:
            break
        if t.pos_ == "NOUN":
            out.append(("subjhead:" + lemmas[i], WEIGHTS["subjhead"]))
            break

    for i, t in enumerate(toks):
        dep = (t.dep_ or "").lower()
        lem = lemmas[i]
        if dep == "appos" and t.pos_ in _NOUN_POS:
            out.append(("appos:" + lem, WEIGHTS["appos"]))
        if dep in ("nsubj", "nsubjpass") and t.pos_ == "PRON":
            if lem in _PERSON_PRON:
                out.append(("subjpron:person", WEIGHTS["subjpron"]))
            elif lem in _THING_PRON:
                out.append(("subjpron:thing", WEIGHTS["subjpron"]))
            elif lem in _GROUP_PRON:
                out.append(("subjpron:group", WEIGHTS["subjpron"]))
        if t.pos_ in _NOUN_POS and i != pred_idx:
            out.append(("noun:" + lem, WEIGHTS["noun"]))
        elif t.pos_ == "VERB" and lem not in _BE:
            out.append(("verb:" + lem, WEIGHTS["verb"]))
        if getattr(t, "like_num", False):
            out.append(("num", WEIGHTS["num"]))
    for e in ents:
        label = getattr(e, "label_", None) or getattr(e, "label", None)
        if label:
            out.append(("ner:" + label, WEIGHTS["ner"]))
    return out


def vectorize(tokens: Sequence, ents: Iterable = ()) -> np.ndarray:
    """L2-normalised hashed feature vector; all-zero if the sentence has no cues."""
    v = np.zeros(DIM, dtype=np.float32)
    for key, w in cues(tokens, ents):
        idx, sign = _bucket(key)
        v[idx] += sign * w
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def sparse(tokens: Sequence, ents: Iterable = ()):
    """(sorted bucket indices, values) of ``vectorize``'s non-zeros.

    Same float32 accumulation order and the same norm (taken over a dense
    scratch array, so rounding matches exactly), without the dense scan.
    """
    v = np.zeros(DIM, dtype=np.float32)
    touched = []
    for key, w in cues(tokens, ents):
        idx, sign = _bucket(key)
        v[idx] += sign * w
        touched.append(idx)
    idx = np.unique(np.array(touched, dtype=np.int64))
    idx = idx[v[idx] != 0]
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return idx, v[idx]


def title_kind(tokens: Sequence) -> Optional[str]:
    """TITLE_KINDS kind of the prepended title, if this sentence opens with
    one: its first token is the underscore-joined title (no running text
    produces such a token)."""
    if not tokens:
        return None
    first = tokens[0].text or ""
    if "_" not in first or " " in first:
        return None
    for kind, rx in TITLE_KINDS:
        if rx.search(first):
            return kind
    return None


def span_cues(tokens: Sequence, ents: Iterable = ()) -> List[tuple]:
    """Cues of one sentence as the head and the training set see it.

    A sentence opening with a list-style title gets title:<kind>, plus its
    own cues when it is more than the title fragment; any other fragment or
    editorial pointer gets none; anything else gets ``cues``.
    """
    kind = title_kind(tokens)
    if is_pointer(tokens):
        return [("title:" + kind, WEIGHTS["title"])] if kind else []
    out = cues(tokens, ents)
    return ([("title:" + kind, WEIGHTS["title"])] + out) if kind else out


def _accumulate(pairs):
    v = np.zeros(DIM, dtype=np.float32)
    touched = []
    for key, w in pairs:
        idx, sign = _bucket(key)
        v[idx] += sign * w
        touched.append(idx)
    return v, touched


def sparse_span(span):
    """``sparse`` of a sentence span's ``span_cues``; empty without cues."""
    toks = list(span)
    if title_kind(toks) is None:
        if is_pointer(toks):
            return np.empty(0, np.int64), np.empty(0, np.float32)
        return sparse(toks, getattr(span, "ents", []) or [])
    v, touched = _accumulate(span_cues(toks, getattr(span, "ents", []) or []))
    idx = np.unique(np.array(touched, dtype=np.int64))
    idx = idx[v[idx] != 0]
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return idx, v[idx]


def is_pointer(tokens: Sequence) -> bool:
    """A fragment or an editorial pointer/note: carries no cues."""
    words = [t for t in tokens if not getattr(t, "is_punct", False)]
    if len(words) < MIN_TOKENS:
        return True
    return bool(POINTER_RE.match(" ".join((t.text or "") for t in tokens[:30])))


def vectorize_span(span) -> np.ndarray:
    """Vector for a sentence span of a ParsedDocument (``doc.sents`` item).

    All-zero for a fragment (fewer than MIN_TOKENS words) or an editorial
    pointer/note -- exactly the sentences the training set skips.
    """
    toks = list(span)
    if title_kind(toks) is None:
        if is_pointer(toks):
            return np.zeros(DIM, dtype=np.float32)
        return vectorize(toks, getattr(span, "ents", []) or [])
    v, _ = _accumulate(span_cues(toks, getattr(span, "ents", []) or []))
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v
