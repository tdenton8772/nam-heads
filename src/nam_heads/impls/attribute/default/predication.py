"""Predication extraction: what the text says an entity IS or HAS.

The attribute axis needs, per entity, the specifically dependent continuants
(qualities, roles, dispositions) the text asserts of it. Encyclopedic text
asserts them in a handful of surface constructions:

    copular      "Albert Brooks (born ...) is an American actor, comedian ..."
                 "Anacondas are a group of large snakes"
                 "He became the second president", "it remains a port"
    relative     "Henry Walter Bates, who was a naturalist, ..."
    appositive   "Lincoln, the sixteenth president, ..."
                 "Acapulco, a major seaport city, ..."
                 "Zhang Sixun, a Chinese astronomer and mechanical engineer"
    title        "the American physicist Richard Feynman", "King Charles"
    as-role      "Speer served as the Minister of Armaments", "known as a ..."
                 "As mayor of Paris, he ..."
    modifier     "large snakes", "Permanent anchors" (ADJ before the noun)
    pronoun      "He was a member of ..." -> the record's main subject
    anaphor      "This airfield was ...", "The company expanded" -> the
                 main subject; the kind noun is a value of it
    kind_of      "the town of Lostwithiel" -> Lostwithiel: town
    hearst       "genera like Leidyosuchus" -> Leidyosuchus: genus

The main subject is the record title's mention in the text (ingest prefixes
the title; ``title`` reaches every head through ctx["title"]), else the first
copular subject. Pronouns and kind-noun anaphors resolve to it; whether the
resolution holds (he/she vs a non-person subject, "the company" vs a person,
an abstract "this story") is decided when the values are settled, from the
head's store (role and kind flags), and a rejected resolution leaves the
predication anchored on the pronoun / anaphor itself.

The rule NLP backend's dependency labels are unreliable for these
constructions (it attaches most predicate nominals as ``dep`` of a later
verb), so extraction works on token order, POS and paren depth -- the parts
of the parse the backend gets right -- not on dependency arcs. Parser
defects the backend cannot change (the codebook pins its output) are worked
around here: a curly "She’s" is three tokens ("She", "’", "s"/NOUN), a 3sg
verb after a subject noun is often tagged NOUN ("This family exemplifies"),
participles opening an NP are tagged VERB ("a landlocked country").

Each predication carries an ``anchor``: the token span of the subject it is
predicated of. The bundler links an attribute to an entity when the entity's
tokens overlap the anchor; it never re-derives the link from the parse.
"""

import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

COPULA_FORMS = frozenset({"is", "was", "are", "were", "be", "been", "'s"})
# Copula-like verbs: what follows is a predicate of the subject as with "be".
BECOME_FORMS = frozenset({"become", "became", "becomes", "remain", "remains", "remained"})
# "'s" is the copula only after these ("he's", "it's", "that's"); after a
# name it is the possessive ("Gary's impact" is not "Gary is impact").
POSS_COPULA_HOSTS = frozenset({"he", "she", "it", "that", "there", "what", "who", "this",
                               "which", "here"})
APOSTROPHES = ("'", "’", "‘")
# Auxiliaries that may stand between the subject and the copula:
# "Rusyns have been", "it will be", "she had become".
AUX_BEFORE_COPULA = frozenset({"have", "has", "had", "will", "would", "can", "could", "may",
                               "might", "shall", "should", "must", "having"})
SUBJ_PRONOUNS = frozenset({"he", "she", "it", "they"})
# Pronouns that name a non-person subject: "This is a 1999 film".
NONPERSON_PRONOUNS = frozenset({"it", "this"})
# Interrogative subjects ("Who was the German architect ...?"): the query
# side's predication has no entity subject, but its values must match ingest.
WH_SUBJECTS = frozenset({"who", "what", "which"})
REL_PRONOUNS = frozenset({"who", "which", "that"})
AS_ROLE_VERBS = frozenset({"serve", "served", "serves", "serving", "work",
                           "worked", "works", "working", "act", "acted",
                           "acts", "acting",
                           # "known as a ...", "recognized as the ...", "operates as a ..."
                           "known", "recognized", "recognised", "regarded", "considered",
                           "honored", "honoured", "started", "began", "operates", "operated",
                           "operate", "established", "elected", "appointed", "crowned",
                           "described", "designated", "classified", "listed", "ranked",
                           "emerged", "functioned", "functions", "debuted", "retired"})
REFLEXIVES = frozenset({"itself", "himself", "herself", "themselves"})
CLAUSE_OPENERS = frozenset({"and", "but", "so", "yet", "where", "when", "while", "whereas", "because",
                            "although", "though", "since", "after", "before", "until", "once", "if"})
PERSONAL_PRONOUNS = frozenset({"he", "she", "his", "her", "him", "himself", "herself", "hers"})
# Determiners of a kind-noun anaphor: "this airfield", "these snakes", "the company".
ANAPHOR_DETS = frozenset({"this", "these", "the"})
NEGATORS = frozenset({"not", "never", "no", "n't"})
# Regnal numerals closing a name before an appositive comma ("Alaric I, ...").
REGNAL = frozenset({"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"})
DETERMINERS = frozenset({"a", "an", "the", "another", "any", "one"})
# Heads that quantify or partition another noun: "a group of large snakes",
# "one of the earliest settlers". The predication continues after "of".
# Quantifying heads whose of-complement is a different entity.
RELATIONAL_QUANT = frozenset({"member", "members", "part"})
QUANT_HEADS = frozenset({
    "group", "class", "type", "kind", "form", "family", "genus", "species",
    "set", "series", "number", "pair", "branch", "member", "members", "one",
    "part", "collection", "range", "variety", "subset", "category", "sort",
    "order", "subfamily", "line", "list", "couple", "body",
})
NOMINAL_POS = frozenset({"NOUN", "PROPN"})
# Phrase-internal tokens a predicate NP may contain.
NP_POS = frozenset({"ADJ", "NOUN", "PROPN", "NUM"})
MAX_CONJUNCTS = 6
MAX_NP_LEN = 8


@dataclass
class Word:
    i: int            # token index in the doc
    text: str
    lemma: str
    pos: str
    role: str         # "head" | "mod" | "adj" | "quant"
    ner: str = ""     # NER label of the token, if any
    # A lone noun conjunct before a compound NP, "film and television
    # producer": it may share that NP's head (then it is a modifier).
    # attribute_values decides, using whether it names a kind of person.
    shared_head: bool = False


@dataclass
class Predication:
    anchor: Tuple[int, int]        # [start, end) token span of the subject
    # copular | relative | appositive | title | as_role | modifier | pronoun
    # | anaphor | kind_of | hearst | query
    link: str
    words: List[Word] = field(default_factory=list)
    sentence: int = 0              # index of the sentence it was read from
    # Surface evidence that the subject is a person, independent of the
    # (noisy) ontology and NER labels: a he/she subject, "X, who was ...",
    # "... a vlogger who ...", or a birth/lifespan parenthetical between
    # subject and copula ("Albert Brooks (born ...; July 22, 1947) is").
    person_cue: bool = False
    # Resolution of the subject to the main subject: "pronoun" (he/she/it/
    # they/this) or "anaphor" ("this airfield"). ``referent`` is the pronoun
    # or the anaphor's kind lemma; ``own`` the subject's own span, the anchor
    # when the resolution is rejected at settle time (or there is no main
    # subject). ``kind_value``: this predication IS the anaphor's kind noun.
    resolved: str = ""
    referent: str = ""
    own: Optional[Tuple[int, int]] = None
    kind_value: bool = False
    # A kind noun the store must not mark as a non-kind (learning.kind_status)
    # for this predication to hold: kind_of, hearst and anaphor readings.
    needs_kind: str = ""
    # Whether the subject is a name (capitalised): the store learns kind
    # nouns only from predications of named subjects.
    named_subject: bool = False
    # The predicate NP is followed by "of" ("the capital of X"): a relational
    # use, not evidence that its noun names a kind.
    of_complement: bool = False
    # An anaphor the text ties to the main subject: a demonstrative, or a
    # definite whose noun is in the title or was predicated of it earlier.
    supported: bool = True


class Predications(list):
    """The predications of one document, plus what extraction learned about
    the document's main subject: its span (``main``), every mention of the
    record title (``title_spans``), the title's parenthetical qualifier
    ("Yves Rocher (company)" -> "company"), and a surface person cue: the
    title mention is NER-labelled PERSON throughout and the text uses a
    personal pronoun (he/she/his/her/...)."""
    main: Optional[Tuple[int, int]] = None
    title_spans: Tuple[Tuple[int, int], ...] = ()
    qualifier: str = ""
    person_cue: bool = False


def _ner_map(doc) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for ent in getattr(doc, "ents", None) or []:
        for tok in ent:
            out[tok.i] = ent.label_
    return out


def _sentences(doc) -> List[Tuple[int, int]]:
    sents = getattr(doc, "sents", None)
    out = []
    if sents:
        for s in sents:
            out.append((s.start, s.end))
    if not out:
        out = [(0, len(doc))]
    return out


def _depths(doc, start: int, end: int) -> List[int]:
    """Bracket depth of each token in [start, end)."""
    d, out = 0, []
    for i in range(start, end):
        t = doc[i].text
        if t in ("(", "[", "{"):
            out.append(d + 1)
            d += 1
            continue
        if t in (")", "]", "}"):
            out.append(d)
            d = max(0, d - 1)
            continue
        out.append(d)
    return out


def _is_cap(tok) -> bool:
    return tok.text[:1].isupper()


def _lemma(tok) -> str:
    """The token's lemma, except where the rule lemmatizer strips suffixes
    it should not: a positive-form adjective (tag JJ) is the word itself
    ("challenging", not "challenge"), and so is a noun in -ing ("building",
    not "build")."""
    if tok.pos_ == "ADJ" and getattr(tok, "tag_", "") == "JJ":
        return tok.text.lower()
    if tok.pos_ == "NOUN" and tok.text.isalpha():
        # a noun in -ing is not a verb form: "building", "meeting" (the
        # lemmatizer gives "build", "meet"); its plural drops the s
        t = tok.text.lower()
        if t.endswith("ing"):
            return t
        if t.endswith("ings"):
            return t[:-1]
    return tok.lemma_.lower()


def _is_copula(tok) -> bool:
    t = tok.text.lower()
    return (t in COPULA_FORMS or t in BECOME_FORMS) and tok.pos_ in ("AUX", "VERB")


def _copula_at(doc, s: "_Sent", c: int) -> Optional[Tuple[int, int]]:
    """(first, last) token of a copula starting at depth-0 token ``c``.

    "is"/"was"/... and become/remain; "'s" only after a pronoun ("he's",
    not "Gary's"); the curly contraction, which the tokenizer splits into
    "’" + "s", after a pronoun ("She’s" -> She / ’ / s)."""
    tok = doc[c]
    t = tok.text.lower()
    prev = doc[c - 1].text.lower() if c > s.start else ""
    if t == "'s":
        return (c, c) if tok.pos_ in ("AUX", "VERB") and prev in POSS_COPULA_HOSTS else None
    if t in COPULA_FORMS or t in BECOME_FORMS:
        return (c, c) if tok.pos_ in ("AUX", "VERB") else None
    if tok.text in APOSTROPHES and c + 1 < s.end and doc[c + 1].text.lower() == "s" \
            and prev in POSS_COPULA_HOSTS:
        return (c, c + 1)
    return None


def _is_possessive_at(doc, i: int, end: int) -> int:
    """Width of a possessive marker at token ``i`` ("'s", "'", or the curly
    "’" + "s"); 0 when there is none."""
    t = doc[i].text
    if t == "'s":
        return 1
    if t in APOSTROPHES:
        return 2 if i + 1 < end and doc[i + 1].text.lower() == "s" else 1
    return 0


class _Sent:
    """Depth-0 view of one sentence."""

    def __init__(self, doc, start: int, end: int):
        self.doc = doc
        self.start, self.end = start, end
        depths = _depths(doc, start, end)
        self.top = [start + k for k, dep in enumerate(depths) if dep == 0
                    and doc[start + k].text not in (")", "]", "}")]

    def next_top(self, i: int) -> Optional[int]:
        for j in self.top:
            if j > i:
                return j
        return None


def _subject_span(s: _Sent, pivot: int) -> Optional[Tuple[int, int]]:
    """The subject NP of the clause whose verb/copula is at ``pivot``.

    Depth-0 tokens before the pivot. A leading prepositional phrase ending in
    a comma ("In Norse mythology, ...") is skipped, and so is a leading
    participial or adverbial clause ("Situated in Lincolnshire, this
    airfield was", "Historically, Rusyns have been"). The NP is a run of
    DET/ADJ/NOUN/PROPN/NUM tokens, with "of" allowed between capitalised
    tokens ("Anthemius of Tralles"). What follows the NP up to the pivot must
    be empty, a comma-delimited aside, a relative pronoun ("..., who was"),
    or adverbs and auxiliaries ("Rusyns have been", "it has also become").
    A possessor NP is not the subject: "Gary's impact was" anchors on
    "impact".
    """
    doc = s.doc
    pre = [i for i in s.top if i < pivot]
    if not pre:
        return None
    t0 = doc[pre[0]]
    commas = [n for n, i in enumerate(pre) if doc[i].text == ","]
    if t0.pos_ in ("ADP", "SCONJ") or t0.text.lower() in ("during", "in", "on", "at", "as", "after", "before"):
        starts = [n + 1 for n in commas]          # a leading PP: the subject follows one of its commas
    else:
        starts = [0]
        # a participial / adverbial opener: "Founded by X in 1965 in Y, France, the company began"
        if getattr(t0, "tag_", "") in ("VBN", "VBG", "VBD") or t0.pos_ in ("ADV", "VERB"):
            starts += [n + 1 for n in commas]
    for k in starts:
        sp = _subject_span_from(s, pre, k)
        if sp is not None:
            return sp
    return None


def _subject_span_from(s: _Sent, pre: List[int], k: int) -> Optional[Tuple[int, int]]:
    doc = s.doc
    while k < len(pre) and (doc[pre[k]].pos_ in ("DET", "PUNCT") and doc[pre[k]].text != ","):
        k += 1
    if k >= len(pre):
        return None
    first = pre[k]
    ft = doc[first]
    poss_end = None      # index (in pre) after the last possessive marker
    if ft.text.lower() in SUBJ_PRONOUNS or ft.text.lower() in WH_SUBJECTS:
        np_end_k = k + 1
    else:
        if ft.pos_ not in NP_POS:
            return None
        np_end_k = k
        while np_end_k < len(pre):
            t = doc[pre[np_end_k]]
            if t.text in ("-",) or (t.pos_ in NP_POS and not (
                    t.text.lower() == "s" and np_end_k > k and doc[pre[np_end_k - 1]].text in APOSTROPHES)):
                np_end_k += 1
                continue
            width = _is_possessive_at(doc, pre[np_end_k], s.end)
            if width and np_end_k > k and np_end_k + width <= len(pre):
                np_end_k += width
                poss_end = np_end_k
                continue
            # "of" between capitalised tokens: "Republic of Azerbaijan"
            if t.text == "of" and np_end_k + 1 < len(pre) and (
                    _is_cap(doc[pre[np_end_k + 1]]) or doc[pre[np_end_k + 1]].text == "the"):
                np_end_k += 1
                continue
            if t.pos_ == "DET" and np_end_k > k and doc[pre[np_end_k - 1]].text == "of":
                np_end_k += 1
                continue
            break
    if np_end_k == k:
        return None
    rest = [doc[i] for i in pre[np_end_k:]]
    span = (first, pre[np_end_k - 1] + 1)
    # "the western part of Slovene Istria is ...": a lowercase common-noun
    # head before "of" is the subject; the of-phrase names something else.
    # Capitalised heads keep the whole name ("Republic of Azerbaijan").
    for q in range(k, np_end_k):
        t = doc[pre[q]]
        if t.text == "of" and q > k:
            h = doc[pre[q - 1]]
            if h.pos_ == "NOUN" and not _is_cap(h):
                span = (first, pre[q - 1] + 1)
            break
    # "Gary's impact was ...": the possessor is not the subject
    if poss_end is not None and poss_end < np_end_k and pre[poss_end] < span[1]:
        span = (pre[poss_end], span[1])
    if not rest:
        return span
    words = [t.text.lower() for t in rest]
    if words == [","] or words[-1] in REL_PRONOUNS and words[0] == ",":
        return span
    if words[-1] in REL_PRONOUNS and len(words) == 1:
        return span
    # comma-delimited aside directly before the pivot: ", commonly X,"
    if words[0] == "," and words[-1] == ",":
        return span
    # aside ending in relative pronoun: ", also known as X, who"
    if words[0] == "," and len(words) >= 2 and words[-2] == "," and words[-1] in REL_PRONOUNS:
        return span
    # adverbs and auxiliaries only: "Anarchism also is", "Rusyns have been",
    # "it has also become"
    if all(t.pos_ == "ADV" or t.text.lower() in AUX_BEFORE_COPULA for t in rest):
        return span
    return None


def _parse_np(doc, s: _Sent, i: Optional[int], ner: Dict[int, str]) -> Tuple[List[Word], Optional[int]]:
    """Parse one predicate NP starting at depth-0 token ``i``.

    Returns (words, next index after the NP) -- words empty when no NP.
    """
    words: List[Word] = []
    while i is not None and doc[i].text.lower() in DETERMINERS and doc[i].pos_ in ("DET", "NUM", "PRON"):
        if doc[i].text.lower() == "one":
            # "one of the X": quantifier
            nxt = s.next_top(i)
            if nxt is not None and doc[nxt].text == "of":
                i = s.next_top(nxt)
                continue
        i = s.next_top(i)
    if i is not None:
        # "Angola's national military", "Scotland's most famous poets": the
        # possessor names another entity; the NP continues after it
        j, n = i, 0
        while j is not None and n < 4 and (doc[j].pos_ in ("PROPN", "NOUN") or _is_cap(doc[j])) \
                and doc[j].text not in APOSTROPHES:
            j, n = s.next_top(j), n + 1
        width = _is_possessive_at(doc, j, s.end) if (n and j is not None) else 0
        if width:
            i = s.next_top(j + width - 1)
            while i is not None and doc[i].text.lower() in ("most", "more", "least"):
                i = s.next_top(i)
    if i is not None and getattr(doc[i], "tag_", "") in ("VBN", "VBD", "VBG") and not _is_cap(doc[i]) \
            and doc[i].text.isalpha():
        # a participle opening the NP, tagged VERB by the rule tagger: "a
        # landlocked country", "an armored mammal", "a retired professional
        # footballer", "a retired Iranian football defender". The value is the
        # word as written (the verb lemmatizer gives "landlock", "armore").
        # The NP after it must end in a common noun: "a renamed Paris" is not
        # a kind.
        nxt = s.next_top(i)
        if nxt is not None and (doc[nxt].pos_ in ("NOUN", "ADJ") or (doc[nxt].pos_ == "PROPN" and _is_cap(doc[nxt]))):
            more, after = _parse_np(doc, s, nxt, ner)
            if more and any(w.role == "head" and w.pos == "NOUN" and not _is_cap(doc[w.i]) for w in more):
                t = doc[i]
                return [Word(i, t.text, t.text.lower(), "ADJ", "mod", ner.get(i, ""))] + more, after
    run: List[int] = []
    while i is not None and len(run) < MAX_NP_LEN:
        t = doc[i]
        if t.text.lower() == "de" and i + 1 < len(doc) and doc[i + 1].text.lower() in ("facto", "jure"):
            # "the de facto capital": a Latin adverbial, not a modifier
            nxt = s.next_top(i)
            i = s.next_top(nxt) if nxt is not None else None
            continue
        if t.pos_ in NP_POS or t.text == "-":
            run.append(i)
            i = s.next_top(i)
            continue
        # adverb modifying a following adjective: "naturally occurring", "historically left-wing"
        if t.pos_ == "ADV" and t.text.lower() not in NEGATORS:
            nxt = s.next_top(i)
            if nxt is not None and (doc[nxt].pos_ == "ADJ" or getattr(doc[nxt], "tag_", "") in ("VBN", "VBG")):
                i = nxt
                continue
            break
        # participle used as modifier before a noun: "occurring organic compounds"
        if getattr(t, "tag_", "") in ("VBN", "VBG") and run:
            nxt = s.next_top(i)
            if nxt is not None and doc[nxt].pos_ in ("ADJ", "NOUN"):
                run.append(i)
                i = nxt
                continue
        break
    run = [j for j in run if doc[j].text != "-"]
    if not run:
        return [], i
    # head = last common noun; NORP-only phrases ("a Byzantine Greek") take the last token
    # The rule tagger sometimes tags a lowercase noun PROPN ("football
    # player/PROPN"); a lowercase word is never a proper name here.
    def _common(t):
        return t.pos_ == "NOUN" or (t.pos_ == "PROPN" and t.text[:1].islower() and t.text.isalpha())

    head_k = None
    for k in range(len(run) - 1, -1, -1):
        if _common(doc[run[k]]):
            head_k = k
            break
    if head_k is None:
        if any(doc[j].pos_ == "PROPN" or ner.get(j) == "NORP" for j in run):
            head_k = len(run) - 1
    for k, j in enumerate(run):
        t = doc[j]
        if head_k is None:
            role = "adj" if t.pos_ == "ADJ" else "mod"
        elif k == head_k:
            role = "head"
        elif k < head_k:
            role = "mod"
        else:
            continue
        pos = "NOUN" if t.pos_ == "PROPN" and _common(t) else t.pos_
        words.append(Word(j, t.text, _lemma(t), pos, role, ner.get(j, "")))
    if head_k is not None and normalise(words[-1] if words[-1].role == "head" else
                                        next(w for w in words if w.role == "head")) in GENERIC_HEADS:
        # "a vital meeting place": a participle before a generic head forms
        # a compound with it ("meeting place"); neither describes the subject
        words = [w for w in words if w.pos != "VERB"]
    # "a group of large snakes": follow "of" past quantifying heads
    if head_k is not None and doc[run[head_k]].text.lower() in QUANT_HEADS and i is not None \
            and doc[i].text == "of":
        for w in words:
            if w.role == "head":
                w.role = "quant"
        more, i = _parse_np(doc, s, s.next_top(i), ner)
        # "a member of the House": the House is what the subject belongs to,
        # not a property of it; "a group of large snakes": the snakes ARE
        # the subject.
        if doc[run[head_k]].text.lower() not in RELATIONAL_QUANT:
            words.extend(more)
    return words, i


def _parse_predicate(doc, s: _Sent, i: Optional[int], ner: Dict[int, str]) -> Tuple[List[Word], Optional[int]]:
    """Predicate after a copula: adverbs, then NP (or ADJ phrase) conjuncts."""
    while i is not None and doc[i].pos_ == "ADV" and doc[i].text.lower() not in NEGATORS:
        nxt = s.next_top(i)
        if nxt is not None and doc[nxt].pos_ == "ADJ":
            break
        i = nxt
    if i is None or doc[i].text.lower() in NEGATORS:
        return [], i
    t = doc[i]
    if not (t.pos_ in NP_POS or t.pos_ in ("DET", "ADV") or t.text.lower() in DETERMINERS):
        return [], i
    if getattr(t, "tag_", "") in ("VBN", "VBG"):
        return [], i
    words: List[Word] = []
    conjuncts: List[List[Word]] = []
    for _ in range(MAX_CONJUNCTS):
        np_words, i = _parse_np(doc, s, i, ner)
        if not np_words:
            break
        words.extend(np_words)
        conjuncts.append(np_words)
        if i is None:
            break
        # coordination: ", X", "and X", ", and X"
        j = i
        if doc[j].text == ",":
            j = s.next_top(j)
        if j is not None and doc[j].text.lower() in ("and", "or"):
            j = s.next_top(j)
        if j is None or j == i:
            break
        nt = doc[j]
        if nt.pos_ in ("ADJ", "NOUN") or nt.text.lower() in ("a", "an", "the"):
            i = j
            continue
        break
    # "central to Bomber Command operations", "similar to X": an adjective
    # with a to-complement relates the subject to something; not a property
    words = [w for w in words if not (w.pos == "ADJ" and w.i + 1 < len(doc) and doc[w.i + 1].text.lower() == "to")]
    for k, cj in enumerate(conjuncts[:-1]):
        later = conjuncts[k + 1:]
        h = cj[-1]
        if h.role == "head" and h.pos == "NOUN" and (len(cj) == 1 or cj[-2].pos != "NOUN") and any(
                len(c) >= 2 and c[-1].role == "head" and c[-2].pos == "NOUN" for c in later):
            h.shared_head = True
    return words, i


def _lifespan_between(doc, start: int, end: int) -> bool:
    """A birth or lifespan parenthetical between subject and verb: "(born
    ...)" or two years, "(19 March 1905 - 1 September 1981)". One year alone
    is not enough: "Mad Love (2001 film) is ..." """
    depth, years = 0, 0
    for i in range(start, end):
        t = doc[i].text
        if t in ("(", "["):
            depth += 1
        elif t in (")", "]"):
            depth = max(0, depth - 1)
        elif depth and t.lower() == "born":
            return True
        elif depth and t.isdecimal() and 3 <= len(t) <= 4:   # a year: decimal digits, not "²⁵⁶"
            years += 1
    return years >= 2


# extract_predications is pure in (doc, title), and one head batch asks for
# the same doc's predications three times (candidate_values for the store
# read, attribute_values, legacy_anchors). The last few docs' results are
# kept, keyed by object identity and title (the entry holds the doc, so its
# id cannot be reused while cached). Callers treat the returned list as
# read-only.
_PRED_MEMO: Dict[Tuple[int, str], tuple] = {}
_PRED_MEMO_MAX = 128     # above the worker batch (<= 64 records); docs are large


def extract_predications(doc, title: str = "") -> List[Predication]:
    """All predications in a parsed document, in token order (memoized per
    doc object and title; the list is shared -- do not mutate it).
    ``title``: the record's title (ctx["title"]); its mention in the text is
    the main subject."""
    if doc is None or len(doc) == 0:
        return Predications()
    key = (id(doc), title or "")
    hit = _PRED_MEMO.get(key)
    if hit is not None and hit[0] is doc:
        return hit[1]
    out = _extract_predications(doc, title or "")
    if len(_PRED_MEMO) >= _PRED_MEMO_MAX:
        _PRED_MEMO.clear()
    _PRED_MEMO[key] = (doc, out)
    return out


# ------------------------------------------------------------- main subject
_WORD_RE = re.compile(r"\w+")
_QUALIFIER_RE = re.compile(r"\s*\(([^()]*)\)\s*$")


def split_title(title: str) -> Tuple[List[str], str]:
    """(the title's lowercase words, its parenthetical qualifier).

    "Yves Rocher (company)" -> (["yves", "rocher"], "company"); the
    qualifier is the last alphabetic word in the trailing brackets
    ("Mad Love (2001 film)" -> "film")."""
    t = (title or "").strip()
    qualifier = ""
    m = _QUALIFIER_RE.search(t)
    if m:
        words = [w for w in _WORD_RE.findall(m.group(1).lower()) if w.isalpha()]
        qualifier = words[-1] if words else ""
        t = t[:m.start()]
    return _WORD_RE.findall(t.lower()), qualifier


def title_spans(doc, title: str) -> List[Tuple[int, int]]:
    """Every mention of the title in the doc, as token spans.

    Matched on word characters, so tokenization and punctuation do not
    matter ("Great Wyrley F.C." against F / . / C / ., "d'Isonzo" either
    way); case-insensitive; the last word may be plural ("Bee" -> "Bees")."""
    words, _q = split_title(title)
    if not words or doc is None:
        return []
    parts: List[Tuple[str, int]] = []
    for tok in doc:
        for w in _WORD_RE.findall(tok.text.lower()):
            parts.append((w, tok.i))
    n, out, k = len(words), [], 0
    last = (words[-1], words[-1] + "s", words[-1] + "es")    # "Air show" -> "Air shows ..."
    while k + n <= len(parts):
        if all(parts[k + q][0] == words[q] for q in range(n - 1)) and parts[k + n - 1][0] in last:
            out.append((parts[k][1], parts[k + n - 1][1] + 1))
            k += n
        else:
            k += 1
    return out


def _sentence_of(sents: List[Tuple[int, int]], i: int) -> int:
    for si, (a, b) in enumerate(sents):
        if a <= i < b:
            return si
    return -1


def _named(doc, span: Tuple[int, int]) -> bool:
    """A subject that is a name: a capitalised token that is not a pronoun
    or a determiner."""
    return any(_is_cap(doc[k]) and doc[k].text.lower() not in SUBJ_PRONOUNS
               and doc[k].pos_ not in ("DET", "PRON") for k in range(*span))


def _is_pronoun_span(doc, span: Tuple[int, int]) -> bool:
    return span[1] - span[0] == 1 and doc[span[0]].text.lower() in (SUBJ_PRONOUNS | {"this"})


def _subject(s: _Sent, pivot: int) -> Optional[Tuple[int, int]]:
    """_subject_span, else a pronoun directly before the pivot (adverbs and
    auxiliaries may come between) that opens its clause: ", he was",
    "and it is", "-- it's"; a sentence-initial "This is ..."."""
    sp = _subject_span(s, pivot)
    if sp is not None:
        return sp
    doc = s.doc
    prev = [j for j in s.top if j < pivot]
    k = len(prev) - 1
    while k >= 0 and (doc[prev[k]].pos_ == "ADV" or doc[prev[k]].text.lower() in AUX_BEFORE_COPULA):
        k -= 1
    if k < 0:
        return None
    w = doc[prev[k]].text.lower()
    # the pronoun opens its clause: after punctuation, a conjunction or a
    # clause-introducing adverb ("..., where he was eventually honored as")
    opens = k == 0 or doc[prev[k - 1]].pos_ in ("PUNCT", "CCONJ", "SCONJ") or \
        doc[prev[k - 1]].text.lower() in CLAUSE_OPENERS
    if opens and (w in SUBJ_PRONOUNS or w == "this"):
        return (prev[k], prev[k] + 1)
    return None


_FUNCTION_POS = ("DET", "ADP", "ADV", "PRON", "SCONJ", "PART", "ADJ")


def _misread_verb(doc, j: int) -> bool:
    """A 3sg verb the rule tagger read as a NOUN: "This family exemplifies
    how", "The album highlights not just"."""
    t = doc[j]
    return t.pos_ == "NOUN" and not _is_cap(t) and t.text.lower().endswith("s") and j + 1 < len(doc) and (
        doc[j + 1].pos_ in _FUNCTION_POS or doc[j + 1].text.lower() in ("how", "that", "not"))


def _anaphor_body(doc, span: Tuple[int, int]) -> Optional[List[int]]:
    """The kind noun phrase of a "this/these/the <ADJ>* NOUN" subject (token
    indices, head last), or None. Capitalised words, generic heads and
    "the N of X" (relational) do not qualify."""
    a, b = span
    if doc[a].text.lower() in ANAPHOR_DETS:
        body = list(range(a + 1, b))
    elif a > 0 and doc[a - 1].text.lower() in ANAPHOR_DETS:
        body = list(range(a, b))        # _subject_span skips the determiner
    else:
        return None
    if len(body) >= 2 and _misread_verb(doc, body[-1]):
        body = body[:-1]
    # "this unusually tall manuscript": an adverb grading an adjective
    body = [k for k in body if not (doc[k].pos_ == "ADV" and k + 1 in body and doc[k + 1].pos_ == "ADJ")]
    if not body or any(_is_cap(doc[k]) or doc[k].pos_ not in ("ADJ", "NOUN") for k in body):
        return None
    head = doc[body[-1]]
    lem = head.lemma_.lower()
    if head.pos_ != "NOUN" or lem in GENERIC_HEADS or head.text.lower() in GENERIC_HEADS:
        return None
    nxt = body[-1] + 1
    if nxt < len(doc) and doc[nxt].text.lower() == "of":
        return None
    return body


def _named_elsewhere(doc, lemma: str, before: int, skip: set) -> bool:
    """The kind noun already named something other than the main subject
    earlier in the text: as part of a name ("... the First English Civil
    War ... The war"), or introduced as another one or a part of one ("a
    war", "another company", "Spain's war", "its leaves"). Then the anaphor
    refers to that. An earlier "the/this <noun>" is itself an anaphor and
    does not block."""
    for k in range(before):
        if k in skip:
            continue
        t = doc[k]
        if t.pos_ not in ("NOUN", "PROPN") or not (t.lemma_.lower() == lemma or t.text.lower() == lemma):
            continue
        if _is_cap(t) and k > 0 and doc[k - 1].text not in (".", "!", "?", '"'):
            return True
        j = k - 1
        while j >= 0 and doc[j].pos_ == "ADJ":
            j -= 1
        if j >= 0 and (doc[j].text.lower() in ("a", "an", "another", "other", "one", "its", "his", "her",
                                               "their", "our", "my", "your")
                       or _is_possessive_at(doc, j, len(doc))
                       or (j > 0 and doc[j].text.lower() == "s" and doc[j - 1].text in APOSTROPHES)):
            return True
    return False


def _word(doc, k: int, role: str, ner: Dict[int, str]) -> Word:
    t = doc[k]
    return Word(k, t.text, _lemma(t), t.pos_, role, ner.get(k, ""))


def _extract_predications(doc, title: str = "") -> List[Predication]:
    ner = _ner_map(doc)
    out = Predications()
    sents = _sentences(doc)
    spans = title_spans(doc, title) if title else []
    out.title_spans = tuple(spans)
    out.qualifier = split_title(title)[1] if title else ""
    if spans and all(ner.get(k) == "PERSON" for k in range(*spans[0])):
        out.person_cue = any(t.text.lower() in PERSONAL_PRONOUNS for t in doc)
    main: Optional[Tuple[int, int]] = spans[0] if spans else None
    main_sent = _sentence_of(sents, main[0]) if main else -1
    # Tokens that ARE the main subject or were predicated of it: an earlier
    # occurrence of an anaphor's kind noun there does not block it.
    of_main: set = set()
    for a, b in spans:
        of_main.update(range(a, b))
        if b < len(doc) and doc[b].text == "(":         # "Yves Rocher (company)"
            k = b
            while k < len(doc) and doc[k].text != ")":
                of_main.add(k)
                k += 1

    # Nouns known to name the main subject: the title's words and qualifier,
    # and every noun predicated of it so far. A definite "the <noun>" refers
    # to the main subject only through one of these ("The company" in
    # "Yves Rocher (company)", "Kucura is a village ... the village"); a
    # demonstrative "this/these <noun>" needs no such support.
    main_kinds: set = set(split_title(title)[0]) | ({out.qualifier} if out.qualifier else set())

    def on_main(p: Predication):
        if not p.supported:
            return
        if main is not None and (p.anchor == main or any(_overlaps(p.anchor, sp) for sp in spans)):
            of_main.update(w.i for w in p.words)
            main_kinds.update(normalise(w) for w in p.words)

    def resolve_pronoun(p: Predication, subj):
        w = doc[subj[0]].text.lower()
        p.resolved, p.referent, p.own = "pronoun", w, subj
        p.person_cue = w in ("he", "she")
        p.named_subject = False
        p.anchor = main if main is not None else subj

    def anaphor_of(subj, si) -> Optional[Tuple[List[int], str, bool]]:
        """(kind NP token indices, its noun's value, supported) for a
        this/these/the <noun> subject that may refer to the main subject."""
        if main is None or si <= main_sent or subj is None:
            return None
        if any(a <= subj[0] < b for a, b in spans):
            return None
        body = _anaphor_body(doc, subj)
        if body is None:
            return None
        lem = normalise(_word(doc, body[-1], "head", ner))
        if _named_elsewhere(doc, lem, body[0], of_main):
            return None
        det = doc[subj[0]] if doc[subj[0]].text.lower() in ANAPHOR_DETS else doc[subj[0] - 1]
        # A definite "the <noun>" the text has not tied to the main subject
        # ("the gold shield", "the wealthier citizens") holds only if the
        # store knows the noun as a kind (resolution_holds, settle time).
        supported = det.text.lower() != "the" or lem in main_kinds or doc[body[-1]].text.lower() in main_kinds
        return body, lem, supported

    def kind_value(an, si) -> Predication:
        body, lem, supported = an
        ws = [_word(doc, k, "head" if k == body[-1] else "mod", ner) for k in body]
        if supported:
            of_main.update(body)        # a later "the <noun>" refers back through it
            main_kinds.add(lem)
        return Predication(main, "anaphor", ws, sentence=si, resolved="anaphor", referent=lem,
                           own=(body[0], body[-1] + 1), kind_value=True, needs_kind=lem,
                           supported=supported)

    for si, (start, end) in enumerate(sents):
        s = _Sent(doc, start, end)
        if not s.top:
            continue
        top = s.top
        anaphor_done = False
        # ---- copular / relative / pronoun / anaphor ------------------------
        for c in top:
            cop = _copula_at(doc, s, c)
            if cop is None:
                continue
            subj = _subject(s, cop[0])
            if subj is None:
                continue
            words, after = _parse_predicate(doc, s, s.next_top(cop[1]), ner)
            if not words:
                continue
            link = "copular"
            cue = _lifespan_between(doc, subj[1], cop[0]) or (
                after is not None and doc[after].text.lower() == "who")
            pre_pivot = [doc[i].text.lower() for i in top if subj[1] <= i < cop[0]]
            if pre_pivot and pre_pivot[-1] in REL_PRONOUNS:
                link = "relative"
                cue = cue or pre_pivot[-1] == "who"
            p = Predication(subj, link, words, sentence=si, person_cue=cue,
                            named_subject=_named(doc, subj),
                            of_complement=after is not None and doc[after].text.lower() == "of")
            an = None
            if _is_pronoun_span(doc, subj):
                resolve_pronoun(p, subj)
                p.link = "pronoun"
            else:
                an = anaphor_of(subj, si)
            if an is not None:
                kv = kind_value(an, si)
                out.append(kv)
                p.anchor, p.link, p.named_subject = main, "anaphor", False
                p.resolved, p.referent, p.own = "anaphor", kv.referent, subj
                p.supported = kv.supported
                anaphor_done = True
            out.append(p)
            on_main(p)
            if main is None and link == "copular" and si <= 1 and not p.resolved:
                main, main_sent = subj, si
                of_main.update(range(*subj))
                of_main.update(w.i for w in words)
            # "... a movement that is skeptical of ...": relative ADJ predicate
            if after is not None and doc[after].text.lower() in REL_PRONOUNS:
                c2 = s.next_top(after)
                if c2 is not None and _is_copula(doc[c2]):
                    w2, _ = _parse_predicate(doc, s, s.next_top(c2), ner)
                    w2 = [w for w in w2 if w.role == "adj"]
                    if w2:
                        out.append(Predication(p.anchor, "relative", w2, sentence=si,
                                               resolved=p.resolved, referent=p.referent, own=p.own,
                                               named_subject=p.named_subject))
            # "... a leader born in 1801 who became the second president":
            # a later copula introduced by who/which predicates of the same subject
            for c3 in top:
                if c3 <= cop[1] or doc[c3].text.lower() not in ("who", "which"):
                    continue
                c4 = s.next_top(c3)
                cop2 = _copula_at(doc, s, c4) if c4 is not None else None
                if cop2 is None:
                    continue
                w3, _ = _parse_predicate(doc, s, s.next_top(cop2[1]), ner)
                if w3:
                    out.append(Predication(p.anchor, "relative", w3, sentence=si, person_cue=p.person_cue,
                                           resolved=p.resolved, referent=p.referent, own=p.own,
                                           named_subject=p.named_subject))
                    on_main(out[-1])
            break  # one main predication per sentence
        # ---- anaphor subject of any verb: "The company expanded ..." -------
        if not anaphor_done and main is not None and si > main_sent:
            piv, subj = None, None
            for j in top:       # the first verb with a subject: "Situated in X, this airfield was"
                if doc[j].pos_ in ("VERB", "AUX"):
                    piv = piv if piv is not None else j
                    subj = _subject_span(s, j)
                    if subj is not None:
                        break
            if subj is None:
                for j in top:
                    if piv is not None and j >= piv:
                        break
                    if j > top[0] and _misread_verb(doc, j):
                        subj = _subject_span(s, j)
                        break
            an = anaphor_of(subj, si)
            if an is not None:
                out.append(kind_value(an, si))
        # ---- a demonstrative object: "a scribe fashioned this unusually tall
        # manuscript" -- the kind noun is a value of the main subject --------
        if main is not None and si > main_sent:
            for n in range(1, len(top)):
                if doc[top[n]].text.lower() not in ("this", "these") or doc[top[n - 1]].pos_ != "VERB":
                    continue
                q = n + 1
                while q < len(top) and top[q] == top[q - 1] + 1 and doc[top[q]].pos_ in ("ADV", "ADJ", "NOUN") \
                        and not _is_cap(doc[top[q]]):
                    q += 1
                if q == n + 1:
                    continue
                an = anaphor_of((top[n], top[q - 1] + 1), si)
                if an is not None:
                    out.append(kind_value(an, si))
        # ---- fronted "As mayor of Paris, he ..." ----------------------------
        if doc[top[0]].text.lower() == "as":
            words, after = _parse_predicate(doc, s, s.next_top(top[0]), ner)
            # "As mayor of Paris, he": the predicate's of-phrase runs to the comma
            hops = 0
            while words and after is not None and doc[after].text != "," and hops < 8 and (
                    doc[after].text.lower() == "of" or doc[after].pos_ in ("DET", "PROPN", "NOUN", "ADJ", "NUM")):
                after, hops = s.next_top(after), hops + 1
            nxt = s.next_top(after) if (words and after is not None and doc[after].text == ",") else None
            if nxt is not None and doc[nxt].pos_ == "PROPN" and _is_cap(doc[nxt]):
                m = nxt
                while s.next_top(m) == m + 1 and doc[m + 1].pos_ == "PROPN":
                    m += 1
                out.append(Predication((nxt, m + 1), "fronted_as", words, sentence=si, named_subject=True))
            elif nxt is not None and doc[nxt].text.lower() in SUBJ_PRONOUNS:
                p = Predication((nxt, nxt + 1), "fronted_as", words, sentence=si)
                resolve_pronoun(p, (nxt, nxt + 1))
                out.append(p)
        # ---- as-role: "served as the Minister of X", "known as a ..." -------
        for v in top:
            if doc[v].text.lower() not in AS_ROLE_VERBS:
                continue
            a = s.next_top(v)
            if a is not None and doc[a].text.lower() in REFLEXIVES:     # "established itself as"
                a = s.next_top(a)
            if a is None or doc[a].text.lower() != "as":
                continue
            subj = _subject(s, v)
            if subj is None:
                # "X is recognized as", "He was later elected as": the
                # subject of the copula before the verb
                cop = next((j for j in top if j < v and _copula_at(doc, s, j)), None)
                subj = _subject(s, cop) if cop is not None else None
            if subj is None:
                # an aside: "This plant family, commonly known as the mustard
                # family, is ...": the NP before the comma
                pv = [j for j in top if j < v]
                q = len(pv) - 1
                while q >= 0 and doc[pv[q]].pos_ == "ADV":
                    q -= 1
                if q >= 1 and doc[pv[q]].text == ",":
                    subj = _subject(s, pv[q])
            if subj is None:
                continue
            words, _ = _parse_predicate(doc, s, s.next_top(a), ner)
            if not words:
                continue
            p = Predication(subj, "as_role", words, sentence=si,
                            person_cue=_lifespan_between(doc, subj[1], v), named_subject=_named(doc, subj))
            if _is_pronoun_span(doc, subj):
                resolve_pronoun(p, subj)
            else:
                an = anaphor_of(subj, si)
                if an is not None:
                    p.anchor, p.named_subject = main, False
                    p.resolved, p.referent, p.own, p.supported = "anaphor", an[1], subj, an[2]
            out.append(p)
            on_main(p)
        # ---- appositive: "Lincoln, the sixteenth president," ----------------
        for n, c in enumerate(top):
            if doc[c].text != "," or n == 0:
                continue
            prev = doc[top[n - 1]]
            if prev.pos_ != "PROPN":
                # "Alaric I ,": regnal numeral; "Seikilos epitaph ,": name + noun
                if not (n >= 2 and doc[top[n - 2]].pos_ == "PROPN" and top[n - 2] == top[n - 1] - 1
                        and (prev.text in REGNAL or prev.pos_ in ("NUM", "NOUN"))):
                    continue
            nxt = top[n + 1] if n + 1 < len(top) else None
            if nxt is None or doc[nxt].text.lower() not in ("a", "an", "the", "one"):
                continue
            words, after = _parse_predicate(doc, s, nxt, ner)
            if not any(w.role == "head" and w.pos == "NOUN" and not _is_cap(doc[w.i]) for w in words):
                continue
            if after is not None and doc[after].text not in (",", ".", ";", ")", "—", "–", "-") \
                    and doc[after].pos_ != "ADP" and doc[after].text.lower() not in REL_PRONOUNS:
                continue
            k = n - 1
            while k > 0 and doc[top[k - 1]].pos_ == "PROPN" and top[k - 1] == top[k] - 1:
                k -= 1
            p = Predication((top[k], top[n - 1] + 1), "appositive", words, sentence=si, named_subject=True,
                            of_complement=after is not None and doc[after].text.lower() == "of")
            out.append(p)
            on_main(p)
        # ---- title: "the American physicist Richard Feynman", "King Charles"
        for n, j in enumerate(top):
            t = doc[j]
            if t.pos_ != "PROPN" or not _is_cap(t):
                continue
            if n == 0 or (doc[top[n - 1]].pos_ == "PROPN" and top[n - 1] == j - 1):
                continue  # not the start of a PROPN run
            m = n
            while m + 1 < len(top) and doc[top[m + 1]].pos_ == "PROPN" and top[m + 1] == top[m] + 1:
                m += 1
            k = n - 1
            mods: List[int] = []
            cap_title = False
            pt = doc[top[k]]
            # a capitalised role word opening a person's name: "King Charles",
            # "Emperor Ashoka", "Commander David Scott" (the store decides
            # whether the word names a kind of person, at settle time)
            if top[k] == j - 1 and _is_cap(pt) and pt.pos_ in ("NOUN", "PROPN", "ADJ") and pt.text.isalpha() \
                    and ner.get(top[k]) != "PERSON" \
                    and any(ner.get(top[x]) == "PERSON" for x in range(n, m + 1)):
                mods, cap_title, k = [top[k]], True, k - 1
            while k >= 0 and top[k] == top[k + 1] - 1 and (
                    (doc[top[k]].pos_ in ("NOUN", "ADJ") and not _is_cap(doc[top[k]]))
                    or ner.get(top[k]) == "NORP"
                    # "the injured star Terrell Davis": a participle before the role noun
                    or (mods and getattr(doc[top[k]], "tag_", "") in ("VBN", "VBD")
                        and not _is_cap(doc[top[k]]) and doc[top[k]].text.isalpha())):
                mods.insert(0, top[k])
                k -= 1
            if not mods or not (cap_title or doc[mods[-1]].pos_ == "NOUN"):
                continue
            words = []
            for q, idx in enumerate(mods):
                tk = doc[idx]
                role = "head" if q == len(mods) - 1 else "mod"
                if role == "head" and cap_title:
                    words.append(Word(idx, tk.text, tk.text.lower(), "NOUN", role, ""))
                elif tk.pos_ == "VERB":
                    words.append(Word(idx, tk.text, tk.text.lower(), "ADJ", role, ner.get(idx, "")))
                else:
                    words.append(Word(idx, tk.text, tk.lemma_.lower(), tk.pos_, role, ner.get(idx, "")))
            name_spans = [(j, top[m] + 1)]
            head = doc[mods[-1]]
            if head.text.lower() != head.lemma_.lower():
                # a plural title is shared by coordinated names:
                # "creators Len Wein, Marv Wolfman, and John Warner"
                q = m + 1
                while q < len(top):
                    w = doc[top[q]].text.lower()
                    if w in (",", "and", "or"):
                        q += 1
                        continue
                    if doc[top[q]].pos_ == "PROPN" and _is_cap(doc[top[q]]) and \
                            doc[top[q - 1]].text.lower() in (",", "and", "or"):
                        r = q
                        while r + 1 < len(top) and doc[top[r + 1]].pos_ == "PROPN" and top[r + 1] == top[r] + 1:
                            r += 1
                        name_spans.append((top[q], top[r] + 1))
                        q = r + 1
                        continue
                    break
            for sp in name_spans:
                out.append(Predication(sp, "title", words, sentence=si, named_subject=True))
                on_main(out[-1])
        # ---- kind of: "the town of Santa Margarita" -------------------------
        for n in range(len(top) - 3):
            d0, k0, of, p0 = (doc[top[n + x]] for x in range(4))
            if d0.text.lower() not in ("the", "a") or of.text.lower() != "of" or k0.pos_ != "NOUN" \
                    or _is_cap(k0) or not (p0.pos_ == "PROPN" and _is_cap(p0)):
                continue
            if top[n] == s.start:
                continue   # sentence-initial "The history of X": a frame, not a kind
            kw = _word(doc, top[n + 1], "head", ner)
            kl = normalise(kw)
            if kl in GENERIC_HEADS or kl in QUANT_HEADS or kl in RELATIONAL_QUANT or kl in DIRECTIONAL:
                continue
            if kl != k0.text.lower():
                continue   # plural: "the ruins of Alinda" names parts of X, not X
            m = n + 3
            while m + 1 < len(top) and doc[top[m + 1]].pos_ == "PROPN" and top[m + 1] == top[m] + 1:
                m += 1
            out.append(Predication((top[n + 3], top[m] + 1), "kind_of", [kw], sentence=si,
                                   needs_kind=kl, named_subject=True))
            on_main(out[-1])
        # ---- kinds such as / like / including names -------------------------
        for n, j in enumerate(top):
            t = doc[j]
            if t.pos_ != "NOUN" or _is_cap(t):
                continue
            q = n + 1
            if q + 1 < len(top) and doc[top[q]].text.lower() == "such" and doc[top[q + 1]].text.lower() == "as":
                q += 2
            elif q < len(top) and doc[top[q]].text.lower() in ("like", "including"):
                q += 1
            else:
                continue
            # plural kinds only: "a coalition including X" names a group X
            # belongs to, "genera like X" says what X is
            if t.text.lower() == t.lemma_.lower() and not t.text.lower().endswith("s"):
                continue
            hw = _word(doc, j, "head", ner)
            kl = normalise(hw)
            if kl in GENERIC_HEADS:
                continue
            a = n
            while a - 1 >= 0 and top[a - 1] == top[a] - 1 and doc[top[a - 1]].pos_ == "ADJ" \
                    and not _is_cap(doc[top[a - 1]]):
                a -= 1
            kind = [_word(doc, top[x], "mod", ner) for x in range(a, n)] + [hw]
            names = []
            while q < len(top):
                w = doc[top[q]]
                if w.text.lower() in (",", "and", "or", "the"):
                    q += 1
                    continue
                if (w.pos_ == "PROPN" or _is_cap(w)) and w.text != "I":
                    r = q
                    while r + 1 < len(top) and top[r + 1] == top[r] + 1 and (
                            doc[top[r + 1]].pos_ in ("PROPN", "NOUN") or _is_cap(doc[top[r + 1]])):
                        r += 1
                    e = top[r] + 1
                    while e < s.end and doc[e].text == "+":
                        e += 1
                    if e < s.end and _is_possessive_at(doc, e, s.end):
                        break       # "media like Disney+'s Werewolf": the possessor is not the instance
                    names.append((top[q], top[r] + 1))
                    q = r + 1
                    continue
                break
            for sp in names:
                out.append(Predication(sp, "hearst", kind, sentence=si, needs_kind=kl, named_subject=True))
                on_main(out[-1])
        # ---- modifier: ADJ run directly before a noun -----------------------
        n = 0
        while n < len(top):
            j = top[n]
            if doc[j].pos_ != "ADJ":
                n += 1
                continue
            mods = [j]
            m = n + 1
            while m < len(top) and top[m] == top[m - 1] + 1 and doc[top[m]].pos_ == "ADJ":
                mods.append(top[m])
                m += 1
            noun_end = m
            while noun_end < len(top) and top[noun_end] == top[noun_end - 1] + 1 \
                    and doc[top[noun_end]].pos_ in NOMINAL_POS:
                if noun_end > m and _is_cap(doc[top[noun_end]]) and not _is_cap(doc[top[noun_end - 1]]):
                    break  # "offensive linemen Fresno State": the name is a new NP
                noun_end += 1
            if noun_end > m:
                head_tok = doc[top[noun_end - 1]]
                name_internal = _is_cap(doc[top[m]])
                # "Central America", "National Park": a capitalised adjective
                # before a capitalised noun is part of a name, not a modifier.
                keep = [idx for idx in mods if not (name_internal and _is_cap(doc[idx]))]
                # "southern Norway", "northwestern South America": a part of
                # the named place, not a property of it
                if name_internal:
                    keep = [idx for idx in keep if doc[idx].text.lower() not in DIRECTIONAL]
                # "External links", "Professional career": section headings
                nouns = [doc[top[q]] for q in range(m, noun_end)]
                generic = any(t.text.lower() in GENERIC_HEADS or t.lemma_.lower() in GENERIC_HEADS
                              for t in (nouns[0], head_tok))
                if keep and not generic:
                    words = [Word(idx, doc[idx].text, _lemma(doc[idx]), "ADJ", "mod",
                                  ner.get(idx, "")) for idx in keep]
                    out.append(Predication((keep[0], top[noun_end - 1] + 1), "modifier", words, sentence=si))
            n = max(noun_end, n + 1)
    out.main = main
    return out


# ---------------------------------------------------------------------------
# Predications -> attribute values
# ---------------------------------------------------------------------------
# The value is the word's own normalised lemma. There is deliberately no
# mapping onto a small class inventory: the axis must keep the cardinality of
# the text ("actor", "american", "large"), or it collapses into a copy of the
# type segment.

_PERSON_NOUNS_PATH = os.path.join(os.path.dirname(__file__), "person_nouns.txt")

# Determiner-like, ordinal and vague evaluative adjectives: they quantify,
# order or praise; they do not describe ("the first president", "several
# towns", "a notable actor").
NON_DESCRIPTIVE_ADJ = frozenset({
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "last", "next", "previous", "following", "other", "another",
    "many", "several", "various", "same", "such", "own", "certain", "few", "more",
    "most", "much", "whole", "entire", "additional", "further", "only", "sole",
    "former", "latter", "late", "then", "current", "present", "future", "past",
    "new", "main", "major", "minor", "principal", "primary", "secondary", "chief",
    "good", "best", "better", "great", "greatest", "notable", "famous", "important",
    "significant", "popular", "successful", "prominent", "well-known", "known",
    "available", "able", "unable", "possible", "due", "unknown", "similar",
    "different", "specific", "particular", "general", "common", "typical", "usual",
    "real", "true", "actual", "recent", "early", "earliest", "key", "top", "total",
    "respective", "individual", "single", "numerous", "responsible", "related",
    "overall", "original", "final", "full", "likely", "unlikely", "necessary",
    "vital", "crucial", "pivotal", "remarkable", "special", "unique", "essential",
    "leading", "renowned", "iconic", "influential",
})
DIRECTIONAL = frozenset({
    "north", "south", "east", "west", "northern", "southern", "eastern", "western",
    "central", "northeastern", "northwestern", "southeastern", "southwestern",
    "north-eastern", "north-western", "south-eastern", "south-western",
    "upper", "lower", "inner", "outer", "mid", "middle", "greater",
})
# Heads that carry no property of their subject: "X is the name of",
# "the term", "a part of", "a person who".
GENERIC_HEADS = frozenset({
    "name", "term", "word", "one", "thing", "part", "way", "example", "kind",
    "type", "form", "sort", "person", "people", "man", "woman", "individual",
    "figure", "link", "links", "home", "use", "time", "year", "life", "career",
    "death", "area", "side", "place", "number", "list", "title", "case", "fact",
    "result", "subject", "topic", "reference", "references", "article", "see",
    "category", "class", "group", "set", "series", "variety", "range", "collection",
    "lot", "majority", "minority", "half", "rest", "amount", "unit", "version",
    "abbreviation", "acronym", "synonym", "spelling", "surname", "nickname",
    "equivalent", "birth", "mid", "feat", "piece",
})
# Quantifying heads that ARE a property of the subject ("a member of").
KEPT_QUANT = frozenset({"member"})
# Link -> confidence. Copular/appositive/title assertions are explicit
# predications; a pronoun link depends on resolving he/she to the main
# subject; a bare modifier is the weakest.
LINK_WEIGHT = {"copular": 0.9, "appositive": 0.85, "title": 0.85, "as_role": 0.85,
               "fronted_as": 0.85, "kind_of": 0.8, "relative": 0.8, "anaphor": 0.75,
               "hearst": 0.75, "pronoun": 0.7, "modifier": 0.6, "query": 0.9}
# Rules whose head noun must name a kind of person ("the physicist Feynman",
# "As mayor of Paris, he"; not "in his life Afonso", "As a result, X").
NEEDS_PERSON_HEAD = frozenset({"title", "fronted_as"})
# Rules whose head noun must NOT name a kind of person: "the king of France"
# does not say France is a king.
NO_ROLE_LINKS = frozenset({"kind_of"})
# A capitalised nationality modifier the rule NER left unlabelled
# ("Scythian prince"): read as NORP outside the bare-modifier rule.
NATIONALITY_SUFFIXES = ("ian", "ean", "ese", "ish", "ic")


@lru_cache(maxsize=1)
def person_nouns() -> frozenset:
    try:
        with open(_PERSON_NOUNS_PATH) as f:
            return frozenset(line.strip() for line in f if line.strip())
    except OSError:
        return frozenset()


_ABSTRACT_NOUNS_PATH = os.path.join(os.path.dirname(__file__), "abstract_nouns.txt")


@lru_cache(maxsize=1)
def abstract_nouns() -> frozenset:
    """The seed of the store's kind flag (kind=False): see abstract_nouns.txt.
    Extraction never reads this file; it asks the store (learning.kind_status)."""
    try:
        with open(_ABSTRACT_NOUNS_PATH) as f:
            return frozenset(w for w in (line.strip() for line in f) if w and not w.startswith("#"))
    except OSError:
        return frozenset()


def normalise(w: Word) -> str:
    """Lowercase value for a word.

    The rule lemmatizer strips the final s of -ss/-us/-is bases ("genus" ->
    "genu", "actress" -> "actres"); those keep their surface form, and their
    plurals ("actresses") drop "es".
    """
    text = w.text.lower()
    lemma = (w.lemma or text).lower()
    if text.endswith("sses"):
        return text[:-2]
    if text.endswith(("ss", "us", "is")) or not lemma:
        return text
    return lemma


def _keep(w: Word, link: str, value: str, is_person=None, min_len: int = 3) -> bool:
    is_person = is_person or (lambda v: v in person_nouns())
    if len(value) < min_len or any(ch.isdigit() for ch in value):
        return False
    if not value.replace("-", "").replace("'", "").isalpha():
        return False
    if w.role == "quant":
        return value in KEPT_QUANT
    if w.pos == "NUM" or value in NON_DESCRIPTIVE_ADJ or value in GENERIC_HEADS:
        return False
    if w.role == "head":
        if w.pos == "PROPN" and w.ner != "NORP":
            return False  # a proper name is a reference, not a property
        if w.text[:1].isupper() and w.pos == "NOUN" and w.ner != "NORP" and link not in ("title", "query") \
                and not is_person(value):
            # "known as the Heart of Asia": a capitalised head names, it does
            # not describe -- unless it is a role ("was Duke of Modena")
            return False
        if link in NEEDS_PERSON_HEAD and not is_person(value) and w.ner != "NORP":
            return False
    elif w.text[:1].isupper() and w.pos != "ADJ" and w.ner != "NORP":
        # a capitalised NOUN/PROPN modifier is part of a name ("Nazi Germany")
        return False
    return True


@dataclass
class AttributeValue:
    value: str
    link: str            # strongest link that produced it
    confidence: float
    anchors: List[Tuple[int, int]] = field(default_factory=list)
    token_indices: List[int] = field(default_factory=list)
    # Evidence kinds (see evidence_kind); the bundler weighs them by the
    # entity's ontology class. They select evidence, never become the value.
    kinds: List[str] = field(default_factory=list)
    sentences: List[int] = field(default_factory=list)   # where it was read
    # every reading: (anchor, link, kind, sentence) -- the head scores and
    # selects per anchor
    occurrences: List[Tuple[Tuple[int, int], str, str, int]] = field(default_factory=list)
    # sketch_subjects only: a pronoun / anaphor subject's own span -> the
    # main-subject span the head's own resolution gives it (where that
    # resolution holds); the head binds through it when NLP2 leaves the
    # subject in no chain (head._bind_chains)
    resolved: Dict[Tuple[int, int], Tuple[int, int]] = field(default_factory=dict)


def evidence_kind(w: Word, value: str, is_person=None) -> str:
    """norp | role_noun | noun | adj | noun_mod -- what sort of evidence a word is."""
    if w.ner == "NORP":
        return "norp"
    if w.role in ("head", "quant"):
        is_person = is_person or (lambda v: v in person_nouns())
        return "role_noun" if is_person(value) else "noun"
    if w.pos == "ADJ" or w.role == "adj":
        return "adj"
    return "noun_mod"


def query_predications(doc) -> List[Predication]:
    """A query's leading noun phrase, read as a predicate.

    The canonicalizer strips interrogatives, articles and a leading copula,
    so "Who was the German architect who served Hitler?" reaches the heads
    as "German architect who served Hitler?". At ingest the same words are
    the predicate of a copular sentence ("Speer was a German architect"),
    so the query's leading NP is parsed exactly as that predicate is, and
    yields the same values (german, architect).
    """
    out: List[Predication] = []
    if doc is None or len(doc) == 0:
        return out
    ner = _ner_map(doc)
    start, end = _sentences(doc)[0]
    s = _Sent(doc, start, end)
    i = s.top[0] if s.top else None
    while i is not None and (doc[i].text.lower() in WH_SUBJECTS or doc[i].text.lower() in COPULA_FORMS):
        i = s.next_top(i)
    words, after = _parse_predicate(doc, s, i, ner)
    # "Mad Love is a film.": an NP followed by a copula is the subject of a
    # copular sentence (read by extract_predications), not a query predicate
    if words and after is not None and _is_copula(doc[after]):
        words = []
    if words:
        last = (after if after is not None else end)
        out.append(Predication((words[0].i, last), "query", words))
    return out


def candidate_values(doc, query: bool = False, title: str = "") -> List[str]:
    """Every normalised word a predication of ``doc`` could settle on, plus
    the title's qualifier -- the keys the head looks up in its store before
    settling."""
    preds = extract_predications(doc, title)
    out = {normalise(w) for p in preds for w in p.words}
    if query:
        out |= {normalise(w) for p in query_predications(doc) for w in p.words}
    if getattr(preds, "qualifier", ""):
        out.add(preds.qualifier)
    return sorted(out)


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _kind_heads(heads: List[Word], is_kind: Optional[Callable[[str], Optional[bool]]]) -> bool:
    """Every head is a lower-case singular common noun the store knows as a
    kind of thing (``is_kind`` True; unknown is not enough)."""
    if not heads or is_kind is None:
        return False
    for w in heads:
        t = w.text.lower()
        if w.pos != "NOUN" or w.text[:1].isupper() or not t.isalpha():
            return False
        if t.endswith("s") and not t.endswith(("ss", "us", "is")):
            return False        # plural: "romance novels J", "genera like X" are other rules
        if is_kind(normalise(w)) is not True:
            return False
    return True


def main_is_person(preds, is_person: Callable[[str], bool]) -> Optional[bool]:
    """Is the document's main subject a person? True / False / None (no
    evidence).

    The title's qualifier decides when there is one ("Yves Rocher
    (company)" -> no, "(musician)" -> yes). Otherwise the heads of what the
    text literally predicates of a title mention (copular, relative,
    appositive, as-role, title): a role noun or a person cue says person, a
    kind noun says not; conflicting evidence says nothing."""
    q = getattr(preds, "qualifier", "")
    if q:
        return bool(is_person(q))
    mentions = list(getattr(preds, "title_spans", ()) or ())
    main = getattr(preds, "main", None)
    if main is not None and main not in mentions:
        mentions.append(main)
    if not mentions:
        return None
    person = other = False
    for p in preds:
        if p.resolved or p.link not in ("copular", "relative", "appositive", "as_role", "title"):
            continue
        if not any(_overlaps(p.anchor, m) for m in mentions):
            continue
        person = person or p.person_cue
        for w in p.words:
            if w.role != "head":
                continue
            v = normalise(w)
            if is_person(v):
                person = True
            elif w.pos == "NOUN" and v not in GENERIC_HEADS:
                other = True
    if person != other:
        return person
    if not person and getattr(preds, "person_cue", False):
        return True         # nothing predicated: a PERSON-labelled title and he/she in the text
    return None


def resolution_holds(p: Predication, main_person: Optional[bool], is_person: Callable[[str], bool],
                     is_kind: Optional[Callable[[str], Optional[bool]]] = None) -> bool:
    """Does a pronoun / anaphor subject refer to the main subject?

    he/she: unless the main subject is known not to be a person; it/this/
    they: unless it is known to be one. A kind-noun anaphor: its noun must
    not be a non-kind in the store (an abstract or relational noun: "this
    story", "the heart"), and must name a kind of person exactly when the
    main subject is one ("the king" is not the battle, "the company" is not
    the founder). A definite anaphor the text does not tie to the main
    subject (``supported``) needs the store to know its noun as a kind."""
    if p.resolved == "pronoun":
        if p.referent in ("he", "she"):
            return main_person is not False
        return main_person is not True
    if p.resolved == "anaphor":
        kind = is_kind(p.referent) if is_kind is not None else None
        if kind is False:
            return False
        role = bool(is_person(p.referent))
        if not p.supported:
            # an unsupported definite: the noun must be a known kind, and
            # a role noun only for a subject known to be a person
            if kind is not True:
                return False
            return main_person is True if role else main_person is not True
        if main_person is None:
            return True
        return role == main_person
    return True


def _as_norp(doc, w: Word, link: str) -> Word:
    """A capitalised nationality modifier the rule NER did not label NORP
    ("a Scythian prince", NER: PERSON) is NORP evidence, except in the
    bare-modifier rule. It must stand directly before a lowercase word: a
    capitalised word after it makes it part of a name ("a Brian Eno album")."""
    if w.role == "mod" and link != "modifier" and w.text[:1].isupper() and w.text.isalpha() \
            and w.ner not in ("NORP", "GPE") and w.text.lower().endswith(NATIONALITY_SUFFIXES) \
            and w.i + 1 < len(doc) and doc[w.i + 1].text[:1].islower():
        return Word(w.i, w.text, w.lemma, "ADJ", w.role, "NORP", w.shared_head)
    return w


def attribute_values(doc, query: bool = False,
                     admit: Optional[Callable[[str, int], bool]] = None,
                     is_person: Optional[Callable[[str], bool]] = None,
                     observe: Optional[Callable[[str, str, "Predication", Word], None]] = None,
                     min_len: int = 3,
                     skip_links: Tuple[str, ...] = (),
                     title: str = "",
                     is_kind: Optional[Callable[[str], Optional[bool]]] = None,
                     sketch_subjects: bool = False,
                     ) -> List[AttributeValue]:
    """Distinct attribute values of a document, each with every subject span
    (anchor) it is predicated of, in order of first occurrence.

    ``query`` adds the leading-NP predicate of a canonicalized query.
    ``admit(kind, sentence_index)`` -- the ontology steer -- may reject a
    kind of evidence for the sentence it was read from; it selects evidence
    and never changes a value.
    ``is_person(value)`` says whether a noun names a kind of person (the
    head answers from its store, seeded with person_nouns.txt and grown by
    learning); default: the seed file alone.
    ``is_kind(noun)`` says whether a noun names a kind of thing (True), is
    an abstract / relational noun (False) or is unknown (None) -- the
    store's kind flag (learning.kind_status); default: unknown for all.
    ``title``: the record title; its mention is the main subject that
    pronouns and kind-noun anaphors resolve to (resolution_holds decides,
    else they stay anchored on themselves).
    ``observe(value, kind, predication, word)`` sees every kept word BEFORE
    the ontology steer, so the store can learn from evidence the steer
    rejects today (an unknown role noun under a Person).
    ``sketch_subjects``: the record carries the resolved sentence sketch,
    whose chains say which participant every pronoun and description
    refers to (NLP2). A pronoun or anaphor predication then stays anchored
    on its own subject span -- the head's main-subject resolution is not
    applied; the chain NLP2 put that mention in decides -- except that an
    anaphor's kind noun the store knows as a non-kind is still no value
    ("this story").
    """
    by_value: Dict[str, AttributeValue] = {}
    if is_person is None:
        persons = person_nouns()
        is_person = persons.__contains__
    doc_preds = extract_predications(doc, title)
    preds = (query_predications(doc) + list(doc_preds)) if query else doc_preds
    main_person = main_is_person(doc_preds, is_person) if any(p.resolved for p in doc_preds) else None
    for p in preds:
        # ``seen``: the predication as the head's own resolution reads it --
        # what the store learns from, with or without the sketch; ``p``: the
        # predication as it binds.
        seen: Optional[Predication] = p
        if p.resolved and p.own is not None and p.anchor != p.own and \
                not resolution_holds(p, main_person, is_person, is_kind):
            # "this story" is not a value of the main subject
            seen = None if p.kind_value else replace(p, anchor=p.own, named_subject=False)
        if sketch_subjects:
            if p.resolved and p.own is not None and p.anchor != p.own:
                if p.kind_value and is_kind is not None and is_kind(p.referent) is False:
                    p = None    # "this story" is no kind
                else:
                    p = replace(p, anchor=p.own, named_subject=False)
        else:
            p = seen
        if p is None:
            if seen is None:
                continue
            p, bind = seen, False   # observed only
        else:
            bind = True
        if p.needs_kind and is_kind is not None and is_kind(p.needs_kind) is False:
            continue            # "the heart of X", "influences such as X"
        if p.link in NEEDS_PERSON_HEAD:
            # the title rule holds when its head names a kind of person;
            # otherwise its modifiers are not about the PROPN either
            # ("romance novels J") -- unless the head is a singular common
            # noun the store knows as a kind of thing: "the genus Huia",
            # "the crime drama film Guilty Bystander" say what the name is,
            # as "the town of Lostwithiel" does (kind_of)
            heads = [w for w in p.words if w.role == "head"]
            if not any(is_person(normalise(w)) or w.ner == "NORP" for w in heads):
                if not (p.link == "title" and _kind_heads(heads, is_kind)):
                    continue
                p = replace(p, link="kind_of", needs_kind=normalise(heads[-1]))
        if p.link in NO_ROLE_LINKS and any(is_person(normalise(w)) for w in p.words if w.role == "head"):
            continue
        for w in p.words:
            value = normalise(w)
            if w.shared_head and w.role == "head" and not is_person(value):
                w = Word(w.i, w.text, w.lemma, w.pos, "mod", w.ner)
            w = _as_norp(doc, w, p.link)
            if w.role == "mod" and w.ner != "NORP" and w.text[:1].isupper() and w.i + 1 < len(doc) \
                    and doc[w.i + 1].text[:1].isupper():
                continue    # "a Royal Air Force station", "a Rear Admiral": part of a name
            if not _keep(w, p.link, value, is_person, min_len):
                continue
            kind = evidence_kind(w, value, is_person)
            if observe is not None and seen is not None:
                observe(value, kind, seen, w)
            if p.link in skip_links or not bind:
                continue   # observed (learning), but settled by another source
            if admit is not None and not admit(kind, p.sentence):
                continue
            conf = LINK_WEIGHT.get(p.link, 0.5)
            av = by_value.get(value)
            if av is None:
                av = by_value[value] = AttributeValue(value, p.link, conf)
            elif conf > av.confidence:
                av.link, av.confidence = p.link, conf
            if p.anchor not in av.anchors:
                av.anchors.append(p.anchor)
            if sketch_subjects and seen is not None and seen.anchor != p.anchor:
                av.resolved.setdefault(p.anchor, seen.anchor)
            if w.i not in av.token_indices:
                av.token_indices.append(w.i)
            if kind not in av.kinds:
                av.kinds.append(kind)
            occ = (p.anchor, p.link, kind, p.sentence)
            if occ not in av.occurrences:
                av.occurrences.append(occ)
            if p.sentence not in av.sentences:
                av.sentences.append(p.sentence)
    return list(by_value.values())


def legacy_anchors(doc, concepts, canon=lambda v: v, title: str = "") -> Dict[str, List[List[int]]]:
    """Anchors for values the legacy extraction selected.

    A legacy adjective value is anchored on the noun phrase it directly
    precedes ("large snakes", "Permanent anchors") -- the modifier
    predications -- unless the structural junk rules reject it there
    (name-internal capitals, directional parts of places, section headings).
    Values with no such reading get no anchor and link through the
    bundler's dependency walk only, as before.
    """
    want = set(concepts)
    out: Dict[str, List[List[int]]] = {}
    if doc is None or not want:
        return out
    for p in extract_predications(doc, title):
        if p.link != "modifier":
            continue
        for w in p.words:
            for v in {canon(normalise(w)), canon((w.lemma or "").lower()), canon(w.text.lower())}:
                if v in want:
                    spans = out.setdefault(v, [])
                    if list(p.anchor) not in spans:
                        spans.append(list(p.anchor))
    return out

