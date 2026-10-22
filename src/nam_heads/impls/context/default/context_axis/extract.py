"""Context extraction: one function, used by ingest, query, batch and single.

``extract_context(doc)`` returns every context value the text supports, each
tied to the sentence it was found in, for the bundler to scope to the
entities of that sentence. A value is the PARTICULAR the text names -- a
period, a place, an institution, a field -- prefixed by its dimension:

    t:C19/D1860/Y1865        temporal   (century / decade / year)
    g:NA/US/US-IL/gn4887398  geographic (continent / country / state / city)
    i:university_of_michigan institutional
    d:american_football      domain

Each named place also yields its CONTAINER -- the parent of its path
("g:EU/GB/GB-ENG" for Cornwall) -- carrying ``containers`` (the places it
contains here) and no mention span of its own unless the text also names it.
The bundler gives a container only to the place it contains: that place's
context is where it is, and no mention of the container may be in the text.

It never derives a value from the entity's type and never emits a default.
Its tables are the context store's contents for this batch (``lexicon.py``):
the same parse against the same store state yields the same values, and
ingest and query read the same store.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, List, Optional, Sequence

from . import doc as D
from . import domains, institutions, places, temporal

TEMPORAL, GEOGRAPHIC, INSTITUTIONAL, DOMAIN = "temporal", "geographic", "institutional", "domain"
DIMENSIONS = (TEMPORAL, GEOGRAPHIC, INSTITUTIONAL, DOMAIN)
PREFIX = {TEMPORAL: "t", GEOGRAPHIC: "g", INSTITUTIONAL: "i", DOMAIN: "d"}


@dataclass
class ContextValue:
    dimension: str
    value: str                     # "t:C19/D1860/Y1865"
    surface: str                   # text it was read from
    sentences: List[int] = field(default_factory=list)
    spans: List[tuple] = field(default_factory=list)   # (start, end) token spans
    source: str = ""
    kind: str = ""                 # institutional: the head word's legacy kind (metadata only)
    # geographic: the named places ("g:" values) this value is the container
    # of; their mentions are not spans of this value.
    containers: List[str] = field(default_factory=list)
    # geographic: the spans read from a demonym ("American"), a subset of
    # ``spans`` -- an adjective of nationality or origin, not a named place.
    demonym_spans: List[tuple] = field(default_factory=list)


# Lowercase particles of personal names, and the saint prefixes that open
# place and mission names using the same particles.
NAME_PARTICLES = frozenset({"de", "da", "di", "del", "della", "do", "dos", "das", "van", "von",
                            "der", "den", "du", "le", "la", "bin", "ibn", "al", "ben"})
SAINT_PREFIXES = frozenset({"San", "Santa", "Santo", "São", "Saint", "St", "St.", "Sainte",
                            "Sankt", "Mission"})


# Longest z value, in characters (ASCII, so bytes). Longer values keep their
# first 39 characters and a hash of the whole: still deterministic and
# distinct, and bounded for the 250-byte key cap (lexicon/places paths are
# <= 30 chars; only organization names and minted place slugs reach it).
MAX_VALUE_CHARS = 48


def cap_value(value: str) -> str:
    if len(value) <= MAX_VALUE_CHARS:
        return value
    import hashlib
    return value[:39] + "~" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def extract_context(doc, lex, dimensions: Sequence[str] = DIMENSIONS,
                    sentence_dimensions: Optional[Callable[[int], FrozenSet[str]]] = None,
                    obs: Optional[list] = None, toks: Optional[list] = None,
                    sents: Optional[list] = None, ents: Optional[list] = None) -> List[ContextValue]:
    """``lex``: the batch's ``lexicon.Lexicon`` (store contents). ``obs``: a
    list to receive learning observations (kind, surface, value); pass it
    only with a seed-only lexicon (``Lexicon.seed_only``).

    ``sentence_dimensions(i)`` -- when given -- names the dimensions that
    apply in sentence i (from that sentence's ontology profile, see
    ``profiles.sentence_dimensions``); mentions of other dimensions in that
    sentence are not extracted."""
    # ``toks`` / ``sents`` / ``ents``: list(doc), D.sentences, D.entities of
    # this doc, when the caller already has them (read-only here).
    if toks is None:
        toks = D.tokens(doc)
    if not toks:
        return []
    if sents is None:
        sents = D.sentences(doc, len(toks))
    if ents is None:
        ents = D.entities(doc, toks)
    found = []  # (dimension, start, end, value, source)
    contained = []  # (start, container path, own path): the parent of each named place

    # Tokens of named periods and places: their words name no field.
    name_tokens = set()
    if TEMPORAL in dimensions or DOMAIN in dimensions:
        for s, e, v, src in temporal.extract(toks, [x for x in ents if x.label in ("DATE", "TIME")], lex,
                                             sents=sents,
                                             event_spans=[x for x in ents if x.label == "EVENT"]):
            if TEMPORAL in dimensions:
                found.append((TEMPORAL, s, e, v, src))
            if src in ("named_period", "named_war", "named_dynasty"):
                name_tokens.update(range(s, e))
    place_tokens = set()
    if GEOGRAPHIC in dimensions or INSTITUTIONAL in dimensions or DOMAIN in dimensions:
        for m in places.extract(toks, sents, ents, lex, obs=obs):
            place_tokens.update(range(m.start, m.end))
            if m.source != "norp_demonym":
                name_tokens.update(range(m.start, m.end))
            if GEOGRAPHIC in dimensions:
                found.append((GEOGRAPHIC, m.start, m.end, m.path, m.source))
                if m.source != "norp_demonym" and "/" in m.path:
                    contained.append((m.start, m.path.rsplit("/", 1)[0], m.path))
    if INSTITUTIONAL in dimensions:
        orgs = institutions.extract(toks, sents, lex, claimed=place_tokens, obs=obs)
        org_tokens = {k for s, e, _, _ in orgs for k in range(s, e)}
        # An ORG span the NER found that no head word or gazetteer named
        # (backends with a statistical NER; the rule NER's ORG is a 4+ token
        # proper-noun run or a corporate suffix): its own name is the value.
        for sp in ents:
            if sp.label != "ORG" or not (2 <= sp.end - sp.start <= 6):
                continue
            words = [t.text for t in toks[sp.start:sp.end]]
            if not all(w[:1].isupper() or w in institutions.CONNECTORS for w in words):
                continue
            # A person's name is no organization (the rule NER calls every 4+
            # token proper-noun run ORG): a known first name, or a name
            # particle inside the run ("Juan Ponce de León", "Ludwig van
            # Beethoven") unless it opens with a saint's name, as missions and
            # places do ("San Luis Obispo de Tolosa").
            first = words[1] if words[0] in institutions.LEADING_DROP and len(words) > 1 else words[0]
            if first.lower() in places.first_names():
                continue
            if any(w in NAME_PARTICLES for w in words[1:-1]) and first not in SAINT_PREFIXES:
                continue
            if any(k in org_tokens or k in place_tokens for k in range(sp.start, sp.end)):
                continue
            orgs.append((sp.start, sp.end, institutions.slug(" ".join(words)), "ner_org"))
        for s, e, v, src in orgs:
            if v:
                found.append((INSTITUTIONAL, s, e, v, src))
    if DOMAIN in dimensions:
        for s, e, v, src in domains.extract(toks, sents, lex, ents, claimed=name_tokens):
            found.append((DOMAIN, s, e, v, src))

    sentence_of = D.sentence_lookup(sents)
    if sentence_dimensions is not None:
        found = [f for f in found if f[0] in sentence_dimensions(sentence_of(f[1]))]
        contained = [c for c in contained if GEOGRAPHIC in sentence_dimensions(sentence_of(c[0]))]

    merged: Dict[str, ContextValue] = {}
    # Mentions and containers in text order (a container sorts after the
    # mention it is the parent of), so the value order is the text's.
    items = [(s, dim, 0, e, v, src) for dim, s, e, v, src in found] + \
            [(s, GEOGRAPHIC, 1, None, parent, own) for s, parent, own in contained]
    for s, dim, is_container, e, v, src in sorted(items, key=lambda x: (x[0], x[1], x[2])):
        key = cap_value(f"{PREFIX[dim]}:{v}")
        cv = merged.get(key)
        if is_container:
            if cv is None:
                cv = merged[key] = ContextValue(dim, key, "", source="place_container")
            own = cap_value(f"{PREFIX[GEOGRAPHIC]}:{src}")
            if own not in cv.containers:
                cv.containers.append(own)
            continue
        if cv is None:
            kind = ""
            if dim == INSTITUTIONAL:
                kind = next((lex.head_kind[t.text] for t in toks[s:e] if t.text in lex.head_kind), "")
            cv = merged[key] = ContextValue(dim, key, " ".join(t.text for t in toks[s:e]),
                                            source=src, kind=kind)
        elif not cv.spans:
            # first mention of a value seen so far only as a container
            cv.surface, cv.source = " ".join(t.text for t in toks[s:e]), src
        si = sentence_of(s)
        if si not in cv.sentences:
            cv.sentences.append(si)
        cv.spans.append((s, e))
        if src == "norp_demonym":
            cv.demonym_spans.append((s, e))
    return list(merged.values())
