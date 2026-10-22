"""Geographic context: the places a sentence names, normalized by gazetteer.

Value: the place's containment path (continent / country / US-or-GB
subdivision / city), e.g. "NA/US/US-IL/gn4887398" for Chicago, so a query at
country level is a prefix of every place inside it. See
``data/generate_places.py`` for the path scheme and sources.

The rule NER labels a span GPE only when it is in its 490-entry gazetteer;
every other capitalized run becomes PERSON (<= 3 tokens) or ORG, and it splits
"New Jersey" and "North Carolina" (ner.py:386-406). So places are matched
here directly on the tokens, longest window first, with the NER label used
only as one piece of evidence:

* continents, regions, countries, states/provinces, historical countries --
  accepted wherever they are written capitalized, unless they sit inside a
  person's name ("George Washington", "Jack London");
* cities and ambiguous-word countries ("Jersey", "Chad") -- accepted only
  with evidence that a place is meant: the rule NER called the span GPE, a
  locative word precedes it ("in", "at", "near", "from"...), or it is followed
  by ", <its state or country>".

A surface naming several places is resolved toward the places the same
document names unambiguously (shared path prefix); a city name with no such
help is taken only when its largest namesake is at least twice the next.
Demonyms ("American", "Soviet") resolve to the place they name.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DATA = Path(__file__).parent / "data" / "places.json"
_FIRST_NAMES = Path(__file__).parent / "data" / "gazetteers" / "person_first.txt"

MAX_WINDOW = 6
CONNECTORS = frozenset({"of", "and", "the", "de", "-", "upon", "on", "la", "del", "da"})
LOCATIVES = frozenset({"in", "at", "near", "from", "outside", "around", "across",
                       "throughout", "within", "between", "towards", "toward", "into"})
TITLES = frozenset({"mr", "mrs", "ms", "dr", "sir", "dame", "lord", "lady", "king",
                    "queen", "prince", "princess", "saint", "st", "st.", "president",
                    "general", "governor", "senator", "duke", "earl", "count", "baron",
                    "emperor", "pope", "bishop", "captain", "colonel", "admiral", "fort",
                    "mount", "lake", "cape"})
CUE_TIERS = frozenset({"city", "country_cue"})
# "<Peerage> of <Place>" names a person by a title ("the Earl of Essex"); the
# place in it, and its bare recurrences ("Essex managed to escape"), are not
# where anything happens. A locative word still makes a recurrence a place.
PEERAGE = frozenset({"earl", "earls", "duke", "dukes", "count", "counts", "baron", "barons",
                     "marquess", "marquis", "viscount", "countess", "duchess", "baroness",
                     "margrave", "landgrave"})


@lru_cache(maxsize=1)
def gazetteer() -> Tuple[Dict[str, list], Dict[str, list]]:
    """SEED SOURCE ONLY (seed.py). Extraction reads places from the store."""
    doc = json.loads(DATA.read_text())
    return doc["places"], doc["demonyms"]


@lru_cache(maxsize=1)
def first_names() -> frozenset:
    try:
        return frozenset(l.strip().lower() for l in _FIRST_NAMES.read_text().splitlines() if l.strip())
    except OSError:
        return frozenset()


def _cap(t) -> bool:
    s = t.text
    return bool(s) and s[0].isupper()


def _window_key(toks, i: int, j: int) -> Optional[str]:
    ws = toks[i:j]
    if not _cap(ws[0]) or not _cap(ws[-1]):
        return None
    for w in ws[1:-1]:
        if not _cap(w) and w.text.lower() not in CONNECTORS:
            return None
    # Re-join "Baden - Württemberg" style hyphen tokens without spaces.
    out = []
    for w in ws:
        if w.text == "-" and out:
            out[-1] = out[-1] + "-"
        elif out and out[-1].endswith("-"):
            out[-1] = out[-1] + w.text
        else:
            out.append(w.text)
    return " ".join(out).lower()


# windows() of the last few token lists, per start index: the context head
# asks for the same windows twice per document (lexicon.candidate_queries_and_
# keys, then extract). Keyed by list identity (the entry holds the list).
_WIN_MEMO: Dict[int, tuple] = {}


def windows(toks, i: int, max_len: int = MAX_WINDOW):
    """(end, key) for every valid window starting at i, shortest first, built
    incrementally -- same keys as ``_window_key`` at a fraction of the cost.
    The returned list may be shared: do not modify it."""
    ent = _WIN_MEMO.get(id(toks))
    if ent is None or ent[0] is not toks:
        if len(_WIN_MEMO) >= 64:
            _WIN_MEMO.clear()
        ent = (toks, {})
        _WIN_MEMO[id(toks)] = ent
    memo = ent[1]
    hit = memo.get((i, max_len))
    if hit is None:
        hit = memo[(i, max_len)] = _windows(toks, i, max_len)
    return hit


def _windows(toks, i: int, max_len: int):
    n = len(toks)
    first = toks[i].text
    if not first or not first[0].isupper():
        return []
    out = []
    parts = [first]
    if first != "-":
        out.append((i + 1, first.lower()))
    for j in range(i + 1, min(n, i + max_len)):
        t = toks[j].text
        cap = bool(t) and t[0].isupper()
        if not cap and t.lower() not in CONNECTORS:
            break
        if t == "-":
            parts[-1] = parts[-1] + "-"
        elif parts[-1].endswith("-"):
            parts[-1] = parts[-1] + t
        else:
            parts.append(t)
        if cap:
            out.append((j + 1, " ".join(parts).lower()))
    return out


class PlaceMention:
    __slots__ = ("start", "end", "surface", "cands", "source", "path", "tier")

    def __init__(self, start, end, surface, cands, source):
        self.start, self.end, self.surface, self.cands, self.source = start, end, surface, cands, source
        self.path = self.tier = None


def _inside_person_name(toks, start: int, end: int, person_tokens: Set[int]) -> bool:
    if start > 0:
        p = toks[start - 1]
        pl = p.text.lower().rstrip(".")
        if _cap(p) and (pl in first_names() or pl in TITLES):
            return True
        if pl in ("fort", "mount", "lake", "cape", "port"):
            return True
        # A one-word place right after another proper noun is the tail of a
        # longer name ("George Washington", "Jack London"); the gazetteer
        # already tried the whole window, so the pair is not a place.
        if end - start == 1 and _cap(p) and p.pos_ == "PROPN" and not p.text.isupper():
            return True
    # "Washington" when the doc has a PERSON span ending "... Washington"
    return (end - start == 1) and start in person_tokens


def _locative(toks, start: int) -> bool:
    return start > 0 and toks[start - 1].text.lower() in LOCATIVES


def _peerage_names(toks) -> Set[str]:
    """Window keys that follow "<Peerage> of" anywhere in the doc ("Earl of
    Essex" -> "essex"): names used as a person's title."""
    out: Set[str] = set()
    for i in range(2, len(toks)):
        if toks[i - 1].text.lower() == "of" and _cap(toks[i - 2]) and \
                toks[i - 2].text.lower() in PEERAGE and _cap(toks[i]):
            out.update(k for _, k in windows(toks, i))
    return out


def _has_cue(toks, start: int, end: int, gpe_tokens: Set[int], places: Dict) -> bool:
    if any(k in gpe_tokens for k in range(start, end)):
        return True
    if start > 0 and toks[start - 1].text.lower() in LOCATIVES:
        return True
    if start > 1 and toks[start - 1].text.lower() == "of" and \
            toks[start - 2].text.lower() in ("city", "town", "village", "mayor", "suburb",
                                            "borough", "municipality", "capital", "county"):
        return True
    return False


def _followed_by_container(toks, end: int, places: Dict) -> Optional[str]:
    """", Illinois" after a city: return the container's path."""
    if end + 1 < len(toks) and toks[end].text == ",":
        for ln in range(min(4, len(toks) - end - 1), 0, -1):
            k = _window_key(toks, end + 1, end + 1 + ln)
            if k and k in places:
                c = places[k][0]
                if c[1] != "city":
                    return c[0]
    return None


def _new_place(toks, i: int, n: int, places: Dict, gpe_tokens: Set[int]):
    """An unknown capitalized name followed by ", <known state/country>" and
    preceded by a locative word or GPE label: a place inside that container
    the gazetteer does not list ("born in Waunakee, Wisconsin").
    Returns (end, surface, path) or None."""
    if not (_locative(toks, i) or i in gpe_tokens):
        return None
    # The tail of a longer capitalized name ("Quebec City") is not a name.
    if i > 0 and _cap(toks[i - 1]):
        return None
    for j in range(min(n, i + 3), i, -1):
        k = _window_key(toks, i, j)
        if not k or k in places:
            continue
        container = _followed_by_container(toks, j, places)
        if container:
            slug = "n-" + "".join(ch if ch.isalnum() else "_" for ch in k).strip("_")
            return j, k, f"{container}/{slug}"
    return None


def extract(toks, sents, ents, lex, claimed: Optional[Set[int]] = None,
            obs: Optional[list] = None) -> List[PlaceMention]:
    """``lex``: a ``lexicon.Lexicon`` (store-backed tables for this batch).
    ``obs``: when a list, observations to learn are appended to it as
    (kind, surface, value) -- see ``lexicon.LEARN_KINDS``."""
    places, demonyms = lex.places, lex.demonyms
    claimed = set(claimed or ())
    # Place labels from the NER (the rule NER emits GPE; other backends LOC, FAC).
    gpe_tokens = {k for e in ents if e.label in ("GPE", "LOC", "FAC") for k in range(e.start, e.end)}
    # Last token of every multi-token PERSON span: a surname, not a place,
    # wherever it recurs alone in the doc ("Lincoln" after "Abraham Lincoln").
    surnames = {toks[e.end - 1].text.lower() for e in ents
                if e.label == "PERSON" and e.end - e.start >= 2}
    person_tokens = {i for i, t in enumerate(toks) if t.text.lower() in surnames}
    sent_starts = {s for s, _ in sents}
    peerage = _peerage_names(toks)

    mentions: List[PlaceMention] = []
    n = len(toks)
    i = 0
    while i < n:
        if i in claimed or not _cap(toks[i]):
            i += 1
            continue
        hit = None
        for j, k in reversed(windows(toks, i)):
            if claimed and any(x in claimed for x in range(i, j)):
                continue
            if k in places:
                hit = ("place", j, k)
                break
            if k in demonyms:
                hit = ("demonym", j, k)
                break
        if not hit:
            new = _new_place(toks, i, n, places, gpe_tokens)
            if new and not any(k in claimed for k in range(i, new[0])):
                j, key, path = new
                m = PlaceMention(i, j, key, [[path, "city", 0]], "comma_container")
                m.path, m.tier = path, "city"
                mentions.append(m)
                if obs is not None:
                    obs.append(("place", key, path))
                i = j
                continue
            i += 1
            continue
        kind, j, key = hit
        if kind == "demonym":
            # Inside a longer name ("Central European Media Enterprises") a
            # demonym is part of the name, not a statement of origin.
            if i > 0 and _cap(toks[i - 1]) and i not in sent_starts and \
                    toks[i - 1].text.lower() not in demonyms:
                i = j
                continue
            path, tier = demonyms[key]
            m = PlaceMention(i, j, key, [[path, tier, 0]], "norp_demonym")
            m.path, m.tier = path, tier
            mentions.append(m)
            i = j
            continue
        cands = places[key]
        if _inside_person_name(toks, i, j, person_tokens if j - i == 1 else set()) and \
                not (j - i == 1 and _followed_by_container(toks, j, places)):
            i = j
            continue
        if key in peerage and not _locative(toks, i):
            i = j
            continue
        container = _followed_by_container(toks, j, places)
        if container:
            inside = [c for c in cands if c[0].startswith(container + "/")]
            if inside:
                # ", Virginia" settled an ambiguous name: learn which place it was.
                if obs is not None and len(inside) == 1 and len({c[0] for c in cands}) > 1:
                    obs.append(("amb", key, inside[0][0]))
                cands = inside
        cue = container is not None or _has_cue(toks, i, j, gpe_tokens, places)
        # Single-token match at a sentence start that is not a GPE: a sentence
        # opener is capitalized by position, not because it names a place.
        if not cue and i in sent_starts and j - i == 1 and toks[i].pos_ != "PROPN":
            i = j
            continue
        # A one-word city that is also a first name ("Gilbert", "Warren")
        # needs a locative word; the rule NER's GPE label is not enough.
        if j - i == 1 and key in first_names() and not _locative(toks, i):
            cands = [c for c in cands if c[1] not in CUE_TIERS]
        # A city name followed by another proper noun is the head of a longer
        # name ("Clay County", "Belle Haven Consultants"), not the city.
        # (The rule tagger often tags "County", "Haven" NOUN: casing decides.)
        if container is None and j < n and _cap(toks[j]) and not toks[j].text.isupper():
            cands = [c for c in cands if c[1] not in CUE_TIERS]
        usable = [c for c in cands if c[1] not in CUE_TIERS or cue]
        if not usable:
            i = j
            continue
        source = "ner_gpe" if any(k in gpe_tokens for k in range(i, j)) else "gazetteer"
        mentions.append(PlaceMention(i, j, key, usable, source))
        i = j

    _resolve(mentions, lex.learned_amb, obs)
    return [m for m in mentions if m.path]


def _common_prefix(a: str, b: str) -> int:
    pa, pb = a.split("/"), b.split("/")
    k = 0
    while k < min(len(pa), len(pb)) and pa[k] == pb[k]:
        k += 1
    return k


def _resolve(mentions: List[PlaceMention], learned_amb: Optional[Dict[str, str]] = None,
             obs: Optional[list] = None) -> None:
    anchors = []
    for m in mentions:
        if m.path:
            anchors.append(m.path)
        elif len(m.cands) == 1 or m.cands[0][1] != "city":
            # One candidate, or a state/country that outranks its namesake
            # cities ("Virginia"): settled, and an anchor for the rest.
            m.path, m.tier = m.cands[0][0], m.cands[0][1]
            anchors.append(m.path)
    for m in mentions:
        if m.path:
            continue
        others = [a for a in anchors]
        scored = [(max((_common_prefix(c[0], a) for a in others), default=0), c) for c in m.cands]
        best = max(s for s, _ in scored)
        if best > 0:
            top = [c for s, c in scored if s == best]
            c = top[0]
            # The document settled an ambiguous name: learn which place it was.
            if obs is not None and len({x[0] for x in m.cands}) > 1 and \
                    sum(1 for s, _ in scored if s == best) == 1:
                obs.append(("amb", m.surface, c[0]))
        elif learned_amb and learned_amb.get(m.surface) in {x[0] for x in m.cands}:
            # Nothing in this document settles it; other documents did.
            path = learned_amb[m.surface]
            c = next(x for x in m.cands if x[0] == path)
        else:
            c = m.cands[0]
            nxt = m.cands[1] if len(m.cands) > 1 else None
            if c[1] == "city" and nxt is not None and nxt[1] == "city" and c[2] < 2 * nxt[2]:
                continue  # ambiguous city name, nothing in the doc to settle it
        m.path, m.tier = c[0], c[1]
