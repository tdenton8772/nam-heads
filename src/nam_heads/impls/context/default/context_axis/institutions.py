"""Institutional context: the organizations a sentence names.

Value: the organization's normalized name ("university_of_michigan",
"bbc"), i.e. the particular institution, never its kind.

Evidence, on the rule backend's tokens:

* a capitalized run (internal "of", "and", "the", "for", "de" allowed)
  containing an institutional head word -- University, Army, Parliament,
  Party, Church, Records, Hospital... ("American Baptist Church",
  "University of Michigan", "Royal Air Force"); the rule NER splits most of
  these, so runs are rebuilt from tokens;
* an exact, CASE-SENSITIVE hit in the rule backend's own organization
  gazetteer (organizations.txt). Its matcher lowercases spans
  (ner.py:313-320), so it labels "who", "time", "science", "care" ORG; here
  a single-word entry counts only when written as an acronym (>= 2 capitals)
  or listed in UNAMBIGUOUS.

Nothing is emitted for an ORG span with no head word and no gazetteer hit:
an unknown organization has no kind and no default.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_ORGS = Path(__file__).parent / "data" / "gazetteers" / "organizations.txt"

HEAD_WORDS = frozenset("""
University College School Academy Institute Polytechnic Seminary Conservatory
Conservatoire Gymnasium Lyceum
Army Navy Force Corps Marines Regiment Brigade Battalion Fleet Militia Legion
Guard Squadron
Parliament Congress Senate Assembly Ministry Department Agency Bureau
Commission Council Court Government Cabinet Police Administration Office Duma
Party Union Association Society Foundation Federation League Club Church
Diocese Archdiocese Order Committee Trust Confederation Alliance Movement
Brotherhood Synod
Company Corporation Inc Inc. Ltd Ltd. LLC plc Records Studios Studio Pictures
Airlines Airways Bank Motors Industries Railway Railways Railroad Publishing
Press Entertainment Group Holdings Enterprises Productions Broadcasting
Network Television Radio Newspaper Magazine Times Herald Post Gazette Journal
Hospital Museum Library Gallery Orchestra Opera Theatre Theater Observatory
Zoo FC F.C. AFC United City Rovers Wanderers Athletic
Center Centre Laboratory Laboratories Service Authority Board Institution
""".split())
# Things named after an organization that are not one: dropped from the end
# of a name ("National Football League Draft", "Academy Award").
NOT_ORG_TAIL = frozenset("""
Draft Award Awards Prize Prizes Cup Championship Championships Trophy Medal
Tournament Series Season Games Building Stadium Act Bill Hall Square Station
Road Street Line Building Report Album
""".split())
# Military units a number designates ("No. 57 Squadron", "1st Battalion"):
# the number is part of the name; without one the unit is no particular.
NUMBERED_UNITS = frozenset({"Squadron", "Regiment", "Brigade", "Battalion", "Flotilla"})
_DESIGNATOR = re.compile(r"^\d+(?:st|nd|rd|th)?$", re.I)
# Head words that also end ordinary capitalized phrases; they count only with
# at least one other capitalized word before them in the run.
WEAK_HEADS = frozenset({"United", "City", "Athletic", "Times", "Post", "Journal",
                        "Order", "Office", "Court", "Network", "Group", "Press",
                        "Television", "Radio", "Force", "Guard", "Movement",
                        "Union", "Alliance", "Theatre", "Theater", "Gallery",
                        # from the current head's INSTITUTIONAL_KEYWORDS
                        "Team", "Defense", "Military", "Broadcast", "Faculty"})
CONNECTORS = frozenset({"of", "and", "the", "for", "de", "&", "du", "des", "la", "von", "van"})
LEADING_DROP = frozenset({"The", "In", "At", "From", "On", "By", "For", "With", "A", "An",
                          "His", "Her", "Their", "Its", "This", "That", "After", "During"})
UNAMBIGUOUS = frozenset({"Harvard", "Stanford", "Yale", "Princeton", "Cornell", "Dartmouth",
                         "Georgetown", "Caltech", "Microsoft", "Google", "Netflix", "Samsung",
                         "Toyota", "Boeing", "Airbus", "Interpol", "Greenpeace", "Nintendo",
                         "Siemens", "Volkswagen", "Huawei", "Nvidia", "Pfizer",
                         "Reuters", "Bundesliga", "Walmart", "Starbucks", "Facebook",
                         "Twitter", "Honda", "Sony", "McGill", "Sorbonne"})


@lru_cache(maxsize=1)
def org_gazetteer() -> dict:
    """SEED SOURCE ONLY (seed.py): exact-case surface tuple -> name, from the
    rule backend's list. Extraction reads organizations from the store."""
    out = {}
    try:
        lines = [l.strip() for l in _ORGS.read_text().splitlines() if l.strip()]
    except OSError:
        return out
    for name in lines:
        words = tuple(name.split())
        if len(words) == 1 and sum(c.isupper() for c in name) < 2 and name not in UNAMBIGUOUS:
            continue
        out[words] = name
    return out


# Letters NFKD does not decompose into an ASCII base letter.
_FOLD = str.maketrans({"ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe",
                       "Œ": "OE", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ł": "l",
                       "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "ħ": "h", "Ħ": "H"})


def slug(name: str) -> str:
    """Lowercase name, accents folded to their base letter ("León" -> "leon";
    the old ASCII-only rule cut "le_n"), every other run of non-letters and
    non-digits -> "_". Letters with no ASCII base (other scripts) are kept."""
    import unicodedata
    s = re.sub(r"'s\b", "", name).translate(_FOLD)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    out, gap = [], False
    for ch in s:
        if ch.isalnum():
            if gap and out:
                out.append("_")
            out.append(ch.lower())
            gap = False
        else:
            gap = True
    return "".join(out)


def _cap(t) -> bool:
    return bool(t.text) and (t.text[0].isupper() or t.text[0].isdigit() and t.text.isupper())


def _runs(toks, start: int, end: int) -> List[Tuple[int, int]]:
    """Maximal capitalized runs in toks[start:end], connectors allowed inside."""
    runs, i = [], start
    while i < end:
        if not _cap(toks[i]):
            i += 1
            continue
        j = i + 1
        while j < end:
            if _cap(toks[j]):
                j += 1
            elif toks[j].text.lower() in CONNECTORS and j + 1 < end and _cap(toks[j + 1]):
                j += 2
            else:
                break
        runs.append((i, j))
        i = j
    return runs


def _is_head(words, k, heads=HEAD_WORDS) -> bool:
    w = words[k]
    return w in heads and (w not in WEAK_HEADS or k >= 1)


def _named_orgs(toks, i: int, j: int, heads=HEAD_WORDS) -> List[Tuple[int, int]]:
    """Organization names inside the capitalized run toks[i:j].

    The run is cut at "and"/"&": a piece with no head word is a complement of
    the previous named piece ("Bank for Reconstruction and Development") or,
    with none before it, a separate name that is not an organization
    ("Jacques Piccard and US Navy"). In each named piece, words after the
    last head word are dropped unless an "of"/"for" complement follows it
    ("National Football League Draft" -> "National Football League"), and a
    head-less "X of" prefix is dropped ("Master of University College").
    """
    pieces, a = [], i
    for k in range(i, j + 1):
        if k == j or toks[k].text in ("and", "&"):
            if k > a:
                pieces.append([a, k])
            a = k + 1
    from .places import first_names
    named, pending = [], None
    for a, b in pieces:
        words = [t.text for t in toks[a:b]]
        if any(_is_head(words, k, heads) for k in range(len(words))):
            # A head-less piece just before joins this name ("American
            # Telephone and Telegraph Company") unless it opens with a
            # person's first name ("Jacques Piccard and US Navy").
            named.append([pending if pending is not None else a, b])
            pending = None
        elif named:
            named[-1][1] = b
        elif words[0].lower() not in first_names():
            pending = a
    out = []
    for a, b in named:
        words = [t.text for t in toks[a:b]]
        stripped = False
        while len(words) > 1 and words[-1] in NOT_ORG_TAIL:
            b, words, stripped = b - 1, words[:-1], True
        if stripped and words[-1] in WEAK_HEADS:
            # "Los Angeles City Hall", "Union Station": the weak head word
            # belongs to the named thing, which is not an organization.
            b, words = b - 1, words[:-1]
        hk = [k for k in range(len(words)) if _is_head(words, k, heads)]
        if not hk:
            continue
        if "of" in words:
            o = words.index("of")
            if o < hk[0] and o + 1 < len(words):
                a, words = a + o + 1, words[o + 1:]
        if words[0] in NUMBERED_UNITS:
            if a > 0 and _DESIGNATOR.match(toks[a - 1].text):
                a -= 1
                if a > 0 and toks[a - 1].text.rstrip(".").lower() == "no":
                    a -= 1
                elif a > 1 and toks[a - 1].text == "." and toks[a - 2].text.lower() == "no":
                    a -= 2
                out.append((a, b))
            continue
        if len(words) >= 2:
            out.append((a, b))
    return out


_ACRONYM = re.compile(r"^[A-Z][A-Z0-9&]{1,7}$")


_FIRST_LENS: Dict[int, tuple] = {}


def _first_word_lengths(gaz) -> Dict[str, Tuple[int, ...]]:
    """first word -> the token lengths of the names that start with it,
    longest first, for a dict or a ChainMap of dicts. Each dict's index is
    cached by identity and size (the seed gazetteer is immutable; it was
    scanned for its longest name on every document)."""
    maps = getattr(gaz, "maps", None) or [gaz]
    parts = []
    for m in maps:
        ent = _FIRST_LENS.get(id(m))
        if ent is None or ent[0] is not m or ent[1] != len(m):
            if len(_FIRST_LENS) > 64:
                _FIRST_LENS.clear()
            idx: Dict[str, set] = {}
            for key in m:
                if key:
                    idx.setdefault(key[0], set()).add(len(key))
            ent = (m, len(m), {w: tuple(sorted(ls, reverse=True)) for w, ls in idx.items()})
            _FIRST_LENS[id(m)] = ent
        parts.append(ent[2])
    if len(parts) == 1:
        return parts[0]
    ck = tuple(id(m) for m in maps)
    ent = _FIRST_LENS.get(ck)
    if ent is not None and all(x is y for x, y in zip(ent[0], parts)):
        return ent[1]
    out: Dict[str, set] = {}
    for p in parts:
        for w, ls in p.items():
            out.setdefault(w, set()).update(ls)
    merged = {w: tuple(sorted(ls, reverse=True)) for w, ls in out.items()}
    _FIRST_LENS[ck] = (tuple(parts), merged)
    return merged


def extract(toks, sents, lex, claimed: Optional[Set[int]] = None,
            obs: Optional[list] = None) -> List[Tuple[int, int, str, str]]:
    """Return (start, end, value, source) per organization mention.

    ``lex.org_heads`` (head words) and ``lex.orgs`` (exact-case word tuple ->
    slug: seed gazetteer plus learned aliases) come from the context store.
    A defined acronym -- "National Football League ( NFL )" -- is appended
    to ``obs`` as ("org", "NFL", "national_football_league")."""
    gaz = lex.orgs
    first_lens = _first_word_lengths(gaz)
    out = []
    claimed = set(claimed or ())
    for s, e in sents:
        for i, j in _runs(toks, s, e):
            while i < j and (toks[i].text in LEADING_DROP or (i == s and toks[i].pos_ not in ("PROPN", "NOUN"))):
                i += 1
            while j > i and toks[j - 1].text.lower() in CONNECTORS:
                j -= 1
            if i >= j:
                continue
            names = [n for n in _named_orgs(toks, i, j, lex.org_heads)
                     if not all(k in claimed for k in range(n[0], n[1]))]
            if names:
                for a, b in names:
                    v = slug(" ".join(t.text for t in toks[a:b]))
                    out.append((a, b, v, "org_head"))
                    if obs is not None and b + 2 < len(toks) and toks[b].text == "(" \
                            and toks[b + 2].text == ")" and _ACRONYM.match(toks[b + 1].text):
                        obs.append(("org", toks[b + 1].text, v))
                continue
            # Gazetteer: longest exact-case window inside the run. Only the
            # lengths of names that start with this word are tried (the same
            # windows, longest first, that can match).
            k = i
            while k < j:
                hit = None
                for ln in first_lens.get(toks[k].text, ()):
                    if ln > j - k:
                        continue
                    w = tuple(t.text for t in toks[k:k + ln])
                    if w in gaz:
                        hit = (k, k + ln, gaz[w])
                        break
                if hit:
                    out.append((hit[0], hit[1], hit[2], "org_store"))
                    k = hit[1]
                else:
                    k += 1
    return out
