"""Temporal context: the period a sentence places its facts in.

Value: the containment path of the finest period the text states, coarse to
fine -- century / decade / year -- so a coarser period is a prefix of a finer
one ("C19/D1860/Y1865" lies inside "C19/D1860" inside "C19"). Years before
the common era are negative ("C-1/D-40/Y-44"). A named period is placed in
the coarsest century that contains it and keeps its own name ("C20/E-world_war_ii").

Evidence: the rule NER's DATE spans, plus patterns it does not emit (a
number with an era marker, "44 BC" / "AD 800"; an ordinal century, "the
first century", "mid-20th century", "17th-century"; a three-digit year after
a temporal preposition, "in 405", "by 802"), a fixed table of named periods
and wars, and two name patterns: "<Name> War(s)" / "War(s) of <Name>" and
"<Name> dynasty" (era slug from the name, no century).

One period, one value: every surface of a period -- its names, the current
head's legacy names and era words ("the Renaissance", "Renaissance",
"renaissance"; "WWII", "Second World War") -- resolves to the period's
canonical slug and path (``PERIOD_ALIASES``, ``canonical_periods``).
"""

import re
from typing import List, Optional, Set, Tuple

_YEAR = re.compile(r"^(1\d{3}|20\d{2})$")
_DECADE = re.compile(r"^(1\d{3}|20\d{2})s$")
_ANY_YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")
_ORD_NUM = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$", re.I)
_ERA_BC = {"bc", "bce", "b.c.", "b.c.e.", "b.c"}
_ERA_AD = {"ad", "ce", "a.d.", "c.e.", "a.d"}
_ORD_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "twenty-first": 21,
}
# A four-digit number followed by one of these counts something, not a year.
_COUNT_NOUNS = frozenset("""
people persons inhabitants residents metres meters m km kilometres kilometers
feet ft foot miles mi acres hectares ha square sq tons tonnes copies units votes
students employees members soldiers troops men women children households
families pages words species km2 mm cm kg lb lbs pounds dollars euros points
yards horsepower hp seats rpm am fm khz mhz calories ad-hoc runs goals wickets games matches
""".split())
_CURRENCY = frozenset({"$", "£", "€", "¥", "US$", "A$", "C$", "no.", "No.", "#", "number"})

# Named periods: surface (as written) -> (century path, era slug). The century
# is the coarsest one that contains the whole period; periods spanning several
# centuries carry the era slug alone.
NAMED_PERIODS = {
    "World War I": ("C20/D1910", "world_war_i"),
    "First World War": ("C20/D1910", "world_war_i"),
    "World War II": ("C20", "world_war_ii"),
    "Second World War": ("C20", "world_war_ii"),
    "Great Depression": ("C20/D1930", "great_depression"),
    "Cold War": ("C20", "cold_war"),
    "American Civil War": ("C19/D1860", "american_civil_war"),
    "American Revolution": ("C18/D1770", "american_revolution"),
    "French Revolution": ("C18", "french_revolution"),
    "Napoleonic Wars": ("C19/D1800", "napoleonic_wars"),
    "Industrial Revolution": ("", "industrial_revolution"),
    "Victorian era": ("C19", "victorian_era"),
    "Middle Ages": ("", "middle_ages"),
    "Renaissance": ("", "renaissance"),
    "Enlightenment": ("", "enlightenment"),
    "Bronze Age": ("", "bronze_age"),
    "Iron Age": ("", "iron_age"),
    "Stone Age": ("", "stone_age"),
    "Neolithic": ("", "neolithic"),
    "Antiquity": ("", "antiquity"),
    "Classical antiquity": ("", "classical_antiquity"),
    "Late Antiquity": ("", "late_antiquity"),
    "Early Modern period": ("", "early_modern_period"),
    "Reformation": ("C16", "reformation"),
    "Great War": ("C20/D1910", "world_war_i"),
    "WWI": ("C20/D1910", "world_war_i"),
    "WWII": ("C20", "world_war_ii"),
    # Wars, keyed without apostrophes and dashes (``period_words``). The path
    # is the finest century / decade containing the whole war.
    "English Civil War": ("C17", "english_civil_war"),
    "Wars of the Three Kingdoms": ("C17", "wars_of_the_three_kingdoms"),
    "Thirty Years War": ("C17", "thirty_years_war"),
    "Eighty Years War": ("", "eighty_years_war"),
    "Hundred Years War": ("", "hundred_years_war"),
    "Seven Years War": ("C18", "seven_years_war"),
    "Wars of the Roses": ("C15", "wars_of_the_roses"),
    "War of the Spanish Succession": ("C18", "war_of_the_spanish_succession"),
    "War of the Austrian Succession": ("C18/D1740", "war_of_the_austrian_succession"),
    "French and Indian War": ("C18", "french_and_indian_war"),
    "American Revolutionary War": ("C18", "american_revolutionary_war"),
    "War of 1812": ("C19/D1810", "war_of_1812"),
    "Opium Wars": ("C19", "opium_wars"),
    "Mexican American War": ("C19/D1840", "mexican_american_war"),
    "Crimean War": ("C19/D1850", "crimean_war"),
    "Franco-Prussian War": ("C19/D1870", "franco_prussian_war"),
    "Anglo-Zulu War": ("C19/D1870", "anglo_zulu_war"),
    "Spanish American War": ("C19/D1890", "spanish_american_war"),
    "Russo-Japanese War": ("C20/D1900", "russo_japanese_war"),
    "Balkan Wars": ("C20/D1910", "balkan_wars"),
    "Russian Revolution": ("C20/D1910", "russian_revolution"),
    "Russian Civil War": ("C20", "russian_civil_war"),
    "Mexican Revolution": ("C20", "mexican_revolution"),
    "Chinese Civil War": ("C20", "chinese_civil_war"),
    "Spanish Civil War": ("C20/D1930", "spanish_civil_war"),
    "Winter War": ("C20", "winter_war"),
    "Korean War": ("C20/D1950", "korean_war"),
    "Vietnam War": ("C20", "vietnam_war"),
    "Six-Day War": ("C20/D1960", "six_day_war"),
    "Yom Kippur War": ("C20/D1970", "yom_kippur_war"),
    "Iran Iraq War": ("C20/D1980", "iran_iraq_war"),
    "Falklands War": ("C20/D1980", "falklands_war"),
    "Gulf War": ("C20/D1990", "gulf_war"),
    "Iraq War": ("C21", "iraq_war"),
    "Glorious Revolution": ("C17/D1680", "glorious_revolution"),
    "Peloponnesian War": ("C-5", "peloponnesian_war"),
    "Greco-Persian Wars": ("C-5", "greco_persian_wars"),
    "Punic Wars": ("", "punic_wars"),
    "Trojan War": ("", "trojan_war"),
}
# Other names of a period (legacy TEMPORAL_PERIODS slugs, legacy era words)
# -> the period's canonical slug.
PERIOD_ALIASES = {
    "first_world_war": "world_war_i", "great_war": "world_war_i", "wwi": "world_war_i",
    "second_world_war": "world_war_ii", "wwii": "world_war_ii",
    "medieval": "middle_ages", "medieval_era": "middle_ages",
    "renaissance_era": "renaissance", "enlightenment_era": "enlightenment",
    "industrial_era": "industrial_revolution",
}
# Punctuation tokens a period name may carry that its key leaves out
# ("Thirty Years ' War", "Mexican – American War").
_NAME_PUNCT = frozenset({"'", "\u2019", "\u2013", "\u2014"})


def canonical_periods() -> dict:
    """Canonical slug -> path, from NAMED_PERIODS."""
    return {slug: path for path, slug in NAMED_PERIODS.values()}


def canonical(path: str, slug: str) -> Tuple[str, str]:
    """A period (path, slug) from any table -> its canonical (path, slug)."""
    slug = PERIOD_ALIASES.get(slug, slug)
    canon = _CANON.get(slug)
    return (canon, slug) if canon is not None else (path, slug)


_CANON = canonical_periods()
_PERIOD_TOKENS = {tuple(k.split()): v for k, v in NAMED_PERIODS.items()}
_MAX_PERIOD_LEN = max(len(k) for k in _PERIOD_TOKENS)

# Era words (the current head's TEMPORAL_KEYWORD_PATTERNS) that are ordinary
# adjectives as often as periods ("modern media", "contemporary art",
# "industrial city", "spiritual enlightenment"): they count only before a
# period noun ("the modern era", "contemporary history").
AMBIGUOUS_ERA_WORDS = frozenset({"modern", "contemporary", "industrial", "enlightenment"})
PERIOD_NOUNS = frozenset({"era", "eras", "period", "periods", "age", "ages", "times", "history"})

# Name patterns. Words that do not name a particular war on their own.
_WAR_HEADS = frozenset({"war", "wars"})
_GENERIC_WAR_WORDS = frozenset({
    "the", "of", "and", "civil", "great", "holy", "world", "cold", "total", "secret",
    "first", "second", "third", "fourth", "fifth", "last", "independence", "liberation",
    "revolutionary", "national", "patriotic", "people's", "new", "old"})
_TITLE_WORDS = frozenset({"secretary", "minister", "department", "ministry", "office", "museum",
                          "memorial", "council", "cabinet", "college", "school", "veterans",
                          "correspondent", "crimes", "criminal", "tribunal", "game", "games"})
_ORDINAL_PREFIX = re.compile(r"^(?:(early|mid|late)-)?(.+?)(-century)?$")
_ORD_SUFFIX_NUM = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)$")
# Three-digit years: only right after a temporal preposition (optionally one
# hedge word between), or joined to such a year ("977 or 979").
_YEAR_PREPS = frozenset({"in", "by", "until", "till", "since", "circa", "c.", "ca.", "year"})
_YEAR_HEDGES = frozenset({"about", "around", "approximately", "either", "c", "ca", "circa", "c.", "ca."})
_YEAR_JOINERS = frozenset({"or", "and", "to", "-", "\u2013"})
_NOT_AFTER_YEAR_POS = frozenset({"NOUN", "ADJ", "NUM", "PROPN"})


def century_of(year: int) -> int:
    # Popular convention (1900 is in the 20th century) so decades nest in
    # centuries: every year of D1900 is in C20.
    return year // 100 + 1 if year > 0 else -(abs(year) // 100 + 1)


def decade_of(year: int) -> int:
    return (year // 10) * 10 if year > 0 else -((abs(year) // 10) * 10)


def year_path(year: int) -> str:
    return f"C{century_of(year)}/D{decade_of(year)}/Y{year}"


def decade_path(decade: int) -> str:
    return f"C{century_of(decade)}/D{decade}"


def century_path(c: int) -> str:
    return f"C{c}"


def _is_count(toks, end: int) -> bool:
    return end < len(toks) and toks[end].text.lower() in _COUNT_NOUNS


def _preceded_by_currency(toks, start: int) -> bool:
    if start == 0:
        return False
    p = toks[start - 1].text
    if p in _CURRENCY:
        return True
    return start >= 2 and toks[start - 2].text.lower() == "population" and p.lower() == "of"


def _era_after(toks, end: int) -> Optional[int]:
    if end < len(toks):
        t = toks[end].text.lower()
        if t in _ERA_BC:
            return -1
        if t in _ERA_AD:
            return 1
    return None


def from_date_span(toks, start: int, end: int, text: str) -> Optional[str]:
    """Normalize one rule-NER DATE span; None if it states no year/decade/century."""
    if _preceded_by_currency(toks, start) or _is_count(toks, end):
        return None
    words = text.split()
    low = [w.lower() for w in words]
    if "century" in low:
        k = low.index("century")
        if k >= 1:
            m = _ORD_NUM.match(words[k - 1])
            c = int(m.group(1)) if m else _ORD_WORDS.get(low[k - 1])
            if c:
                sign = -1 if _era_after(toks, end) == -1 else 1
                return century_path(sign * c)
        return None
    if len(words) == 1 and _DECADE.match(words[0]):
        return decade_path(int(words[0][:4]))
    if len(words) == 1 and _YEAR.match(words[0]):
        y = int(words[0])
        return year_path(-y if _era_after(toks, end) == -1 else y)
    m = _ANY_YEAR.search(text)  # "May 1865", "January 15, 2024"
    if m:
        return year_path(int(m.group(1)))
    return None


_PERIOD_INDEX: dict = {}


def _period_index(periods) -> Tuple[int, set]:
    """(longest period in tokens, first words) of a period table, cached per
    table object (the seed table is immutable and was rescanned per doc)."""
    ent = _PERIOD_INDEX.get(id(periods))
    if ent is None or ent[0] is not periods or ent[1] != len(periods):
        if len(_PERIOD_INDEX) > 16:
            _PERIOD_INDEX.clear()
        ent = (periods, len(periods), max((len(k) for k in periods), default=1),
               {k[0] for k in periods})
        _PERIOD_INDEX[id(periods)] = ent
    return ent[2], ent[3]


def _ordinal(word: str) -> Optional[int]:
    """1..21 for "17th" / "seventeenth"; None otherwise."""
    m = _ORD_SUFFIX_NUM.match(word)
    if m:
        v = int(m.group(1))
        return v if 1 <= v <= 21 else None
    return _ORD_WORDS.get(word)


def _hyphen_century(toks, i: int) -> Optional[Tuple[int, int, int]]:
    """"mid-20th century", "17th-century", "late-nineteenth century" (one
    rule-tokenizer token, the first two followed by "century"):
    (start, end, signed century) or None."""
    low = toks[i].text.lower()
    if "-" not in low:
        return None
    m = _ORDINAL_PREFIX.match(low)
    if not m or not (m.group(1) or m.group(3)):
        return None
    c = _ordinal(m.group(2))
    if c is None:
        return None
    end = i + 1
    if not m.group(3):
        if end >= len(toks) or toks[end].text.lower() not in ("century", "centuries"):
            return None
        end += 1
    return i, end, (-c if _era_after(toks, end) == -1 else c)


def _three_digit_year(toks, i: int, years: Set[int]) -> bool:
    """A 100-999 number that is a year: right after a temporal preposition
    ("in 405", "by 802", "in the year 640"), after one hedge word ("in about
    977", "in either 977"), or joined to such a year ("977 or 979"); and not
    followed by a noun, adjective or number ("in 405 cases")."""
    txt = toks[i].text
    if not (txt.isdecimal() and len(txt) == 3 and txt[0] != "0"):
        return False
    if i + 1 < len(toks):
        nxt = toks[i + 1]
        if nxt.text.lower() in _COUNT_NOUNS or getattr(nxt, "pos_", "") in _NOT_AFTER_YEAR_POS:
            return False
    p1 = toks[i - 1].text.lower() if i >= 1 else ""
    p2 = toks[i - 2].text.lower() if i >= 2 else ""
    if p1 in _YEAR_PREPS:
        return True
    if p1 in _YEAR_HEDGES and p2 in _YEAR_PREPS:
        return True
    return p1 in _YEAR_JOINERS and (i - 2) in years


def _period_window(toks, i: int, plen: int) -> Tuple[list, list]:
    """Words and token indices from i, skipping apostrophe / dash tokens,
    up to ``plen`` words (``NAMED_PERIODS`` keys leave them out)."""
    words, idx = [], []
    k = i
    while k < len(toks) and len(words) < plen:
        t = toks[k].text
        if t in _NAME_PUNCT:
            if not words:
                break
            k += 1
            continue
        words.append(t.lower())
        idx.append(k)
        k += 1
    return words, idx


def _name_run(toks, i: int, limit: int = 8) -> List[int]:
    """Token indices of the capitalized name starting at i: capitalized words,
    with "of" / "the" / "and" and apostrophe / dash tokens inside."""
    run = []
    k = i
    while k < len(toks) and len(run) < limit:
        t = toks[k].text
        if t[:1].isupper() or t[:1].isdecimal():
            run.append(k)
        elif t.lower() in ("of", "the", "and") or t in _NAME_PUNCT:
            if not run or k + 1 >= len(toks) or not (toks[k + 1].text[:1].isupper()
                                                      or toks[k + 1].text[:1].isdecimal()):
                break
            run.append(k)
        else:
            break
        k += 1
    while run and toks[run[-1]].text.lower() in ("of", "the", "and"):
        run.pop()
    while run and toks[run[0]].text.lower() == "the":
        run = run[1:]
    return run


def _slug(words) -> str:
    return re.sub(r"[^0-9a-z]+", "_", " ".join(words).lower()).strip("_")


def _named_war(toks, i: int, periods, plen: int, heads) -> Optional[Tuple[int, int, str, str]]:
    """A war's name from token i: (start, end, value, source). A table period
    ending where the name ends gives the table's value ("First English Civil
    War" is the English Civil War; "Wars of the Three Kingdoms"). Otherwise
    "<Name> War" (singular: "Star Wars", "Clone Wars" are not wars) is its own
    period, slug of the name, when a word particularises it ("Civil War" does
    not) and no institution or title word is in it ("Department of War")."""
    run = _name_run(toks, i)
    if not run:
        return None
    low = [toks[k].text.lower() for k in run]
    if low[0] in _WAR_HEADS:
        if len(low) < 3 or low[1] != "of":
            return None
    else:
        last = max((p for p, w in enumerate(low) if w in _WAR_HEADS), default=None)
        if last is None or last == 0:
            return None
        run, low = run[:last + 1], low[:last + 1]
    end = run[-1] + 1
    words = [w for w in low if w not in _NAME_PUNCT]
    if any(w in _TITLE_WORDS or w.capitalize() in heads for w in words if w not in _WAR_HEADS):
        return None
    for k in run:
        ws, idx = _period_window(toks, k, plen)
        for ln in range(len(ws), 0, -1):
            if idx[ln - 1] + 1 != end:
                continue
            hit = periods.get(tuple(ws[:ln]))
            if hit:
                path, slug = canonical(*hit)
                return run[0], end, (f"{path}/E-{slug}" if path else f"E-{slug}"), "named_period"
    if low[-1] != "war" or not any(w not in _GENERIC_WAR_WORDS and w not in _WAR_HEADS
                                   and not w.isdecimal() for w in words):
        return None
    return run[0], end, f"E-{_slug(words)}", "named_war"


def _dynasty(toks, i: int) -> Optional[Tuple[int, str]]:
    """"<Name> dynasty" (1-3 capitalized words, then "dynasty"): (end, value)."""
    k = i
    while k < len(toks) and k - i < 3 and toks[k].text[:1].isupper() and \
            toks[k].text.lower() not in ("the", "dynasty"):
        k += 1
    if k == i or k >= len(toks) or toks[k].text.lower() != "dynasty":
        return None
    return k + 1, f"E-{_slug([t.text for t in toks[i:k]] + ['dynasty'])}"


def extract(toks, date_spans, lex, sents=None,
            event_spans=()) -> List[Tuple[int, int, str, str]]:
    """Return (start, end, value, source) for every temporal mention.

    Named periods (``lex.periods``: NAMED_PERIODS plus the current head's
    TEMPORAL_PERIODS, canonicalised) match case-insensitively on a window
    whose first token is capitalized; then named wars and dynasties by
    pattern. Years, decades and centuries are computed. An NER EVENT span
    (backends that emit it) is searched for a year like a DATE span.
    Era words (``lex.era_words``: the current head's
    TEMPORAL_KEYWORD_PATTERNS) count only in a sentence that states no
    year/decade/century -- the old head's fallback condition, per sentence --
    and an ambiguous one only before a period noun."""
    periods = lex.periods
    plen, first_words = _period_index(periods)
    heads = getattr(lex, "org_heads", ()) or ()
    out = []
    claimed = set()
    for sp in date_spans:
        v = from_date_span(toks, sp.start, sp.end, sp.text)
        claimed.update(range(sp.start, sp.end))
        if v:
            out.append((sp.start, sp.end, v, "ner_date"))
    for sp in event_spans:
        m = _ANY_YEAR.search(sp.text)
        if m and not any(k in claimed for k in range(sp.start, sp.end)):
            out.append((sp.start, sp.end, year_path(int(m.group(1))), "ner_event"))
    n = len(toks)
    years: Set[int] = set()
    skip = -1
    # Name patterns run only where their head word is near (cheap gate).
    war_at = [k for k in range(n) if toks[k].text.lower() in _WAR_HEADS]
    dyn_at = [k for k in range(n) if toks[k].text.lower() == "dynasty"]
    for i, t in enumerate(toks):
        if i in claimed or i < skip:
            continue
        txt = t.text
        low = txt.lower()
        # "44 BC", "476 AD". A number is a run of decimal digits
        # (str.isdecimal: ASCII or any script's 0-9, exactly what int() reads
        # and what the regexes' \d matches). str.isdigit() is also True for
        # superscripts and circled digits ("²", "⑦"), which int() rejects:
        # one such token used to fail the whole head batch.
        if txt.isdecimal() and 1 <= len(txt) <= 4 and int(txt) > 0:
            era = _era_after(toks, i + 1)
            if era is not None:
                out.append((i, i + 2, year_path(era * int(txt)), "era_year"))
                continue
            if _three_digit_year(toks, i, years):
                years.add(i)
                out.append((i, i + 1, year_path(int(txt)), "year_after_preposition"))
                continue
        # "AD 800"
        if low in _ERA_AD and i + 1 < n and toks[i + 1].text.isdecimal() \
                and len(toks[i + 1].text) <= 4 and (i + 1) not in claimed:
            out.append((i, i + 2, year_path(int(toks[i + 1].text)), "era_year"))
            continue
        # "the first century (BC)", "the 10th century" outside a DATE span
        c = _ordinal(low)
        if c is not None and i + 1 < n and toks[i + 1].text.lower() in ("century", "centuries"):
            sign = -1 if _era_after(toks, i + 2) == -1 else 1
            out.append((i, i + 2, century_path(sign * c), "ordinal_century"))
            continue
        # "mid-20th century", "17th-century"
        hc = _hyphen_century(toks, i)
        if hc:
            out.append((hc[0], hc[1], century_path(hc[2]), "ordinal_century"))
            skip = hc[1]
            continue
        if not txt[:1].isupper():
            continue
        # Named periods: a capitalized first token ("World War II", "WWII").
        hit = None
        if low in first_words:
            words, idx = _period_window(toks, i, plen)
            for ln in range(len(words), 0, -1):
                h = periods.get(tuple(words[:ln]))
                if h:
                    path, slug = canonical(*h)
                    hit = (i, idx[ln - 1] + 1, f"{path}/E-{slug}" if path else f"E-{slug}", "named_period")
                    break
        if hit is None and any(i <= k < i + 8 for k in war_at):
            hit = _named_war(toks, i, periods, plen, heads)
        if hit is None and any(i < k <= i + 3 for k in dyn_at):
            d = _dynasty(toks, i)
            if d:
                hit = (i, d[0], d[1], "named_dynasty")
        if hit:
            out.append(hit)
            skip = hit[1]
    if sents and lex.era_words:
        era_paths = getattr(lex, "era_paths", None) or {}
        stated = {next((si for si, (a, b) in enumerate(sents) if a <= o[0] < b), -1) for o in out}
        for si, (a, b) in enumerate(sents):
            if si in stated:
                continue
            for i in range(a, b):
                low = toks[i].text.lower()
                era = lex.era_words.get(low)
                if not era:
                    continue
                nxt = toks[i + 1].text.lower() if i + 1 < b else ""
                if low == "modern" and i > a and toks[i - 1].text.lower() == "early":
                    path, slug = canonical("", "early_modern_period")
                    out.append((i - 1, i + 1, f"{path}/E-{slug}" if path else f"E-{slug}",
                                "legacy_era_word"))
                    break
                if low in AMBIGUOUS_ERA_WORDS and nxt not in PERIOD_NOUNS:
                    continue
                path = era_paths.get(era, "")
                out.append((i, i + 1, f"{path}/E-{era}" if path else f"E-{era}", "legacy_era_word"))
                break
    return out
