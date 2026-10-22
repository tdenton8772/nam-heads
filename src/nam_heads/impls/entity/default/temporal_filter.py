# src/encoders/heads/entity/temporal_filter.py

"""Mentions the entity head never emits: temporal values and 1-2 character fragments.

Owner rule: dates, years, decades and centuries ("1980", "1980s", "the 1980s",
"19th century", "century") are temporal CONTEXT, never entities, and a 1-2
character fragment ("st", "ii", "re") is never an entity either -- but a
2-letter name written in capitals (UN, DC, NY, GM) is a name. Each such
"entity" became a partition that every record mentioning it appended to --
the largest partitions on the reference deployment (2026-10-05) were `1980`, `1970`, `1990`,
`century` and `st`.

Nothing is lost: the context head reads the same rule-NLP DATE/TIME spans
from the parse itself, not from entity head output, and turns each year,
decade or century into a temporal value -- the default head in
``src/context_axis`` (extract.py -> temporal.from_date_span: "the 1980s" ->
C20/D1980, "19th century" -> C19), the legacy head via
``normalize_temporal_rules``. A month or weekday with no year carries no
value in either.

``is_non_entity_mention`` is the one predicate; EntityEncoder applies it to
ingest and query candidates alike, so a query never looks for a partition that
ingest does not write.
"""

import re
from typing import AbstractSet, Optional

# Rule-NLP NER labels for temporal spans (nlp/rule/ner.py, layer 1).
TEMPORAL_NER_LABELS = frozenset({"DATE", "TIME"})

# Labels the rule NER gives a single token only from a gazetteer (UK, US, EU
# -> GPE; HP, BP -> ORG). PERSON is not here: a lone PROPN token is labelled
# PERSON by default (ner.py _classify_propn_span) -- that is how "St", "II",
# "BC" and "Dr" all come out PERSON -- and gazetteer PERSON needs first + last.
_LEXICON_LABELS = frozenset({"GPE", "ORG", "NORP", "LOC", "FAC"})

# Two capital letters that are not names: the Roman numerals 2-20 that are
# two letters (regnal and sequel numbers -- "II" is the PROPN tail of every
# "World War II"), the era markers, and TV. Larger two-letter numerals are
# left out on purpose: DC, MD, MI, CD are far more often names than 600,
# 1500, 1001, 400.
_ACRONYM_STOPLIST = frozenset({
    "II", "IV", "VI", "IX", "XI", "XV", "XX",
    "AD", "BC", "CE",
    "TV",
})

# Month/day abbreviations the rule NER tags DATE wherever they occur, so its
# DATE label alone does not make them temporal: "the Sun", "Jan", "Mar".
_DATE_ABBREVIATIONS = frozenset({
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec", "mon", "tue", "tues", "wed", "thu", "thurs", "fri", "sat", "sun",
})

_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|"
    "november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_ORDINAL_WORDS = (
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    "eighteenth|nineteenth|twentieth|twenty_first"
)
_YEAR = r"(?:1\d{3}|20\d{2})"           # the rule NER's year range (ner.py _YEAR_RE)
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_ERA = r"(?:bc|bce|ad|ce)"

# A whole normalized mention (normalize_mention: snake_case, punctuation
# dropped) that is a temporal value and nothing else.
_TEMPORAL_VALUE_RE = re.compile(
    r"^(?:the_)?(?:(?:early|mid|late)_)?(?:"
    rf"{_YEAR}s?"                                   # 1980, 1980s
    rf"|{_YEAR}_(?:\d{{2}}|{_YEAR})"                # 1914_1918, 2004_05
    r"|\d0s"                                        # 80s
    rf"|\d{{1,4}}_{_ERA}|ad_\d{{1,4}}"              # 800_ad, 44_bc, ad_800
    rf"|(?:\d{{1,2}}(?:st|nd|rd|th)|{_ORDINAL_WORDS})_(?:century|centuries|millennium|millennia)"
    r"|(?:half_)?century|centuries"                 # bare century (owner's example)
    rf"|{_ERA}"                                     # an era marker alone
    rf"|(?:{_MONTHS})_{_DAY}(?:_{_YEAR})?"          # march_15, march_15_1950
    rf"|{_DAY}_(?:{_MONTHS})(?:_{_YEAR})?"          # 15_march, 15_march_1950
    rf"|(?:{_MONTHS})_{_YEAR}"                      # march_1950
    rf")(?:_{_ERA})?$"
)

# A letter-digit designation is a name, not a fragment: U2, F1, M6, A1.
_DESIGNATION_RE = re.compile(r"^[A-Z]\d$")


def is_temporal_value(normalized: str) -> bool:
    """True if *normalized* (snake_case) is a pure temporal value: a year,
    decade, year range, century or millennium (optionally early/mid/late and
    an era marker), an era marker alone, a date of month + day/year, or the
    bare word century/centuries."""
    return bool(normalized) and bool(_TEMPORAL_VALUE_RE.match(normalized))


def is_designation(surface: str) -> bool:
    """A capital letter + digit name such as U2, F1, K-2 -- not a fragment.

    Letter first only: digit-first pairs in the wiki corpus are mostly
    URL-escape debris ("%2C" -> "2C", 134 of 3,000 records)."""
    return bool(_DESIGNATION_RE.match(re.sub(r"[\s\-]", "", surface or "")))


def is_capital_acronym(surface: str) -> bool:
    """Exactly two ASCII capital letters as written ("UN", "D.C.", "NY"),
    not on the stoplist (Roman numerals II-XX, AD/BC/CE, TV). "St", "Dr",
    "Li" and lower-case "un" are not."""
    letters = re.sub(r"[.\s]", "", surface or "")
    return (len(letters) == 2 and letters.isascii() and letters.isalpha()
            and letters.isupper() and letters not in _ACRONYM_STOPLIST)


def _is_ner_temporal(text: str, normalized: str,
                     ner_labels: Optional[AbstractSet[str]]) -> bool:
    """Every content token of the mention sits in a rule-NLP DATE/TIME span.

    Two rule-NER false positives are excluded for a single token: a month/day
    abbreviation ("the Sun") and a lower-case word ("march" the noun -- the
    rule NER matches month names case-blind)."""
    if not ner_labels or not set(ner_labels) <= TEMPORAL_NER_LABELS:
        return False
    if "_" not in normalized:
        if normalized in _DATE_ABBREVIATIONS:
            return False
        surface = (text or "").strip()
        if surface[:1].isalpha() and surface[:1].islower():
            return False
    return True


def is_fragment(text: str, normalized: str,
                ner_labels: Optional[AbstractSet[str]] = None) -> bool:
    """1-2 characters ("st", "ii", "re", "10", or nothing at all after
    normalization), unless
    - two capital letters as written (UN, DC, NY, GM), off the stoplist
      (II, IV ... XX, AD, BC, CE, TV) -- is_capital_acronym;
    - a letter-digit designation (U2, F1) -- is_designation;
    - a token the rule NER typed from a gazetteer (UK, US, EU, HP, BP), so a
      lower-case "uk" still counts.
    The query resolver uses this same rule for its 1-2 character mentions."""
    if len(normalized.replace("_", "")) > 2:
        return False
    if is_capital_acronym(text) or is_designation(text):
        return False
    if ner_labels and len(ner_labels) == 1 and next(iter(ner_labels)) in _LEXICON_LABELS:
        return False
    return True


def is_non_entity_mention(text: str, normalized: str,
                          ner_labels: Optional[AbstractSet[str]] = None) -> bool:
    """True if a candidate mention must not become an entity.

    text:       the surface form as written (case matters for the exceptions).
    normalized: its normalize_mention() form.
    ner_labels: rule-NLP NER labels over the mention's content tokens ("O" for
                an unlabelled token), or None when unknown (title, query
                full-text and fallback candidates). With None only the
                form rules apply.
    """
    if is_temporal_value(normalized):
        return True
    if _is_ner_temporal(text, normalized, ner_labels):
        return True
    return is_fragment(text, normalized, ner_labels)
