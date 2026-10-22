"""The current context head's tables, folded into the particular-evidence
extractor as seed and evidence -- nothing it uses is silently dropped.

Read from ``heads/context/definitions.py`` at seed time (single source: when
the ontology session re-keys that module, this follows it):

* ``DOMAIN_KEYWORD_PATTERNS`` -- keyword -> field, with the old head's rule
  kept: a field needs >= 2 distinct of its keywords, now per sentence
  instead of per record. Labels map to field names below.
* ``TEMPORAL_KEYWORD_PATTERNS`` -- era words ("medieval", "colonial") ->
  ``t:E-<era>``, only in a sentence with no stated year/decade/century (the
  old head's fallback condition, per sentence).
* ``TEMPORAL_PERIODS`` -- named periods; each becomes a named period with
  the containment path of its canonical value ("1940s" -> C20/D1940).
* ``INSTITUTIONAL_KEYWORDS`` -- each keyword is an organization head word;
  its kind (academic, military, ...) is kept as METADATA on the value, never
  as the value (that was the 'commercial' collapse).
* ``GEOGRAPHIC_SEED`` -- surfaces the gazetteer lacks are added, pointing at
  the place their canonical value names.
* ``DOMAIN_NER_OVERRIDES`` -- an NER span with that label (EVENT,
  WORK_OF_ART, LAW, LANGUAGE) is domain evidence for its sentence.

Not used to emit values: ``DOMAIN_SEEDS`` (ontology -> domain) and
``INSTITUTIONAL_DEFAULT`` ('commercial' for any ORG) -- they derive a value
from a type or a default, which is the collapse being fixed.
"""

import re
from functools import lru_cache
from typing import Dict, Tuple

# DOMAIN_KEYWORD_PATTERNS label -> field name used by domains.py.
DOMAIN_LABEL_FIELD = {
    "political": "politics", "military": "military", "scientific": "science",
    "cultural": "culture", "medical": "medicine", "legal": "law",
    "religious": "religion", "economic": "economics",
    "technological": "technology", "educational": "education",
}
# DOMAIN_NER_OVERRIDES label -> field name.
NER_DOMAIN_FIELD = {"historical": "history", "cultural": "culture", "legal": "law"}
LEGACY_MIN_HITS = 2


def _defs():
    from .. import definitions
    return definitions


@lru_cache(maxsize=1)
def keyword_fields() -> Dict[str, str]:
    """Legacy domain keyword -> field (needs LEGACY_MIN_HITS per sentence)."""
    out = {}
    for label, words in _defs().DOMAIN_KEYWORD_PATTERNS.items():
        field = DOMAIN_LABEL_FIELD.get(label, label)
        for w in words:
            out.setdefault(w.lower(), field)
    return out


@lru_cache(maxsize=1)
def era_words() -> Dict[str, str]:
    return {k.lower(): v for k, v in _defs().TEMPORAL_KEYWORD_PATTERNS.items()}


@lru_cache(maxsize=1)
def ner_domain_fields() -> Dict[str, str]:
    return {lab: NER_DOMAIN_FIELD.get(lbl, lbl) for lab, lbl in _defs().DOMAIN_NER_OVERRIDES.items()}


def period_path(canonical: str) -> str:
    """Containment path of a legacy canonical period value."""
    from .temporal import century_path, decade_path
    m = re.fullmatch(r"(\d{3,4})s", canonical)
    if m:
        return decade_path(int(m.group(1)))
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)_century(_bce)?", canonical)
    if m:
        return century_path(-int(m.group(1)) if m.group(2) else int(m.group(1)))
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)_millennium(_bce)?", canonical)
    if m:
        return f"M{'-' if m.group(2) else ''}{m.group(1)}"
    return ""


@lru_cache(maxsize=1)
def named_periods() -> Dict[str, Tuple[str, str]]:
    """Legacy TEMPORAL_PERIODS as lowercase surface -> (path, era slug)."""
    out = {}
    for surface, canonical in _defs().TEMPORAL_PERIODS.items():
        s = surface.lower()
        slug = re.sub(r"^the ", "", s)
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        out[s] = (period_path(canonical), slug)
    return out


@lru_cache(maxsize=1)
def institutional_heads() -> Dict[str, str]:
    """Capitalized head word -> legacy kind (metadata only)."""
    out = {}
    for kind, words in _defs().INSTITUTIONAL_KEYWORDS.items():
        for w in words.split():
            out.setdefault(w.capitalize(), kind)
    return out


@lru_cache(maxsize=1)
def geographic_seed() -> Dict[str, str]:
    return dict(_defs().GEOGRAPHIC_SEED)
