# src/encoders/heads/entity/generic_nouns.py

"""Generic nouns: lower-case single words that are never an entity.

"career", "community", "commitment", "town", "years" name no particular
thing; as entities each became a partition that a large share of all records
appended to ("career" was a bundled entity in 6.5% of 20,000 passages,
2026-10-07 audit). The codebook classifier cannot tell them apart: it types
by word shape, keeps "career" and "community" and rejects "gene" and
"psychology".

The list is FROZEN and versioned (codebook/generic_nouns_v1.json), built
offline by scripts/build_entity_generic_nouns.py from document frequency over
a corpus slice: a word is generic when, as a lower-case single-word candidate
of this head, it occurs in at least ``min_article_share`` of the articles.
The head never learns or changes it at run time, so the same text gives the
same entities on every pod. A new list is a new version and needs a re-land
of the entity partitions.

Only a lower-case single-word mention that is not a name is ever generic; the
record title and every capitalised name pass. Looked up by the canonical form
(the head's entity id form) and the normalized form.
"""

import json
from pathlib import Path
from typing import FrozenSet, Optional

from .normalization import EntityCandidate

ARTIFACT = Path(__file__).parent / "codebook" / "generic_nouns_v1.json"
VERSION = 1

_nouns: Optional[FrozenSet[str]] = None
_meta: dict = {}


def _load() -> FrozenSet[str]:
    global _nouns, _meta
    if _nouns is None:
        data = json.loads(ARTIFACT.read_text())
        if data.get("version") != VERSION:
            raise ValueError(f"{ARTIFACT.name}: version {data.get('version')!r}, expected {VERSION}")
        _meta = {k: v for k, v in data.items() if k != "nouns"}
        _nouns = frozenset(data["nouns"])
    return _nouns


def generic_nouns() -> FrozenSet[str]:
    """The frozen generic-noun set (canonical forms)."""
    return _load()


def artifact_meta() -> dict:
    """Provenance of the loaded list (corpus, slice, threshold, parser)."""
    _load()
    return dict(_meta)


def is_generic_noun(cand: EntityCandidate, canonical: str) -> bool:
    """True if *cand* is a generic noun the head must not emit.

    - one word whose canonical (or normalized) form is on the list;
    - a listed noun with only adjectives in front of it ("political
      landscape", "early years", "Scottish culture"): an adjective does not
      make a generic noun name anything.

    A name (EntityCandidate.is_name: the record title, a capitalised span,
    a capitalised word inside a sentence) never is. A word capitalised only
    because it opens a sentence ("Community ...") is not a name and is looked
    up like any other. ``canonical`` is the head's canonical form of the
    mention (its last word singularised).
    """
    if cand.is_name:
        return False
    nouns = _load()
    if "_" not in cand.normalized:
        return canonical in nouns or cand.normalized in nouns
    tags = cand.pos_pattern.split("_") if cand.pos_pattern and cand.pos_pattern != "UNK" else []
    if len(tags) < 2 or tags[-1] not in ("NOUN", "PROPN") or any(t != "ADJ" for t in tags[:-1]):
        return False
    return canonical.rsplit("_", 1)[-1] in nouns
