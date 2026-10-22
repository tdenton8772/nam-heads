"""Seed version for the affordance store.

Seeding writes absent keys only, so a session store that predates a changed
seed artifact would keep the old seeded rows forever. Every seed row is
stamped with ``seed_version`` -- a hash of the seed sources -- and a row whose
stamp differs is re-seeded. Rows without a stamp are re-seeded only when they
look like the old seed (category rows at confidence 1.0); learned rows
(auto-classified at 0.5, learned lexicon docs) are never touched.
"""

import hashlib
import os
from functools import lru_cache
from typing import Optional

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
_SOURCES = ("verb_bases.json", "verb_synonyms.json", "verb_synonyms_unfolded.json",
            "verb_categories.json")


@lru_cache(maxsize=1)
def seed_version() -> str:
    """Hash of the seed data files and the in-code seed tables."""
    h = hashlib.sha256()
    for name in _SOURCES:
        try:
            with open(os.path.join(_DATA, name), "rb") as f:
                h.update(f.read())
        except OSError:
            h.update(name.encode())
    from .. import definitions as D
    from .. import predicates as P
    h.update(repr(sorted((k, sorted(v)) for k, v in D.VERB_CATEGORIES.items())).encode())
    h.update(repr(sorted(D.ONTOLOGY_CATEGORY_OVERRIDES.items())).encode())
    h.update(repr(sorted(P.PHRASAL.items())).encode())
    return h.hexdigest()[:16]


def needs_seed(existing: Optional[dict], version: Optional[str]) -> bool:
    """Should a seed row with ``version`` be written over ``existing``?"""
    if not existing:
        return True
    if existing.get("kind") == "learned":
        return False
    stamp = existing.get("seed_version")
    if stamp is not None:
        return stamp != version
    if "kind" in existing:                       # unstamped seed lexicon doc
        return existing.get("kind") == "seed"
    # unstamped category row: the old seed wrote confidence 1.0,
    # auto-classification writes 0.5
    try:
        return float(existing.get("confidence", 0.0)) >= 1.0
    except (TypeError, ValueError):
        return False
