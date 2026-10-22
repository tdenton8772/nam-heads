"""Seed of the context store for the particular-evidence extractor.

The sealed artifact is ``data/places.json`` (GeoNames + curated places and
demonyms, see ``data/generate_places.py``) together with the tables in
``institutions.py`` (head words, the rule backend's organization list),
``temporal.py`` (named periods) and ``domains.py`` (field triggers). They are
loaded ONLY here, into the store, through the store's seeding path
(``seed_batch`` writes missing keys only, so learned and existing entries are
never overwritten). Extraction reads the store, never these tables.
"""

import json
import logging
from functools import lru_cache
from typing import List, Tuple

logger = logging.getLogger(__name__)

Entry = Tuple[str, str, str, float]  # (phrase, dimension, canonical_value, confidence)


@lru_cache(maxsize=1)
def seed_entries() -> Tuple[Entry, ...]:
    """One store document per surface (``cx:s:``) and per token (``cx:t:``);
    layout in lexicon.py. Includes the current head's tables (legacy.py)."""
    from .places import gazetteer, _window_key  # noqa: F401
    from .institutions import HEAD_WORDS, org_gazetteer, slug
    from .temporal import NAMED_PERIODS, canonical as canonical_period
    from .domains import TRIGGERS
    from .lexicon import DIM
    from . import legacy
    places, demonyms = gazetteer()
    surf: dict = {}
    tok: dict = {}
    for k, cands in places.items():
        surf.setdefault(k, {})["place"] = cands
    # Current head's GEOGRAPHIC_SEED: surfaces the gazetteer lacks point at
    # the place their canonical value names.
    for surface, canonical in legacy.geographic_seed().items():
        if surface in places:
            continue
        target = places.get(canonical.replace("_", " "))
        if target:
            surf.setdefault(surface, {})["place"] = target
    for k, v in demonyms.items():
        surf.setdefault(k, {})["dem"] = v
    for words, name in org_gazetteer().items():
        exact = " ".join(words)
        surf.setdefault(exact.lower(), {}).setdefault("org", {})[exact] = slug(name)
    # One period, one value: legacy names and era words resolve to the
    # period's canonical slug and path (temporal.canonical).
    periods = {k.lower(): [path, era] for k, (path, era) in NAMED_PERIODS.items()}
    for k, (path, era) in legacy.named_periods().items():
        periods.setdefault(k, list(canonical_period(path, era)))
    for k, pe in periods.items():
        surf.setdefault(k, {})["period"] = pe
    heads = {w: None for w in HEAD_WORDS}
    heads.update(legacy.institutional_heads())
    for w, kind in sorted(heads.items()):
        d = tok.setdefault(w.lower(), {})
        d.setdefault("head", []).append(w)
        if kind:
            d["kind"] = kind
    for word, field in TRIGGERS.items():
        tok.setdefault(word, {})["field"] = field
    for word, field in legacy.keyword_fields().items():
        tok.setdefault(word, {})["lfield"] = field
    for word, era in legacy.era_words().items():
        path, era = canonical_period("", era)
        tok.setdefault(word, {})["era"] = era
        if path:
            tok[word]["era_path"] = path
    for label, field in legacy.ner_domain_fields().items():
        tok.setdefault(f"ner:{label.lower()}", {})["field"] = field
    out: List[Entry] = []
    for k, d in sorted(surf.items()):
        out.append((f"cx:s:{k}", DIM, json.dumps(d, sort_keys=True, separators=(",", ":")), 1.0))
    for k, d in sorted(tok.items()):
        out.append((f"cx:t:{k}", DIM, json.dumps(d, sort_keys=True, separators=(",", ":")), 1.0))
    return tuple(out)


def seed_store(store) -> int:
    entries = seed_entries()
    if hasattr(store, "seed_batch"):
        added = store.seed_batch(entries)
    else:
        added = 0
        for phrase, dim, value, conf in entries:
            if not store.has(phrase, dim):
                store.put(phrase, dim, value, confidence=conf)
                added += 1
    if added:
        logger.info("Seeded context-axis store with %d entries", added)
    return added


def seed_kv_payloads() -> List[Tuple[str, bytes]]:
    """(key, payload) pairs in the KV store's own format, for supervisor/seeds.py."""
    return [(f"ctx::{p}::{d}", json.dumps({"canonical_value": v, "confidence": c}).encode())
            for p, d, v, c in seed_entries()]
