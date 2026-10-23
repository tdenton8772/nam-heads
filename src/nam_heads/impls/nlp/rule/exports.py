"""What host stages read from the rule parser besides the parser itself (IMPL_META "provides"; the
contract's exports are listed in nam_heads.sdk.exports): its tagger dictionary and NER gazetteers, for the
sentence sketch's lexicon."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, FrozenSet

_DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=1)
def pos_dictionary() -> Dict[str, str]:
    """The tagger's dictionary: word -> most frequent POS."""
    with open(_DATA / "pos_dictionary.json") as f:
        return json.load(f)


def _gazetteer(name: str) -> FrozenSet[str]:
    with open(_DATA / "gazetteers" / name) as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


@lru_cache(maxsize=1)
def countries() -> FrozenSet[str]:
    """Country names of the NER gazetteer, lower case."""
    return _gazetteer("countries.txt")


@lru_cache(maxsize=1)
def cities() -> FrozenSet[str]:
    """City names of the NER gazetteer, lower case."""
    return _gazetteer("cities.txt")
