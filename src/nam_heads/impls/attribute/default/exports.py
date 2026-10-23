"""What host stages read from the attribute head besides the head itself (IMPL_META "provides"; the
contract's exports are listed in nam_heads.sdk.exports)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def seed() -> List[Tuple[str, bytes]]:
    """The category store's seed: ``acat::<lemma>`` and ``acat::<lemma>::<ontology>``."""
    from .definitions import ATTRIBUTE_CATEGORIES, ONTOLOGY_CATEGORY_OVERRIDES
    out = []
    for category, lemmas in ATTRIBUTE_CATEGORIES.items():
        for lemma in lemmas:
            out.append((f"acat::{lemma}", json.dumps({"category": category, "confidence": 1.0}).encode("utf-8")))
    for ontology_type, overrides in ONTOLOGY_CATEGORY_OVERRIDES.items():
        for lemma, category in overrides.items():
            out.append((f"acat::{lemma}::{ontology_type}",
                        json.dumps({"category": category, "confidence": 1.0}).encode("utf-8")))
    return out


def axis_encoder():
    """This head's own encoder, from its own package (``encoder/charcnn``)."""
    from src.encoders.axis_encoder import package_axis_encoder
    return package_axis_encoder("attribute", Path(__file__).resolve().parent / "encoder" / "charcnn")
