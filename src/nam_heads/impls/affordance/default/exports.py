"""What host stages read from the affordance head besides the head itself (IMPL_META "provides"; the
contract's exports are listed in nam_heads.sdk.exports)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def link_profile(ontology: Optional[str], conditioned: bool = True) -> Dict:
    """How predicates link to an entity of ``ontology``'s class; unconditioned, the default profile."""
    from .definitions import LINK_PROFILE_DEFAULT, ONTOLOGY_LINK_PROFILES, ontology_class, ontology_lookup
    if not conditioned:
        return dict(LINK_PROFILE_DEFAULT)
    prof = dict(ontology_lookup(ONTOLOGY_LINK_PROFILES, ontology, LINK_PROFILE_DEFAULT))
    prof["_class"] = ontology_class(ontology)
    return prof


def seed() -> List[Tuple[str, bytes]]:
    """The category store's seed: ``vcat::<lemma>`` and ``vcat::<lemma>::<ontology>``."""
    from .definitions import ONTOLOGY_CATEGORY_OVERRIDES, VERB_CATEGORIES
    out = []
    for category, lemmas in VERB_CATEGORIES.items():
        for lemma in lemmas:
            out.append((f"vcat::{lemma}", json.dumps({"category": category, "confidence": 1.0}).encode("utf-8")))
    for ontology_type, overrides in ONTOLOGY_CATEGORY_OVERRIDES.items():
        for lemma, category in overrides.items():
            out.append((f"vcat::{lemma}::{ontology_type}",
                        json.dumps({"category": category, "confidence": 1.0}).encode("utf-8")))
    return out


def axis_encoder():
    """This head's own encoder: the CharCNN in its package (``encoder/charcnn``), or with verb senses on
    (NAM_AFFORDANCE_SENSES=1) its sense table."""
    if os.getenv("NAM_AFFORDANCE_SENSES", "0") == "1":
        from .senses import SenseTableAxisEncoder, table
        return SenseTableAxisEncoder(table())
    from src.encoders.axis_encoder import package_axis_encoder
    return package_axis_encoder("affordance", Path(__file__).resolve().parent / "encoder" / "charcnn")
