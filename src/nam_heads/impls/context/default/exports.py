"""What host stages read from the context head besides the head itself (IMPL_META "provides"; the
contract's exports are listed in nam_heads.sdk.exports)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


def seed() -> List[Tuple[str, bytes]]:
    """The context store's seed: the dimension tables (``ctx::<form>::geographic|temporal``,
    ``ctx::domain::<ontology>::domain``, ``ctx::inst::<keyword>::institutional``) and the particular-evidence
    extractor's ``ctx::cx:*`` keys (context_axis/seed.py)."""
    from .context_axis.seed import seed_kv_payloads
    from .definitions import DOMAIN_SEEDS, GEOGRAPHIC_SEED, INSTITUTIONAL_KEYWORDS, TEMPORAL_PERIODS

    def entry(key, value):
        return key, json.dumps({"canonical_value": value, "confidence": 1.0}).encode("utf-8")

    out = [entry(f"ctx::{form}::geographic", c) for form, c in GEOGRAPHIC_SEED.items()]
    out += [entry(f"ctx::{form}::temporal", c) for form, c in TEMPORAL_PERIODS.items()]
    out += [entry(f"ctx::domain::{ont}::domain", label) for ont, label in DOMAIN_SEEDS.items()]
    out += [entry(f"ctx::inst::{kw}::institutional", inst)
            for inst, kws in INSTITUTIONAL_KEYWORDS.items() for kw in kws.split()]
    return out + list(seed_kv_payloads())


def axis_encoder():
    """This head's own encoder, from its own package (``encoder/charcnn``)."""
    from src.encoders.axis_encoder import package_axis_encoder
    return package_axis_encoder("context", Path(__file__).resolve().parent / "encoder" / "charcnn")
