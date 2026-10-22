"""
Context dimension normalizers.

One function per dimension. Each takes raw text + store, returns canonical
value or None. Results are cached in the store for fast subsequent lookups.
"""

from typing import Optional, Set
from nam_heads.ontology import bindings as B

from .definitions import (
    normalize_temporal_rules,
    DOMAIN_SEEDS,
    DOMAIN_NER_OVERRIDES,
    DOMAIN_ORG_KEYWORDS,
    INSTITUTIONAL_KEYWORDS,
    INSTITUTIONAL_DEFAULT,
)
from .storage.base import ContextStore


def normalize_temporal(ent_text: str, store: ContextStore) -> Optional[str]:
    """Normalize a DATE/TIME/EVENT entity to a canonical temporal value.

    1. Check store for known mapping (fast path).
    2. Try regex rules (year->decade, century, decade, year range).
    3. Store result if found.
    """
    key = ent_text.lower().strip()

    # Fast path: store lookup
    cached = store.get(key, "temporal")
    if cached is not None:
        return cached

    # Rule-based normalization
    result = normalize_temporal_rules(ent_text)
    if result is not None:
        store.put(key, "temporal", result, confidence=0.9)
        return result

    return None


def normalize_geographic(ent_text: str, store: ContextStore) -> Optional[str]:
    """Normalize a GPE/LOC/FAC entity to a canonical geographic value.

    1. Check store for known mapping (fast path -- includes seeded countries).
    2. Return None for unknown places (will be classified when store grows).
    """
    key = ent_text.lower().strip()

    cached = store.get(key, "geographic")
    if cached is not None:
        return cached

    return None


def normalize_domain(
    ontology_type: Optional[str],
    ner_labels: Set[str],
    text: str,
    store: ContextStore,
) -> Optional[str]:
    """Derive a domain label from ontology type + NER entity types.

    Priority:
    1. NER-type overrides (EVENT->"historical", WORK_OF_ART->"cultural").
    2. ORG keyword refinement (university->"academic", army->"military").
    3. Ontology type -> domain mapping from seeds.
    """
    # NER-type overrides take priority
    for ner_label, domain in DOMAIN_NER_OVERRIDES.items():
        if ner_label in ner_labels:
            return domain

    # ORG keyword refinement
    if ontology_type == B.ORGANIZATION and "ORG" in ner_labels:
        text_lower = text.lower()
        for domain, keywords_str in DOMAIN_ORG_KEYWORDS.items():
            for keyword in keywords_str.split():
                if keyword in text_lower:
                    return domain

    # Ontology type -> domain from seeds
    if ontology_type:
        key = f"domain::{ontology_type}"
        cached = store.get(key, "domain")
        if cached is not None:
            return cached
        # Direct lookup in static seeds as fallback
        if ontology_type in DOMAIN_SEEDS:
            return DOMAIN_SEEDS[ontology_type]

    return None


def normalize_institutional(ent_text: str, store: ContextStore) -> Optional[str]:
    """Normalize an ORG entity to an institutional type.

    1. Check store for known org name.
    2. Keyword scan against institutional seeds.
    3. Default to "commercial" for unmatched ORG entities.
    """
    key = ent_text.lower().strip()

    # Fast path: store lookup
    cached = store.get(key, "institutional")
    if cached is not None:
        return cached

    # Keyword scan
    for inst_type, keywords_str in INSTITUTIONAL_KEYWORDS.items():
        for keyword in keywords_str.split():
            if keyword in key:
                store.put(key, "institutional", inst_type, confidence=0.85)
                return inst_type

    # Default for unmatched ORG
    store.put(key, "institutional", INSTITUTIONAL_DEFAULT, confidence=0.5)
    return INSTITUTIONAL_DEFAULT
