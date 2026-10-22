"""
Attribute vocabulary: seed categories and ontology relevance weights.

``ATTRIBUTE_CATEGORIES`` is **seed data** used to bootstrap the category store
on first run.  At runtime the store (memory, SQLite, or eventually Couchbase)
is the authoritative source for lemma -> category mappings.  New adjectives
discovered during ingest can be classified and added to the store without
redeploying code.

``ONTOLOGY_RELEVANCE`` is a static config -- it maps ontology types to
per-category score multipliers and is unlikely to change at runtime.
"""

from typing import Dict, Optional, Set

from nam_heads.ontology import bindings as B

# ---------------------------------------------------------------------------
# Seed categories -- used by ``storage.factory._seed_store()`` to bootstrap
# a fresh category store.  ~200 lemmas across 7 semantic groups.
# ---------------------------------------------------------------------------
ATTRIBUTE_CATEGORIES: Dict[str, Set[str]] = {
    "physical": {
        "large", "small", "tall", "short", "heavy", "light", "round", "flat",
        "smooth", "rough", "hard", "soft", "sharp", "thick", "thin",
        "red", "blue", "green", "yellow", "black", "white", "dark", "bright",
        "hot", "cold", "warm", "cool", "wet", "dry", "clean", "dirty",
        "broken", "solid", "hollow", "narrow", "wide", "deep", "shallow",
        "dense", "loose", "tight", "stiff", "flexible", "opaque", "transparent",
    },
    "emotional": {
        "happy", "sad", "angry", "calm", "brave", "kind", "cruel", "shy",
        "friendly", "hostile", "loyal", "proud", "humble", "generous", "greedy",
        "jealous", "grateful", "anxious", "cheerful", "miserable", "content",
        "furious", "gentle", "fierce", "passionate", "indifferent", "eager",
    },
    "temporal": {
        "old", "young", "new", "recent", "ancient", "modern", "brief",
        "permanent", "frequent", "rare", "early", "late", "immediate",
        "gradual", "sudden", "continuous", "occasional", "daily", "annual",
        "temporary", "chronic", "prolonged", "swift", "slow",
    },
    "evaluative": {
        "excellent", "poor", "important", "trivial", "easy", "difficult",
        "significant", "great", "terrible", "wonderful", "awful", "superb",
        "mediocre", "outstanding", "brilliant", "dreadful", "remarkable",
        "ordinary", "impressive", "disappointing", "valuable", "worthless",
        "effective", "useless", "successful", "failed",
    },
    "spatial": {
        "near", "far", "central", "remote", "northern", "southern",
        "eastern", "western", "upper", "lower", "inner", "outer",
        "left", "right", "front", "rear", "adjacent", "distant",
        "fast", "rapid", "quick", "widespread", "scattered", "dense",
        "vertical", "horizontal", "parallel", "diagonal",
    },
    "institutional": {
        "global", "local", "official", "public", "private", "formal",
        "informal", "legal", "illegal", "democratic", "military",
        "political", "economic", "academic", "professional", "commercial",
        "industrial", "domestic", "foreign", "national", "international",
        "active", "passive", "efficient", "bureaucratic",
    },
    "relational": {
        "similar", "different", "unique", "common", "primary", "secondary",
        "equal", "unequal", "relative", "absolute", "mutual", "independent",
        "dependent", "dominant", "subordinate", "compatible", "contrary",
        "typical", "atypical", "standard", "alternative", "comparable",
    },
}

# ---------------------------------------------------------------------------
# Ontology relevance weights  (multiplier applied to base score)
#
# Range 0.8-1.3: conservative nudge, never hard-filters.
# Unlisted ontology types or ``'uncategorized'`` adjectives -> 1.0 (neutral).
#
# Besides the lemma categories (physical, emotional, ...), the predication
# path scores three EVIDENCE categories -- what kind of evidence a reading
# is: a noun naming a kind of person ("role"), another predicate noun
# ("kind"), a nationality ("nationality"). Their weights nudge like any
# other. Excluding evidence outright is a separate, explicit rule:
# EVIDENCE_EXCLUSIONS below.
#
# Tables keyed by ontology class resolve through nam_heads.ontology.lookup (nearest
# of ONTOLOGY_KEYS, subclasses inherit, bfo:entity is neutral).
# ---------------------------------------------------------------------------
EVIDENCE_CATEGORIES = ("role", "kind", "nationality")
_NON_AGENT = {"role": 0.8, "kind": 1.1}
ONTOLOGY_RELEVANCE: Dict[str, Dict[str, float]] = {
    B.PERSON:       {"emotional": 1.3, "relational": 1.1, "physical": 0.9, "institutional": 0.9,
                     "role": 1.3, "nationality": 1.2, "kind": 0.8},  # person
    B.ORGANIZATION: {"institutional": 1.3, "relational": 1.1, "physical": 0.8, "emotional": 0.9},  # organization
    B.GEOSPATIAL_REGION:     {"spatial": 1.3, "physical": 1.1, "emotional": 0.8, **_NON_AGENT},  # location
    B.GEOSPATIAL_LOCATION:        {"spatial": 1.3, "physical": 1.1, "emotional": 0.8, **_NON_AGENT},  # place
    B.MATERIAL_ARTIFACT:       {"physical": 1.3, "spatial": 1.1, "emotional": 0.8, **_NON_AGENT},  # object
    B.ACT:       {"temporal": 1.2, "evaluative": 1.1, **_NON_AGENT},  # action
    B.PROCESS:        {"temporal": 1.2, "evaluative": 1.1, **_NON_AGENT},  # event
    B.STASIS:        {"emotional": 1.2, "temporal": 1.1, "physical": 0.9, **_NON_AGENT},  # state
    B.GROUP_OF_PERSONS:        {"institutional": 1.2, "relational": 1.1, "emotional": 0.9},  # group
    B.TEMPORAL_REGION:         {"temporal": 1.3, "spatial": 0.9, **_NON_AGENT},  # time
    # Extended ontology types
    # "process" binds to bfo:process like "event"; event's row is kept
    B.CHANGE:       {"temporal": 1.2, "evaluative": 1.1, **_NON_AGENT},  # change
    B.CAUSE:        {"evaluative": 1.2, "relational": 1.1, **_NON_AGENT},  # cause
    B.EFFECT:       {"evaluative": 1.2, "relational": 1.1, **_NON_AGENT},  # effect
    B.QUALITY:    {"evaluative": 1.1, "physical": 1.1, **_NON_AGENT},  # attribute
    B.MEASUREMENT_ICE:     {"evaluative": 1.1, "spatial": 1.0, **_NON_AGENT},  # quantity
    B.RELATIONAL_QUALITY: {"relational": 1.3, "institutional": 1.1, **_NON_AGENT},  # relationship
    B.ACT_OF_OWNERSHIP:    {"relational": 1.2, "institutional": 1.1, **_NON_AGENT},  # ownership
    B.ORDINAL_MEASUREMENT_ICE:     {"temporal": 1.3, "spatial": 1.0, **_NON_AGENT},  # sequence
    B.DESCRIPTIVE_ICE:       {"temporal": 1.1, "institutional": 1.1, **_NON_AGENT},  # record
    B.DOCUMENT_CONTENT_ENTITY:     {"institutional": 1.1, "temporal": 1.0, **_NON_AGENT},  # document
    B.DESIGNATIVE_ICE:   {"institutional": 1.1, **_NON_AGENT},  # identifier
    B.OBJECTIVE:         {"evaluative": 1.2, "temporal": 1.1, **_NON_AGENT},  # goal
    B.PRESCRIPTIVE_ICE:    {"evaluative": 1.2, "emotional": 1.1, "physical": 1.0, **_NON_AGENT},  # condition
}

# ---------------------------------------------------------------------------
# Evidence exclusions -- an explicit HARD filter, separate from the nudges.
#
# Evidence categories that do not count as an attribute of an entity of
# that ontology at all: a role noun ("president") is not an attribute of a
# place, artifact, process or information entity; a predicate noun that
# names no kind of person ("film") is not an attribute of a person.
# Applied by the head (per sentence ontology) and the bundler (per entity
# ontology). Resolved like ONTOLOGY_RELEVANCE. OFF by default
# (evidence_exclusions_enabled): config ``evidence_exclusions`` /
# NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS=1 turns it on for the head and the bundler;
# NAM_ATTRIBUTE_CONDITIONING=1/0 overrides it for the bundler alone.
# ---------------------------------------------------------------------------
_NO_ROLES = frozenset({"role"})
EVIDENCE_EXCLUSIONS: Dict[str, frozenset] = {
    B.PERSON: frozenset({"kind"}),
    **{o: _NO_ROLES for o in (
        B.GEOSPATIAL_REGION, B.GEOSPATIAL_LOCATION, B.MATERIAL_ARTIFACT, B.ACT, B.PROCESS,
        B.STASIS, B.TEMPORAL_REGION, B.CHANGE, B.CAUSE, B.EFFECT, B.QUALITY, B.MEASUREMENT_ICE,
        B.RELATIONAL_QUALITY, B.ACT_OF_OWNERSHIP, B.ORDINAL_MEASUREMENT_ICE, B.DESCRIPTIVE_ICE,
        B.DOCUMENT_CONTENT_ENTITY, B.DESIGNATIVE_ICE, B.OBJECTIVE, B.PRESCRIPTIVE_ICE)},
}

# ---------------------------------------------------------------------------
# Ontology-specific category overrides -- seeded into the store so that a
# lemma can map to a different category depending on ontology context.
# ---------------------------------------------------------------------------
ONTOLOGY_CATEGORY_OVERRIDES: Dict[str, Dict[str, str]] = {
    B.PERSON: {  # person
        "private": "emotional",
        "active": "physical",
        "civil": "relational",
    },
    B.ORGANIZATION: {  # organization
        "private": "institutional",
        "active": "institutional",
    },
    B.TEMPORAL_REGION: {  # time
        "old": "temporal",
        "new": "temporal",
        "recent": "temporal",
    },
}

# ---------------------------------------------------------------------------
# Stopword exemptions per ontology type -- these stopwords get a reduced
# penalty (0.1 instead of 0.3) when the given ontology type is active.
# ---------------------------------------------------------------------------
STOPWORD_ONTOLOGY_EXEMPTIONS: Dict[str, Set[str]] = {
    B.TEMPORAL_REGION:         {"old", "new", "recent", "early", "late"},  # time
    B.STASIS:        {"old", "new", "different", "same", "real"},  # state
    B.PROCESS:        {"main", "various", "particular"},  # event
    B.ORGANIZATION: {"public", "general", "main"},  # organization
}

# ---------------------------------------------------------------------------
# NER entity type relevance -- multipliers applied to attribute scores based
# on the parse's NER labels detected in the text.  Uses max() across labels.
# ---------------------------------------------------------------------------
NER_ENTITY_TYPE_RELEVANCE: Dict[str, Dict[str, float]] = {
    "PERSON":  {"emotional": 1.2, "physical": 1.1, "relational": 1.1},
    "ORG":     {"institutional": 1.2, "relational": 1.1},
    "GPE":     {"spatial": 1.3, "physical": 1.1},
    "LOC":     {"spatial": 1.3, "physical": 1.1},
    "PRODUCT": {"physical": 1.3, "evaluative": 1.1},
}


from nam_heads.ontology.lookup import ontology_class, lookup as _lookup  # noqa: E402  (one resolver)


def entity_class(ontology) -> str:
    """ontology_class, or the BFO root (bfo:entity) when nothing resolves."""
    from nam_heads.ontology import bindings as B
    return ontology_class(ontology) or B.ENTITY


def relevance_row(ontology) -> Dict[str, float]:
    return ontology_lookup(ONTOLOGY_RELEVANCE, ontology) or {}


def excluded_categories(ontology) -> frozenset:
    return ontology_lookup(EVIDENCE_EXCLUSIONS, ontology, frozenset()) or frozenset()


def evidence_exclusions_enabled() -> bool:
    """EVIDENCE_EXCLUSIONS is OFF unless NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS=1.

    At current ontology quality the hard filter removed more correct values
    than junk (~17 of 25 removals were correct, axis measurement 2026-10-03),
    so the head and the bundler both default it off. The bundler's
    NAM_ATTRIBUTE_CONDITIONING, when set, overrides this for the bundler."""
    import os
    return os.environ.get("NAM_ATTRIBUTE_EVIDENCE_EXCLUSIONS", "0") not in ("0", "", "false")


def entity_admits(kinds, lemma_category: Optional[str], ontology) -> bool:
    """Does a value read as ``kinds`` count for an entity of ``ontology``?
    True unless every kind is excluded for it (EVIDENCE_EXCLUSIONS). A value
    with no kinds (legacy extraction) is admitted."""
    kinds = list(kinds or [])
    if not kinds or not ontology:
        return True
    excl = excluded_categories(ontology)
    return any(evidence_category(k, lemma_category) not in excl for k in kinds)


def get_ontology_relevance(ontology_type: str, category: str) -> float:
    """Return the relevance multiplier for *category* under *ontology_type*.

    Returns 1.0 (neutral) for unknown ontology types or categories.
    """
    return relevance_row(ontology_type).get(category, 1.0)


def evidence_category(kind: str, lemma_category: Optional[str]) -> Optional[str]:
    """The ONTOLOGY_RELEVANCE category of one piece of predication evidence."""
    if kind == "role_noun":
        return "role"
    if kind == "noun":
        return "kind"
    if kind == "norp":
        return "nationality"
    if lemma_category in (None, "uncategorized") or lemma_category in EVIDENCE_CATEGORIES:
        return None
    return lemma_category


def get_best_category_for_ontology(ontology_type: str) -> Optional[str]:
    """Return the category with the highest relevance weight for *ontology_type*.

    Used to auto-classify unknown adjectives when ontology context is available.
    Returns None if the ontology type has no relevance entries. Evidence
    categories are never a lemma's category.
    """
    weights = {k: v for k, v in relevance_row(ontology_type).items() if k not in EVIDENCE_CATEGORIES}
    if not weights:
        return None
    return max(weights, key=weights.get)


# Every BFO/CCO class a table above is keyed by. The ontology head emits any
# certified class; ``_get_ontology_type`` resolves it to the nearest of these
# (itself or an ancestor, never inheriting from bfo:entity), so a subclass such
# as City uses its Geospatial Region entries.
ONTOLOGY_KEYS = frozenset().union(
    ONTOLOGY_RELEVANCE, ONTOLOGY_CATEGORY_OVERRIDES, STOPWORD_ONTOLOGY_EXEMPTIONS, EVIDENCE_EXCLUSIONS)


def ontology_lookup(table: Dict, ontology: Optional[str], default=None):
    """``table``'s entry for an ontology class, resolved like
    ``_get_ontology_type`` (nearest of ONTOLOGY_KEYS, never bfo:entity);
    no ontology evidence gives ``default``."""
    return _lookup(table, ontology, default, keys=ONTOLOGY_KEYS)
