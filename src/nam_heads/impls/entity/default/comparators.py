"""
Type-dispatched entity comparators for fuzzy resolution.

Each comparator handles a specific ontology type (person, organization, etc.)
and provides:
- generate_keys(): candidate keys for reverse-index lookup
- match(): similarity score between two surface forms
- threshold(): minimum score for positive match
"""
from abc import ABC, abstractmethod
from typing import Dict, List
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

try:
    from rapidfuzz.distance import JaroWinkler
    _jaro_winkler = JaroWinkler.similarity
except ImportError:
    _jaro_winkler = None
    logger.warning("rapidfuzz not available; fuzzy matching disabled")

try:
    from metaphone import doublemetaphone
    _double_metaphone = doublemetaphone
except ImportError:
    _double_metaphone = None
    logger.info("metaphone not available; using Soundex fallback for phonetic keys")


def _normalize_for_key(surface: str) -> str:
    """Unicode-safe lowercase normalization for candidate keys."""
    text = unicodedata.normalize("NFD", surface)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = text.lower().strip()
    # Convert underscores to spaces so multi-word names tokenize correctly
    # (e.g., "albert_einstein" -> "albert einstein" -> per-token phonetic keys)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _soundex(word: str) -> str:
    """Simple Soundex implementation as phonetic fallback."""
    if not word:
        return ""
    word = word.upper()
    code = word[0]
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    prev = mapping.get(word[0], "0")
    for ch in word[1:]:
        digit = mapping.get(ch, "0")
        if digit != "0" and digit != prev:
            code += digit
        prev = digit
        if len(code) == 4:
            break
    return code.ljust(4, "0")


def _phonetic_keys(word: str) -> List[str]:
    """Generate phonetic keys using Double Metaphone or Soundex fallback."""
    if _double_metaphone is not None:
        primary, secondary = _double_metaphone(word)
        keys = []
        if primary:
            keys.append(f"ph:{primary}")
        if secondary and secondary != primary:
            keys.append(f"ph:{secondary}")
        return keys
    return [f"sx:{_soundex(word)}"]


class EntityComparator(ABC):
    """Base class for type-specific entity comparators."""

    @abstractmethod
    def generate_keys(self, surface: str) -> List[str]:
        """Generate candidate keys for reverse-index lookup."""

    @abstractmethod
    def match(self, surface: str, candidate_surface: str) -> float:
        """Similarity score 0.0-1.0 between two surface forms."""

    @abstractmethod
    def threshold(self) -> float:
        """Minimum score for positive match."""


class PersonComparator(EntityComparator):
    """Person entities: Jaro-Winkler + phonetic keys + sorted-token reordering."""

    def generate_keys(self, surface: str) -> List[str]:
        norm = _normalize_for_key(surface)
        tokens = norm.split()
        keys = []
        # Sorted-token key (handles name reordering: "einstein albert" == "albert einstein")
        if tokens:
            keys.append(f"sorted:{'_'.join(sorted(tokens))}")
        # Phonetic keys for each token
        for tok in tokens:
            keys.extend(_phonetic_keys(tok))
        return keys

    def match(self, surface: str, candidate_surface: str) -> float:
        a_norm = _normalize_for_key(surface)
        b_norm = _normalize_for_key(candidate_surface)
        a_tokens = a_norm.split()
        b_tokens = b_norm.split()

        # Guard: multi-word names with different token counts are different entities
        # ("hans albert einstein" != "albert einstein")
        if len(a_tokens) >= 2 and len(b_tokens) >= 2 and len(a_tokens) != len(b_tokens):
            return 0.0

        # Guard: multi-word names must share a similar first token
        # ("bose einstein" != "albert einstein" -- first tokens don't match)
        if len(a_tokens) >= 2 and len(b_tokens) >= 2:
            first_sim = (_jaro_winkler(a_tokens[0], b_tokens[0])
                         if _jaro_winkler else (1.0 if a_tokens[0] == b_tokens[0] else 0.0))
            if first_sim < 0.65:
                return 0.0

        if _jaro_winkler is None:
            return 1.0 if a_norm == b_norm else 0.0
        score = _jaro_winkler(a_norm, b_norm)
        # Bonus for sorted-token match (handles reordering)
        a_sorted = " ".join(sorted(a_tokens))
        b_sorted = " ".join(sorted(b_tokens))
        sorted_score = _jaro_winkler(a_sorted, b_sorted)
        return max(score, sorted_score)

    def threshold(self) -> float:
        return 0.88


class OrganizationComparator(EntityComparator):
    """Organization entities: Jaro-Winkler + abbreviation keys."""

    def generate_keys(self, surface: str) -> List[str]:
        norm = _normalize_for_key(surface)
        tokens = norm.split()
        keys = []
        # Full normalized key
        if norm:
            keys.append(f"norm:{norm.replace(' ', '_')}")
        # Abbreviation key (first letters)
        if len(tokens) >= 2:
            abbrev = "".join(t[0] for t in tokens if t)
            keys.append(f"abbr:{abbrev}")
        # Phonetic on each token
        for tok in tokens:
            keys.extend(_phonetic_keys(tok))
        return keys

    def match(self, surface: str, candidate_surface: str) -> float:
        if _jaro_winkler is None:
            return 1.0 if _normalize_for_key(surface) == _normalize_for_key(candidate_surface) else 0.0
        return _jaro_winkler(_normalize_for_key(surface), _normalize_for_key(candidate_surface))

    def threshold(self) -> float:
        return 0.85


class GroupComparator(EntityComparator):
    """Group entities: delegates to OrganizationComparator."""

    def __init__(self):
        self._delegate = OrganizationComparator()

    def generate_keys(self, surface: str) -> List[str]:
        return self._delegate.generate_keys(surface)

    def match(self, surface: str, candidate_surface: str) -> float:
        return self._delegate.match(surface, candidate_surface)

    def threshold(self) -> float:
        return 0.85


class DefaultComparator(EntityComparator):
    """Default comparator: normalized exact + token overlap ratio."""

    def generate_keys(self, surface: str) -> List[str]:
        norm = _normalize_for_key(surface)
        keys = []
        if norm:
            keys.append(f"norm:{norm.replace(' ', '_')}")
        return keys

    def match(self, surface: str, candidate_surface: str) -> float:
        a = _normalize_for_key(surface)
        b = _normalize_for_key(candidate_surface)
        if a == b:
            return 1.0
        # Token overlap ratio
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        overlap = len(a_tokens & b_tokens)
        return overlap / max(len(a_tokens), len(b_tokens))

    def threshold(self) -> float:
        return 0.90


# Registry keyed by entity type (from codebook or legacy)
_person = PersonComparator()
_org = OrganizationComparator()
_default = DefaultComparator()

COMPARATOR_REGISTRY: Dict[str, EntityComparator] = {
    # Legacy types (backwards compatibility)
    "person": _person,
    "group": GroupComparator(),
    "organization": _org,
    "verb": _default,
    # Codebook-discovered types
    "full_name": _person,
    "titled_entity": _person,
    "proper_noun": _person,
    "acronym": _org,
}

_DEFAULT_COMPARATOR = DefaultComparator()


def get_comparator(entity_type: str) -> EntityComparator:
    """Get comparator for entity type with default fallback."""
    return COMPARATOR_REGISTRY.get(entity_type, _DEFAULT_COMPARATOR)
