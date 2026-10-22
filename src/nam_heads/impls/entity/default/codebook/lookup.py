# Entity type classification via codebook nearest-centroid lookup.
#
# Deterministic: same input always produces the same entity type.
# No ML model at inference time — just feature extraction + L2 nearest neighbor.

import math
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_CODEBOOK_PATH = Path(__file__).parent / "entity_codebook.npz"

# Types that indicate non-entity noise. Candidates classified into these
# types are filtered out before entity creation/resolution.
# Updated when codebook is retrained with rejection clusters.
REJECTION_TYPES = frozenset({
    "rejection_generic_noun",
    "rejection_low_salience",
})


class EntityCodebook:
    """Codebook-based entity type classifier.

    Loads a pre-trained .npz artifact containing 25 centroids in 60-dim
    feature space. Each centroid maps to a discovered entity type label.

    At inference: extract features from normalized text + POS metadata,
    find nearest centroid, return entity type label + distance.
    """

    def __init__(self, path: Optional[str] = None):
        p = Path(path) if path else _CODEBOOK_PATH
        data = np.load(str(p), allow_pickle=True)
        self.centroids = data["centroids"].astype(np.float32)
        self.cluster_labels = list(data["cluster_labels"])
        self.feature_means = data["feature_means"].astype(np.float32)
        self.feature_stds = data["feature_stds"].astype(np.float32)
        self.k = int(data["k"])

        # Word lists for feature computation
        self.suffixes = list(data["suffixes"])
        self.location_words = set(data["location_words"])
        self.person_words = set(data["person_words"])
        self.event_words = set(data["event_words"])
        self.time_words = set(data["time_words"])
        self.org_words = set(data["org_words"])
        self.person_bigrams = list(data["person_bigrams"])
        self.place_bigrams = list(data["place_bigrams"])

        # Build rejection index from cluster labels
        self._rejection_indices = frozenset(
            i for i, label in enumerate(self.cluster_labels)
            if label in REJECTION_TYPES
        )

        n_rejection = len(self._rejection_indices)
        logger.info(
            "EntityCodebook loaded: k=%d, %d features, %d unique types, %d rejection clusters",
            self.k, self.centroids.shape[1], len(set(self.cluster_labels)), n_rejection,
        )

    def classify(
        self,
        normalized: str,
        pos_pattern: str = "UNK",
        text: str = "",
        token_count: int = 1,
        is_propn: bool = False,
        has_digit: bool = False,
        has_hyphen: bool = False,
        confidence: float = 0.0,
        log_freq: float = 0.0,
    ) -> str:
        """Classify an entity candidate into a type.

        All parameters are available from the noun chunk extraction output.
        log_freq should be log2 of the corpus frequency of this normalized
        form; pass 0.0 if unknown (first-time entity).

        Returns the entity type label string.
        """
        vec = self._build_features(
            normalized, pos_pattern, text, token_count,
            is_propn, has_digit, has_hyphen, confidence, log_freq,
        )
        vec_norm = (vec - self.feature_means) / self.feature_stds
        dists = np.sum((self.centroids - vec_norm) ** 2, axis=1)
        nearest = int(np.argmin(dists))
        return self.cluster_labels[nearest]

    def is_rejection_type(self, entity_type: str) -> bool:
        """Return True if entity_type is a rejection class (non-entity noise)."""
        return entity_type in REJECTION_TYPES

    def classify_batch(
        self,
        features_list: list,
    ) -> list:
        """Classify a batch of pre-built feature vectors.

        Each element should be the output of _build_features().
        Returns list of entity type labels.
        """
        if not features_list:
            return []
        X = np.array(features_list, dtype=np.float32)
        X_norm = (X - self.feature_means) / self.feature_stds
        # (n, k) distance matrix
        dists = np.sum(
            (X_norm[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        nearest = np.argmin(dists, axis=1)
        return [self.cluster_labels[int(i)] for i in nearest]

    def _build_features(
        self,
        normalized: str,
        pos_pattern: str,
        text: str,
        token_count: int,
        is_propn: bool,
        has_digit: bool,
        has_hyphen: bool,
        confidence: float,
        log_freq: float,
    ) -> np.ndarray:
        """Build a 60-dim feature vector from entity candidate metadata."""
        # POS bag (7)
        pos_counts = [0.0] * 7
        _POS_MAP = {"NOUN": 0, "PROPN": 1, "ADJ": 2, "DET": 3, "VERB": 4, "ADP": 5}
        if pos_pattern != "UNK":
            for tag in pos_pattern.split("_"):
                pos_counts[_POS_MAP.get(tag, 6)] += 1.0

        parts = normalized.split("_")
        n_parts = float(len(parts))
        char_len = float(len(normalized))
        last_word = parts[-1] if parts else ""
        last_word_len = float(len(last_word))

        suffix_feats = [
            1.0 if last_word.endswith(s) else 0.0 for s in self.suffixes
        ]

        alpha_chars = [c for c in normalized if c.isalpha()]
        vowel_ratio = (
            sum(1 for c in alpha_chars if c in _VOWELS) / max(len(alpha_chars), 1)
        )
        digit_ratio = (
            sum(1 for c in normalized if c.isdigit()) / max(len(normalized), 1)
        )
        is_acronym_val = 1.0 if text.isupper() and len(text) <= 6 else 0.0
        n_unique_chars = float(len(set(normalized.replace("_", ""))))

        entropy = _char_entropy(normalized.replace("_", ""))

        is_high_freq = 1.0 if log_freq > math.log2(1000) else 0.0

        any_location = 1.0 if any(p in self.location_words for p in parts) else 0.0
        any_person = 1.0 if any(p in self.person_words for p in parts) else 0.0
        any_event = 1.0 if any(p in self.event_words for p in parts) else 0.0
        any_time = 1.0 if any(p in self.time_words for p in parts) else 0.0
        any_org = 1.0 if any(p in self.org_words for p in parts) else 0.0
        is_pure_time = 1.0 if normalized in self.time_words else 0.0

        norm_alpha = normalized.replace("_", "")
        person_bgs = [float(norm_alpha.count(bg)) for bg in self.person_bigrams]
        place_bgs = [float(norm_alpha.count(bg)) for bg in self.place_bigrams]

        vec = np.array(
            pos_counts
            + [float(token_count), 1.0 if is_propn else 0.0,
               1.0 if has_digit else 0.0, 1.0 if has_hyphen else 0.0,
               float(confidence)]
            + [n_parts, char_len, last_word_len]
            + suffix_feats
            + [vowel_ratio, digit_ratio, is_acronym_val, n_unique_chars]
            + [entropy]
            + [log_freq, is_high_freq]
            + [any_location, any_person, any_event, any_time, any_org, is_pure_time]
            + person_bgs + place_bgs,
            dtype=np.float32,
        )
        return vec


_VOWELS = set("aeiou")


def _char_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


# Module-level singleton
_codebook: Optional[EntityCodebook] = None


def get_codebook() -> EntityCodebook:
    """Get or create the module-level codebook singleton."""
    global _codebook
    if _codebook is None:
        _codebook = EntityCodebook()
    return _codebook


def classify_entity(
    normalized: str,
    pos_pattern: str = "UNK",
    text: str = "",
    token_count: int = 1,
    is_propn: bool = False,
    has_digit: bool = False,
    has_hyphen: bool = False,
    confidence: float = 0.0,
    log_freq: float = 0.0,
) -> str:
    """Convenience function: classify a single entity candidate."""
    return get_codebook().classify(
        normalized, pos_pattern, text, token_count,
        is_propn, has_digit, has_hyphen, confidence, log_freq,
    )
