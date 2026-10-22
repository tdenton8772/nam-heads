"""Tests for entity codebook rejection types."""

import numpy as np
import pytest
import tempfile
import os

from nam_heads.impls.entity.default.codebook.lookup import (
    EntityCodebook,
    REJECTION_TYPES,
    _char_entropy,
)


def _make_test_codebook(cluster_labels, k=None):
    """Build a temporary codebook .npz for testing."""
    if k is None:
        k = len(cluster_labels)
    dim = 60
    centroids = np.random.randn(k, dim).astype(np.float32)
    feature_means = np.zeros(dim, dtype=np.float32)
    feature_stds = np.ones(dim, dtype=np.float32)

    # Match the real codebook's word lists so _build_features produces 60-dim vectors
    suffixes = ["tion", "ment", "ism", "ist", "ity", "ness", "ance", "ence",
                "ing", "ous", "ive", "able"]  # 12 suffixes
    person_bigrams = ["an", "ar", "el", "er", "on", "en", "al", "in", "is", "os"]  # 10
    place_bigrams = ["bu", "st", "la", "vi", "to", "po", "ri", "be", "ca", "ma"]  # 10

    tmpf = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez_compressed(
        tmpf.name,
        centroids=centroids,
        cluster_labels=np.array(cluster_labels),
        feature_means=feature_means,
        feature_stds=feature_stds,
        k=k,
        feature_dim=dim,
        n_training_samples=1000,
        suffixes=np.array(suffixes),
        location_words=np.array(["city", "river", "lake", "mountain", "island"]),
        person_words=np.array(["mr", "mrs", "dr", "sir", "lord"]),
        event_words=np.array(["war", "battle", "revolution", "election"]),
        time_words=np.array(["century", "decade", "era", "period"]),
        org_words=np.array(["university", "company", "institute", "agency"]),
        person_bigrams=np.array(person_bigrams),
        place_bigrams=np.array(place_bigrams),
    )
    return tmpf.name


class TestRejectionTypes:
    """Verify REJECTION_TYPES constant and is_rejection_type method."""

    def test_rejection_types_exist(self):
        assert "rejection_generic_noun" in REJECTION_TYPES
        assert "rejection_low_salience" in REJECTION_TYPES

    def test_non_rejection_types_not_in_set(self):
        assert "person" not in REJECTION_TYPES
        assert "location" not in REJECTION_TYPES
        assert "proper_noun" not in REJECTION_TYPES
        assert "full_name" not in REJECTION_TYPES

    def test_is_rejection_type_method(self):
        labels = ["person", "rejection_generic_noun", "location"]
        path = _make_test_codebook(labels, k=3)
        try:
            cb = EntityCodebook(path)
            assert cb.is_rejection_type("rejection_generic_noun") is True
            assert cb.is_rejection_type("rejection_low_salience") is True
            assert cb.is_rejection_type("person") is False
            assert cb.is_rejection_type("location") is False
        finally:
            os.unlink(path)

    def test_rejection_indices_built_on_load(self):
        labels = [
            "person", "rejection_generic_noun", "location",
            "rejection_low_salience", "proper_noun",
        ]
        path = _make_test_codebook(labels, k=5)
        try:
            cb = EntityCodebook(path)
            assert cb._rejection_indices == frozenset({1, 3})
        finally:
            os.unlink(path)

    def test_current_codebook_has_rejection_clusters(self):
        """After retraining (k=32), the codebook should have rejection clusters."""
        cb = EntityCodebook()
        rejection_labels = [l for l in cb.cluster_labels if l in REJECTION_TYPES]
        assert len(rejection_labels) >= 1, \
            "Retrained codebook should have at least one rejection cluster"
        assert len(cb._rejection_indices) >= 1
        assert cb.k == 32


class TestBatchClassifyWithRejection:
    """Verify batch classification correctly labels rejection clusters."""

    def test_batch_classify_returns_rejection_labels(self):
        labels = ["person", "rejection_generic_noun"]
        path = _make_test_codebook(labels, k=2)
        try:
            cb = EntityCodebook(path)
            # Build features that will land near each centroid
            # (features are random so just test the pipeline works)
            feat1 = cb._build_features("test", "NOUN", "test", 1, False, False, False, 0.3, 0.0)
            feat2 = cb._build_features("thing", "NOUN", "thing", 1, False, False, False, 0.2, 0.0)
            results = cb.classify_batch([feat1, feat2])
            assert len(results) == 2
            # Both should be valid labels from the codebook
            assert all(r in labels for r in results)
        finally:
            os.unlink(path)

    def test_known_entities_not_rejected(self):
        """Known entities classified with the production codebook must not be rejected."""
        cb = EntityCodebook()
        test_cases = [
            ("ankara", "PROPN", "Ankara", 1, True, False, False, 0.45),
            ("strom_thurmond", "PROPN_PROPN", "Strom Thurmond", 2, True, False, False, 0.60),
            ("background_radiation", "NOUN_NOUN", "background radiation", 2, False, False, False, 0.40),
            ("sinai_peninsula", "PROPN_PROPN", "Sinai Peninsula", 2, True, False, False, 0.60),
        ]
        for norm, pos, text, tc, propn, digit, hyphen, conf in test_cases:
            etype = cb.classify(norm, pos, text, tc, propn, digit, hyphen, conf)
            assert not cb.is_rejection_type(etype), \
                f"{text} classified as {etype} (rejection type)"

    def test_generic_nouns_rejected(self):
        """Generic single-word nouns should be classified as rejection types."""
        cb = EntityCodebook()
        generic_nouns = [
            ("way", "NOUN", "way", 1, False, False, False, 0.25),
            ("time", "NOUN", "time", 1, False, False, False, 0.25),
            ("use", "NOUN", "use", 1, False, False, False, 0.25),
            ("form", "NOUN", "form", 1, False, False, False, 0.25),
        ]
        rejected_count = 0
        for norm, pos, text, tc, propn, digit, hyphen, conf in generic_nouns:
            etype = cb.classify(norm, pos, text, tc, propn, digit, hyphen, conf)
            if cb.is_rejection_type(etype):
                rejected_count += 1
        # At least 3 of 4 generic nouns should be rejected
        assert rejected_count >= 3, \
            f"Only {rejected_count}/4 generic nouns rejected"
