"""Tests for batched entity type classification in encode_batch_texts.

Verifies that the batch classify optimization produces identical results
to per-candidate individual classification, and that it uses the vectorized
classify_batch path.
"""
import inspect
import os
import sys

import pytest


from nam_heads.impls.entity.default.head import EntityEncoder
from nam_heads.impls.entity.default.codebook.lookup import get_codebook, classify_entity


class TestBatchClassifyEquivalence:
    """Verify batch classify produces same types as individual classify."""

    def test_batch_matches_individual(self):
        """classify_batch must produce identical types to per-candidate classify."""
        cb = get_codebook()
        test_cases = [
            ("albert_einstein", "PROPN_PROPN", "Albert Einstein", 2, True, False, False, 0.95),
            ("machine_learning", "NOUN_NOUN", "machine learning", 2, False, False, False, 0.7),
            ("IBM", "PROPN", "IBM", 1, True, False, False, 0.85),
            ("new_york_city", "ADJ_PROPN_PROPN", "New York City", 3, True, False, False, 0.9),
            ("version_2.0", "NOUN_NOUN", "version 2.0", 2, False, True, False, 0.4),
            ("self-driving", "ADJ", "self-driving", 1, False, False, True, 0.6),
        ]

        # Individual classify (old path)
        individual_types = []
        for norm, pos, text, tc, propn, digit, hyphen, conf in test_cases:
            t = classify_entity(norm, pos, text, tc, propn, digit, hyphen, conf)
            individual_types.append(t)

        # Batch classify (new path)
        features = []
        for norm, pos, text, tc, propn, digit, hyphen, conf in test_cases:
            feat = cb._build_features(norm, pos, text, tc, propn, digit, hyphen, conf, 0.0)
            features.append(feat)
        batch_types = cb.classify_batch(features)

        assert batch_types == individual_types, \
            f"Batch types {batch_types} != individual {individual_types}"


class TestBatchClassifyInEncodeBatchTexts:
    """Verify encode_batch_texts uses batch classification."""

    def test_uses_classify_batch(self):
        """encode_batch_texts must call classify_batch, not per-candidate classify."""
        source = inspect.getsource(EntityEncoder.encode_batch_texts)
        assert "classify_batch" in source, \
            "encode_batch_texts must use classify_batch for vectorized classification"
        # Should NOT call _detect_type or classify_entity in p1_prepare loop
        # (they may still be called in fallback paths, but the main loop should batch)
        lines = source.split("\n")
        in_phase1 = False
        phase1_detect_type_calls = []
        for line in lines:
            stripped = line.strip()
            if "Phase 1:" in stripped:
                in_phase1 = True
                continue
            if "Phase 2:" in stripped:
                break
            if in_phase1 and "_detect_type(" in stripped and not stripped.startswith("#"):
                phase1_detect_type_calls.append(stripped)
        assert len(phase1_detect_type_calls) == 0, \
            f"Phase 1 should not call _detect_type per-candidate: {phase1_detect_type_calls}"

    def test_builds_features_for_batch(self):
        """encode_batch_texts must build feature vectors for batch classify."""
        source = inspect.getsource(EntityEncoder.encode_batch_texts)
        assert "_build_features" in source, \
            "encode_batch_texts must build feature vectors for batch classification"

    def test_empty_batch_no_error(self):
        """Empty text list should not crash batch classify."""
        enc = EntityEncoder(head_type="entity", config={
            "db_backend": "memory",
            "db_fallback": "memory",
        })
        results = enc.encode_batch_texts([], [])
        assert results == []
