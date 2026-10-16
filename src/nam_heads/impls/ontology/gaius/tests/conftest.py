"""Fixtures for the gaius head tests.

Most tests run the head on a FAKE late-interaction encoder (hashed token
vectors), so the contract, determinism and abstain tests run anywhere. Tests
marked ``needs_model`` use the real ColBERT-Zero ONNX export; they skip when
it cannot be found, unless NAM_GAIUS_REQUIRED=1 makes that a failure.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")


class FakeEncoder:
    """Deterministic token vectors: each whitespace token hashes to a unit vector."""

    query_length = 39
    document_length = 519

    def __init__(self, model_dir=None, max_threads=None):
        self.model_dir = Path("/fake-colbert")
        self.meta = {"model": "fake", "onnx_sha256": "fake", "precision": "fp32", "source": {}}
        self.calls = 0

    @staticmethod
    def _vec(tok: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(tok.lower().encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(128).astype(np.float32)
        return v / np.linalg.norm(v)

    def _encode(self, text: str, marker: str) -> np.ndarray:
        toks = [marker] + [t.strip(".,;:()") for t in text.split()][:37]
        return np.stack([self._vec(t) for t in toks if t])

    def encode_query(self, text, max_length=None):
        self.calls += 1
        return self._encode(text, "[Q]")

    def encode_document(self, text):
        return self._encode(text, "[D]")


def model_available() -> bool:
    from nam_heads.impls.ontology.gaius.encoder import EncoderError, resolve_model_dir

    try:
        resolve_model_dir()
        return True
    except EncoderError:
        return False


@pytest.fixture
def fake_head(monkeypatch):
    """Factory: a GaiusOntologyEncoder on the fake encoder (fresh shared caches)."""
    from nam_heads.impls.ontology.gaius import head as H

    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "LateInteractionEncoder", FakeEncoder)
    monkeypatch.setattr(H.GaiusOntologyEncoder, "_check_model", lambda self, cal: None)

    def make(**config):
        return H.GaiusOntologyEncoder(head_type="ontology", config=config or None)
    return make


@pytest.fixture(scope="session")
def nlp():
    from nam_heads.sdk.testing import parser_or_skip
    return parser_or_skip()


@pytest.fixture(scope="session")
def real_head():
    if not model_available():
        if os.environ.get("NAM_GAIUS_REQUIRED") == "1":
            pytest.fail("NAM_GAIUS_REQUIRED=1 but the ColBERT-Zero ONNX model was not found "
                        "(set NAM_GAIUS_MODEL_DIR)")
        pytest.skip("ColBERT-Zero ONNX model not found (set NAM_GAIUS_MODEL_DIR)")
    from nam_heads.impls.ontology.gaius.head import GaiusOntologyEncoder
    return GaiusOntologyEncoder(head_type="ontology", config={"max_threads": 1})
