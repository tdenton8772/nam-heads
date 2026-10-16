"""Fixtures for gaius_ref: a fake late-interaction encoder and a fake export dir.

The head tests run on hashed token vectors, so the contract, the gaius scoring
formula and the model pin are tested anywhere. ``real_head`` needs the pinned
lightonai/ColBERT-Zero export in $NAM_GAIUS_REF_MODEL_DIR; it skips without
one unless NAM_GAIUS_REF_REQUIRED=1.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest


class FakeEncoder:
    """Each whitespace token hashes to a unit vector; [Q]/[D] lead."""

    query_length = 39
    document_length = 519

    def __init__(self, model_dir, max_threads=None):
        self.model_dir = Path(model_dir)
        self.queries = []

    @staticmethod
    def vec(tok: str) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(tok.lower().encode()).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(128).astype(np.float32)
        return v / np.linalg.norm(v)

    def _encode(self, text: str, marker: str) -> np.ndarray:
        toks = [marker] + [t.strip(".,;:()") for t in text.split()][:37]
        return np.stack([self.vec(t) for t in toks if t])

    def encode_query(self, text, max_length=None):
        self.queries.append(text)
        return self._encode(text, "[Q]")

    def encode_document(self, text):
        return self._encode(text, "[D]")


def write_export(d: Path, source: dict) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in ("model.onnx", "tokenizer.json", "config_sentence_transformers.json"):
        (d / f).write_text("{}")
    (d / "export.json").write_text(json.dumps({"precision": "fp32", "onnx_sha256": "fake",
                                               "source": source}))
    return d


@pytest.fixture(autouse=True)
def no_real_bank_cache(monkeypatch):
    """A fake encoder's class bank never lands in a real cache directory."""
    monkeypatch.delenv("NAM_GAIUS_CACHE_DIR", raising=False)


@pytest.fixture
def pinned_dir(tmp_path):
    from nam_heads.impls.ontology.gaius_ref.head import MODEL_PIN
    return write_export(tmp_path / "colbert-zero", MODEL_PIN["source"])


@pytest.fixture
def substitute_dir(tmp_path):
    return write_export(tmp_path / "modern", {
        "model.safetensors": "a4217337716a3f8f102082ba84fbe29396b20001486738938639183e0d9e39be",
        "1_Dense/model.safetensors": "d5d1b9bdd11cb7c4d7efa6a48edef52f98cbe79f0368df82b1b49905d522daf2",
        "tokenizer.json": "898c16d22cc76424df313a088e75ffcbcfd567e87f96adf75a644eb10cb6ae19"})


@pytest.fixture
def fake_head(monkeypatch, pinned_dir):
    """Factory: a GaiusRefOntologyEncoder on the fake encoder (fresh shared caches)."""
    from nam_heads.impls.ontology.gaius_ref import head as H

    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "_late_interaction_encoder", lambda d, t: FakeEncoder(d, t))
    monkeypatch.delenv("NAM_GAIUS_CACHE_DIR", raising=False)   # never write a fake bank to a real cache

    def make(**config):
        return H.GaiusRefOntologyEncoder(head_type="ontology",
                                         config={"model_dir": str(pinned_dir), **config})
    return make


@pytest.fixture(scope="session")
def real_head():
    from nam_heads.impls.ontology.gaius_ref import head as H

    problem = H.check_available()
    if problem:
        if os.environ.get("NAM_GAIUS_REF_REQUIRED") == "1":
            pytest.fail(f"NAM_GAIUS_REF_REQUIRED=1 but gaius_ref cannot start: {problem}")
        pytest.skip(f"gaius_ref unavailable: {problem}")
    return H.GaiusRefOntologyEncoder(head_type="ontology", config={"max_threads": 1})
