"""Class bank, MaxSim scoring, calibration and resolution for the gaius head.

- ``ClassBank``: every retrievable class's verbalization embedded once as a
  ColBERT document (token vectors), concatenated, with per-class offsets. The
  class side is fixed by (model, verbalizations), so it is built once per
  process and, when a cache directory is given, stored on disk under its
  digest.
- ``ClassBank.scores``: gaius's MaxSim -- for each class, each query token's
  best cosine among the class's tokens, averaged over the query tokens.
- ``probabilities``: temperature softmax over the retrievable classes; the
  temperature is fitted on labelled validation records (``tools/calibrate``).
- ``resolve``: the top class if its probability clears ``accept``; else the
  deepest ancestor whose subsumed probability clears it; else bfo:entity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nam_heads.ontology.registry import get_registry

logger = logging.getLogger(__name__)

UNCLASSIFIED = "BFO:0000001"


class ClassBank:
    def __init__(self, classes: Sequence[str], vecs: np.ndarray, offsets: np.ndarray, digest: str):
        self.classes = list(classes)
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)      # (T, 128)
        self.offsets = np.asarray(offsets, dtype=np.int64)              # (C + 1,)
        self.digest = digest
        if len(self.offsets) != len(self.classes) + 1 or self.offsets[-1] != len(self.vecs):
            raise ValueError("class bank offsets do not index its vectors")
        if np.any(np.diff(self.offsets) <= 0):
            raise ValueError("a class has no token vectors")

    @staticmethod
    def digest_of(texts: Dict[str, str], model_id: str) -> str:
        doc = {"texts": texts, "model": model_id}
        return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def build(cls, encoder, texts: Dict[str, str], classes: Sequence[str], model_id: str,
              cache_dir: Optional[str] = None) -> "ClassBank":
        digest = cls.digest_of({c: texts[c] for c in classes}, model_id)
        path = Path(cache_dir) / f"gaius_classbank_{digest[:16]}.npz" if cache_dir else None
        if path is not None and path.is_file():
            try:
                with np.load(path, allow_pickle=False) as z:
                    if str(z["digest"]) == digest and list(z["classes"]) == list(classes):
                        return cls(classes, z["vecs"], z["offsets"], digest)
            except Exception as e:  # a bad cache file is rebuilt, never trusted
                logger.warning("gaius.classbank_cache_unreadable path=%s error=%s", path, e)
        mats = [encoder.encode_document(texts[c]) for c in classes]
        offsets = np.concatenate([[0], np.cumsum([m.shape[0] for m in mats])])
        bank = cls(classes, np.concatenate(mats), offsets, digest)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npz")
                np.savez(tmp, classes=np.array(classes), vecs=bank.vecs, offsets=bank.offsets,
                         digest=np.array(digest))
                os.replace(tmp, path)
            except OSError as e:
                logger.warning("gaius.classbank_cache_unwritable path=%s error=%s", path, e)
        return bank

    def scores(self, q: np.ndarray) -> np.ndarray:
        """(C,) MaxSim of query token vectors ``q`` (n, 128) against every class."""
        if q.shape[0] == 0:
            return np.zeros(len(self.classes), dtype=np.float32)
        sim = q @ self.vecs.T                                      # (n, T)
        best = np.maximum.reduceat(sim, self.offsets[:-1], axis=1)  # (n, C)
        return best.mean(axis=0).astype(np.float32)


def probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores.astype(np.float64) / float(temperature)
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()


@dataclass
class Prediction:
    cls: str
    confidence: float
    top: List[Tuple[str, float]]
    backed_off: bool = False
    abstained: bool = False
    probs: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class Resolver:
    """Settles a probability vector over the retrievable classes on one class."""

    def __init__(self, classes: Sequence[str], accept: float, top_k: int = 5):
        self.classes = list(classes)
        self.accept = float(accept)
        self.top_k = int(top_k)
        reg = get_registry()
        self._under: Dict[str, List[int]] = {}
        for i, c in enumerate(self.classes):
            for a in reg.ancestors(c, inclusive=True):
                self._under.setdefault(a, []).append(i)

    def abstain(self) -> Prediction:
        return Prediction(UNCLASSIFIED, 0.0, [], abstained=True)

    def resolve(self, probs: Optional[np.ndarray]) -> Prediction:
        if probs is None:
            return self.abstain()
        # ties go to the earlier class (classes are sorted by CURIE)
        order = np.lexsort((np.arange(len(probs)), -probs))
        top = [(self.classes[i], float(probs[i])) for i in order[:self.top_k]]
        best_cls, best_p = top[0]
        if best_p >= self.accept:
            return Prediction(best_cls, best_p, top, probs=probs)
        reg = get_registry()
        for anc in sorted(reg.ancestors(best_cls), key=lambda a: (-reg.depth(a), a)):
            if anc == UNCLASSIFIED:
                continue
            mass = float(probs[self._under.get(anc, [])].sum())
            if mass >= self.accept:
                return Prediction(anc, mass, top, backed_off=True, probs=probs)
        return Prediction(UNCLASSIFIED, best_p, top, abstained=True, probs=probs)


class TextScoreCache:
    """Bounded text -> scores cache (queries repeat; records rarely do)."""

    def __init__(self, size: int):
        self.size = max(0, int(size))
        self._d: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            v = self._d.pop(key, None)
            if v is not None:
                self._d[key] = v
            return v

    def put(self, key: str, value: np.ndarray) -> None:
        if not self.size:
            return
        with self._lock:
            self._d[key] = value
            while len(self._d) > self.size:
                self._d.pop(next(iter(self._d)))
