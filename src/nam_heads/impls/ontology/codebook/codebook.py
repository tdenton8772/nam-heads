"""CCO ontology codebook: load, score, resolve.

The codebook is a set of centroids in the sentence feature space
(``features.py``), each labelled with a BFO/CCO class from the certified
registry. A sentence scores each class by its best centroid's cosine
similarity; a temperature softmax turns scores into probabilities (temperature
fitted on held-out data). ``resolve`` settles on one class:

1. the top class, if its probability clears the acceptance threshold, or if
   it beats the runner-up by the calibrated margin ratio;
2. else the deepest ancestor whose summed probability (over the trained
   classes it subsumes) clears the threshold -- "a Geospatial Region, not sure
   which kind";
3. else ``bfo:entity`` (BFO:0000001): nothing beyond existence is asserted.

The artifact is validated strictly on load and the head refuses to start on any
mismatch of feature spec, TBox, class set or shapes. There is no fallback to
built-in defaults. A parser fingerprint that differs from the one the codebook
was trained on is NOT a refusal: an NLP change rarely moves the cues far enough
to matter (owner, 2026-10-22: "we can alert. not sure we should disable"), so the
codebook loads and records ``parser_mismatch`` for the host to alert on.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from nam_heads.ontology.registry import get_registry

from . import features

UNCLASSIFIED = "BFO:0000001"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_STEM = "cco_codebook"
SCHEMA = "nam-ontology-cco-codebook/1"


logger = logging.getLogger(__name__)


class CodebookError(RuntimeError):
    """The codebook artifact is missing, malformed or does not match this build."""


PROBE_PATH = ARTIFACT_DIR / "parser_probe.json"


@lru_cache(maxsize=1)
def parser_fingerprint() -> str:
    """sha256 of the cues the head extracts from a fixed probe set.

    The codebook was trained on the production NLP backend's output; a parser
    that tokenises, tags, parses or labels differently yields different cues
    and must not run against it. Hashing the cues on 300 fixed production
    texts (``artifacts/parser_probe.json``) pins that behaviour, while a
    refactor that leaves the output unchanged (a faster gazetteer pass) keeps
    the fingerprint.
    """
    from nam_heads.sdk.host import parser as host_parser

    nlp = host_parser()
    h = hashlib.sha256(features.FEATURE_SPEC.encode())
    for text in json.loads(PROBE_PATH.read_text())["texts"]:
        doc = nlp.parse(text)
        for sp in doc.sents:
            h.update(repr(features.span_cues(list(sp), sp.ents)).encode())
            h.update(repr(features.is_pointer(list(sp))).encode())
            h.update(b"|")
        h.update(b"#")
    return h.hexdigest()


def tbox_fingerprint() -> str:
    reg = get_registry()
    return hashlib.sha256(json.dumps(reg.sources, sort_keys=True).encode()).hexdigest()


@dataclass
class Prediction:
    cls: str                       # resolved class CURIE
    confidence: float              # probability mass behind cls
    top: List[Tuple[str, float]]   # trained classes by probability
    backed_off: bool = False       # cls is an ancestor of the top class
    abstained: bool = False        # cls is bfo:entity for lack of evidence
    # the calibrated softmax over the K trained classes that was resolved
    # (None: no cues); read by diagnostics, never by the classification
    probs: Optional[np.ndarray] = field(default=None, repr=False, compare=False)


class Codebook:
    def __init__(self, centroids: np.ndarray, centroid_class: np.ndarray,
                 classes: List[str], temperature: float, accept: float, meta: dict,
                 idf: Optional[np.ndarray] = None):
        self.centroids = centroids.astype(np.float32)          # (K, DIM), unit rows
        # per-bucket inverse document frequency from training (1.0 = unweighted)
        self.idf = (np.ones(features.DIM, np.float32) if idf is None
                    else idf.astype(np.float32))
        self.centroid_class = centroid_class.astype(np.int32)  # (K,) -> index into classes
        self.classes = list(classes)
        self.temperature = float(temperature)
        self.accept = float(accept)
        self.meta = meta
        # top/runner-up probability ratio at which the top class is accepted
        # below the threshold (0 = rule off); calibrated on validation
        self.margin = float(meta.get("margin", 0.0) or 0.0)
        reg = get_registry()
        # trained classes subsumed by each ancestor (for back-off)
        self._under: Dict[str, List[int]] = {}
        for ci, c in enumerate(self.classes):
            for a in reg.ancestors(c, inclusive=True):
                self._under.setdefault(a, []).append(ci)

    # ---- artifact ---------------------------------------------------------
    @classmethod
    def load(cls, directory: Path = ARTIFACT_DIR, stem: str = ARTIFACT_STEM,
             check_parser: bool = True) -> "Codebook":
        meta_path = Path(directory) / f"{stem}.json"
        arr_path = Path(directory) / f"{stem}.npz"
        if not meta_path.is_file() or not arr_path.is_file():
            raise CodebookError(f"codebook artifact missing: {meta_path} / {arr_path}")
        meta = json.loads(meta_path.read_text())
        problems = []
        if meta.get("schema") != SCHEMA:
            problems.append(f"schema {meta.get('schema')!r} != {SCHEMA!r}")
        if meta.get("feature_spec") != features.FEATURE_SPEC:
            problems.append("feature spec differs from features.py")
        if meta.get("tbox") != tbox_fingerprint():
            problems.append("TBox (BFO/CCO sources) differs from the registry")
        parser_mismatch = None
        if check_parser:
            running = parser_fingerprint()
            if meta.get("parser") != running:
                # Loads anyway: an alert, not a refusal (module docstring).
                parser_mismatch = {"trained": meta.get("parser"), "running": running}
                logger.warning("codebook.parser_mismatch trained=%s running=%s -- the NLP parser's output on the "
                               "probe set differs from the one this codebook was trained on",
                               str(meta.get("parser"))[:12], running[:12])
        reg = get_registry()
        classes = meta.get("classes") or []
        bad = [c for c in classes if not reg.is_class(c)]
        if not classes or bad:
            problems.append(f"classes missing or not in the registry: {bad[:5]}")
        with np.load(arr_path) as z:
            centroids = z["centroids"]
            centroid_class = z["centroid_class"]
            idf = z["idf"] if "idf" in z.files else None
        if idf is None or idf.shape != (features.DIM,) or not np.all(idf > 0):
            problems.append("idf weights missing or malformed")
        if centroids.ndim != 2 or centroids.shape[1] != features.DIM:
            problems.append(f"centroids shape {centroids.shape} != (K, {features.DIM})")
        if centroid_class.shape != (centroids.shape[0],) or \
                (len(centroid_class) and int(centroid_class.max()) >= len(classes)):
            problems.append("centroid_class does not index the class list")
        if sorted(set(int(i) for i in centroid_class)) != list(range(len(classes))):
            problems.append("some class has no centroid")
        for k in ("temperature", "accept"):
            if not isinstance(meta.get(k), (int, float)) or meta[k] <= 0:
                problems.append(f"{k} missing or not positive")
        if problems:
            raise CodebookError("codebook artifact rejected: " + "; ".join(problems))
        book = cls(centroids, centroid_class, classes, meta["temperature"], meta["accept"], meta,
                   idf=idf)
        book.parser_mismatch = parser_mismatch
        return book

    def save(self, directory: Path = ARTIFACT_DIR, stem: str = ARTIFACT_STEM) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / f"{stem}.npz",
                            centroids=self.centroids.astype(np.float16).astype(np.float32),
                            centroid_class=self.centroid_class, idf=self.idf)
        meta = dict(self.meta, schema=SCHEMA, classes=self.classes,
                    temperature=self.temperature, accept=self.accept, margin=self.margin,
                    feature_spec=features.FEATURE_SPEC, tbox=tbox_fingerprint())
        (directory / f"{stem}.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")

    # ---- scoring ----------------------------------------------------------
    def weight(self, vec: np.ndarray) -> np.ndarray:
        """Apply the IDF weights to a features.vectorize() vector and renormalise."""
        v = vec * self.idf
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def class_scores(self, vec: np.ndarray) -> np.ndarray:
        """Best-centroid cosine per class (vec already weighted, unit-norm or zero)."""
        nz = np.flatnonzero(vec)
        sims = self.centroids[:, nz] @ vec[nz] if len(nz) else np.zeros(len(self.centroids))
        out = np.full(len(self.classes), -1.0, dtype=np.float32)
        np.maximum.at(out, self.centroid_class, sims.astype(np.float32))
        return out

    def probabilities(self, vec: np.ndarray) -> Optional[np.ndarray]:
        """Class probabilities, or None if the sentence has no cues."""
        if not np.any(vec):
            return None
        s = self.class_scores(self.weight(vec)) / self.temperature
        s -= s.max()
        e = np.exp(s)
        return e / e.sum()

    def probabilities_sparse(self, idx: np.ndarray, vals: np.ndarray) -> Optional[np.ndarray]:
        """``probabilities`` for a ``features.sparse`` vector (identical result)."""
        if len(idx) == 0:
            return None
        w = vals * self.idf[idx]
        scratch = np.zeros(features.DIM, dtype=np.float32)
        scratch[idx] = w
        n = float(np.linalg.norm(scratch))
        if n > 0:
            w = scratch[idx] / n
        keep = w != 0
        idx, w = idx[keep], w[keep]
        sims = self.centroids[:, idx] @ w if len(idx) else np.zeros(len(self.centroids))
        scores = np.full(len(self.classes), -1.0, dtype=np.float32)
        np.maximum.at(scores, self.centroid_class, sims.astype(np.float32))
        s = scores / self.temperature
        s -= s.max()
        e = np.exp(s)
        return e / e.sum()

    def resolve(self, probs: Optional[np.ndarray]) -> Prediction:
        if probs is None:
            return Prediction(UNCLASSIFIED, 0.0, [], abstained=True)
        order = np.argsort(-probs)
        top = [(self.classes[i], float(probs[i])) for i in order[:5]]
        best_cls, best_p = top[0]
        runner_up = top[1][1] if len(top) > 1 else 0.0
        if best_p >= self.accept or (self.margin and best_p >= self.margin * max(runner_up, 1e-9)):
            return Prediction(best_cls, best_p, top, probs=probs)
        reg = get_registry()
        # deepest ancestor of the top class whose subsumed mass clears the bar
        for anc in sorted(reg.ancestors(best_cls), key=lambda a: -reg.depth(a)):
            if anc == UNCLASSIFIED:
                continue
            mass = float(probs[self._under.get(anc, [])].sum())
            if mass >= self.accept:
                return Prediction(anc, mass, top, backed_off=True, probs=probs)
        return Prediction(UNCLASSIFIED, best_p, top, abstained=True, probs=probs)

    def predict_span(self, span) -> Prediction:
        return self.resolve(self.probabilities_sparse(*features.sparse_span(span)))

    def predict_spans(self, spans: Sequence) -> Tuple[Prediction, List[Prediction]]:
        """Record-level prediction from its sentences, plus per-sentence ones.

        The record's class is its first real sentence's (the first one with
        cues -- in an encyclopedic record, the sentence that says what the
        subject is). Averaging every sentence instead dilutes it with
        sentences about other things: on held-out full records the mean is
        exact 45.6% against 72.8% for the first sentence (offline 2026-10-04).
        With no sentence carrying cues the record abstains.
        """
        per, first = [], None
        for sp in spans:
            p = self.probabilities_sparse(*features.sparse_span(sp))
            per.append(self.resolve(p))
            if first is None and p is not None:
                first = p
        return self.resolve(first), per
