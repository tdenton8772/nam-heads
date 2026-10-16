"""Gaius-inspired ontology head: TBox verbalization + ColBERT MaxSim.

Not a reproduction of gaius: it types against BFO/CCO with a parsed noun phrase
and ModernColBERT-embed-base weights, where gaius types raw text against its SDG
ontology with ColBERT-Zero. See docs/research/gaius-fidelity-audit.md; the
faithful reproduction is impls/ontology/gaius_ref.


An implementation of the ontology slot. It answers the same question as the
codebook head (``nam_heads/impls/ontology/codebook``) -- what BFO/CCO class is this
record about -- by the method of github.com/zndx/gaius (``agents/theta/
tbox.py``, ``tbox_maxsim.py``): every class of the certified TBox is
verbalized (label + definition), embedded once with a ColBERT model, and text is
typed by MaxSim retrieval against those class texts. Nothing is trained; the
softmax temperature is the only fitted number (``tools/calibrate.py``).

Contract (identical to the codebook head's; downstream reads only this):
- ``encode`` / ``encode_query`` -> EncodingResult. ``addresses[0].concept`` is
  the record's class, a CURIE from ``emittable_classes()``; the rest are
  runner-up classes by confidence. ``BFO:0000001`` (bfo:entity) is abstain.
- ``metadata``: ``sentence_ontologies`` {sentence index: class} and
  ``sentence_confidences`` (abstaining sentences absent), ``abstained``
  (``"no_parse"`` without a parse), ``backed_off``, ``top_confidence``,
  ``mode``, ``diagnostics``.
- ``classify_nearest(text, spacy_doc=span)``: one sentence's class or None.
- It never parses: it reads ``context["spacy_doc"]`` (the rule backend's
  ParsedDocument) and abstains without one.

What it embeds (``text.py``): the lead sentence's defining phrase ("was a
German-born theoretical *physicist*" -> "a German-born theoretical
physicist"), else the title and lead; for each further real sentence (up to
``max_sentences``) its own defining phrase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from nam_heads.impls.ontology.codebook.diagnostics import DiagnosticsWindow, compute_diagnostics
from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from nam_heads.ontology.registry import get_registry

from .encoder import EncoderError, LateInteractionEncoder
from .scoring import UNCLASSIFIED, ClassBank, Prediction, Resolver, TextScoreCache, probabilities
from .text import defining_phrase, detokenize, is_real, record_text
from .verbalize import digest as texts_digest, verbalizations

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
CALIBRATION_PATH = ARTIFACT_DIR / "calibration.json"
CALIBRATION_SCHEMA = "nam-gaius-calibration/1"
VERSION = "1.0.0"
# Probe scores must reproduce within this (fp32 across CPUs differs ~1e-6).
PROBE_TOLERANCE = 2e-3


def class_set_id(classes) -> str:
    """Identity of a set of emittable classes: the same set gives the same id
    whichever head emits it."""
    return "bfo2020-cco2.2:" + hashlib.sha256("\n".join(sorted(classes)).encode()).hexdigest()[:16]


def _calibration() -> dict:
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except (OSError, ValueError):
        return {}


IMPL_META = {
    "slot": "ontology",
    "name": "gaius",
    "version": VERSION,
    "contract": "ontology-v1",
    "classes": _calibration().get("class_set_id", "unknown"),
    "description": ("Gaius-inspired zero-shot typing: BFO 2020 / CCO v2.2 class verbalizations "
                    "(label + definition) embedded with ModernColBERT-embed-base; a record's defining phrase "
                    "is typed by MaxSim retrieval. Calibrated softmax, abstain = BFO:0000001."),
    # Python packages it imports at runtime (no torch, no pylate: the model
    # runs on onnxruntime). nam-nats-worker and nam-query install all three.
    "requires": ["numpy", "onnxruntime", "tokenizers"],
    # Files it needs that the images do not ship today: the fp32 ONNX export
    # of ColBERT-Zero (~600 MB), built by tools/export_onnx.py.
    "assets": [{"name": "colbert-zero-onnx", "env": "NAM_GAIUS_MODEL_DIR",
                "default": "/app/colbert-zero-onnx",
                "files": ["model.onnx", "export.json", "tokenizer.json",
                          "config_sentence_transformers.json"]}],
}


def check_available() -> Optional[str]:
    """None when this implementation can start here; else why not (a missing
    package, model or calibration). Cheap: imports and file checks only."""
    import importlib

    for mod in IMPL_META["requires"]:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            return f"missing package {mod}: {e}"
    try:
        from .encoder import resolve_model_dir
        resolve_model_dir()
    except EncoderError as e:
        return str(e)
    if not CALIBRATION_PATH.is_file():
        return f"calibration missing: {CALIBRATION_PATH}"
    return None

DEFAULT_CONFIG = {
    "model_dir": None,            # None -> $NAM_GAIUS_MODEL_DIR, /app/colbert-zero-onnx, docker/assets
    "cache_dir": None,            # class-bank disk cache; None -> $NAM_GAIUS_CACHE_DIR or memory only
    "max_threads": None,          # ORT intra-op cap; None -> cpu_threads.threads()
    "max_sentences": 4,           # real sentences classified per record (lead included)
    "max_addresses": 5,
    "min_address_confidence": 0.05,
    "accept": None,               # None -> calibration's
    "text_cache_size": 4096,
}

# Diagnostics: the ontology slot's knobs (same env names as the codebook head)
DIAGNOSTICS_CONFIG = {
    "enable_diagnostics": ("NAM_ONTOLOGY_DIAGNOSTICS", True, bool),
    "unknown_threshold": ("NAM_ONTOLOGY_UNKNOWN_THRESHOLD", 0.25, float),
    "collapse_entropy_threshold": ("NAM_ONTOLOGY_COLLAPSE_ENTROPY", 0.80, float),
    "min_active_for_collapse": ("NAM_ONTOLOGY_MIN_ACTIVE", 5, int),
    "active_floor": ("NAM_ONTOLOGY_ACTIVE_FLOOR", 0.02, float),
    "diagnostics_window": ("NAM_ONTOLOGY_DIAG_WINDOW", 1000, int),
    "diagnostics_window_seconds": ("NAM_ONTOLOGY_DIAG_WINDOW_SECONDS", 300.0, float),
    "unclassified_rate_amber": ("NAM_ONTOLOGY_RATE_UNCLASSIFIED_AMBER", 0.25, float),
    "unclassified_rate_red": ("NAM_ONTOLOGY_RATE_UNCLASSIFIED_RED", 0.50, float),
    "overactivated_rate_amber": ("NAM_ONTOLOGY_RATE_OVERACTIVATED_AMBER", 0.25, float),
    "overactivated_rate_red": ("NAM_ONTOLOGY_RATE_OVERACTIVATED_RED", 0.50, float),
}


class GaiusError(RuntimeError):
    """The head cannot start: artifacts missing or not the ones it was calibrated on."""


def _setting(cfg: Dict, key: str):
    env, default, typ = DIAGNOSTICS_CONFIG[key]
    if key in cfg and cfg[key] is not None:
        val = cfg[key]
    else:
        val = os.environ.get(env)
        if val is None or val == "":
            return default
    if typ is bool:
        return val if isinstance(val, bool) else str(val).strip().lower() not in ("0", "false", "no", "off")
    return typ(val)


# One encoder and one class bank per process, shared by every instance.
_SHARED: Dict[tuple, object] = {}
_SHARED_LOCK = threading.Lock()


def _shared(key: tuple, build):
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = build()
        return _SHARED[key]


def load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    if not Path(path).is_file():
        raise GaiusError(f"calibration missing: {path} (run tools/calibrate.py)")
    cal = json.loads(Path(path).read_text())
    problems = []
    if cal.get("schema") != CALIBRATION_SCHEMA:
        problems.append(f"schema {cal.get('schema')!r} != {CALIBRATION_SCHEMA!r}")
    reg = get_registry()
    if cal.get("tbox") != reg.sources:
        problems.append("calibrated against a different TBox than the registry's")
    cands = cal.get("candidates") or []
    bad = [c for c in cands if not reg.is_class(c) or c == UNCLASSIFIED]
    if not cands or bad:
        problems.append(f"candidate classes missing or invalid: {bad[:5]}")
    for k in ("temperature",):
        if not isinstance(cal.get(k), (int, float)) or cal[k] <= 0:
            problems.append(f"{k} missing or not positive")
    if not isinstance(cal.get("accept"), (int, float)) or not 0 <= cal["accept"] <= 1:
        problems.append("accept missing or outside [0, 1]")
    if problems:
        raise GaiusError("calibration rejected: " + "; ".join(problems))
    return cal


class GaiusOntologyEncoder(EncoderHead):
    """Types text with a BFO/CCO class by MaxSim against TBox verbalizations."""

    HEAD_TYPE = "ontology"
    IMPL = "gaius"

    # ---- start-up ---------------------------------------------------------
    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **(self.config or {})}
        self.max_sentences = max(1, int(cfg["max_sentences"]))
        self.max_addresses = int(cfg["max_addresses"])
        self.min_address_confidence = float(cfg["min_address_confidence"])
        cal = load_calibration()
        self.calibration = cal
        self.candidates: List[str] = list(cal["candidates"])
        self.style = cal["style"]
        self.temperature = float(cal["temperature"])
        self.accept = float(cfg["accept"] if cfg["accept"] is not None else cal["accept"])
        self.resolver = Resolver(self.candidates, self.accept)

        texts = verbalizations(self.candidates, self.style)   # checks the TBox pin
        model_dir = cfg["model_dir"]
        max_threads = cfg["max_threads"]
        try:
            self.encoder = _shared(("encoder", model_dir, max_threads),
                                   lambda: LateInteractionEncoder(model_dir, max_threads=max_threads))
        except EncoderError as e:
            raise GaiusError(str(e)) from e
        self._check_model(cal)
        model_id = f"{self.encoder.meta.get('model')}:{self.encoder.meta.get('onnx_sha256')}"
        cache_dir = cfg["cache_dir"] or os.environ.get("NAM_GAIUS_CACHE_DIR") or None
        self.bank: ClassBank = _shared(
            ("bank", str(self.encoder.model_dir), ClassBank.digest_of(texts, model_id)),
            lambda: ClassBank.build(self.encoder, texts, self.candidates, model_id, cache_dir))
        if cal.get("verbalizations_sha256") != texts_digest(texts):
            raise GaiusError("class verbalizations differ from the calibrated ones")
        self._cache = TextScoreCache(cfg["text_cache_size"])
        self.class_set_id = class_set_id(self.emittable_classes())

        user = self.config or {}
        d = {k: _setting(user, k) for k in DIAGNOSTICS_CONFIG}
        self.enable_diagnostics = d["enable_diagnostics"]
        self.unknown_threshold = d["unknown_threshold"]
        self.collapse_entropy_threshold = d["collapse_entropy_threshold"]
        self.min_active_for_collapse = d["min_active_for_collapse"]
        self.active_floor = d["active_floor"]
        self.diagnostics = None
        if self.enable_diagnostics:
            self.diagnostics = DiagnosticsWindow(
                d["diagnostics_window"], len(self.emittable_classes()),
                {"unknown": self.unknown_threshold,
                 "collapse_entropy": self.collapse_entropy_threshold,
                 "min_active": self.min_active_for_collapse,
                 "active_floor": self.active_floor,
                 "rates": {
                     "unclassified": {"amber": d["unclassified_rate_amber"],
                                      "red": d["unclassified_rate_red"]},
                     "overactivated": {"amber": d["overactivated_rate_amber"],
                                       "red": d["overactivated_rate_red"]},
                 }},
                seconds=d["diagnostics_window_seconds"])
        logger.info("GaiusOntologyEncoder initialized: candidates=%d style=%s T=%.5f accept=%.3f "
                    "model=%s class_set=%s", len(self.candidates), self.style, self.temperature,
                    self.accept, self.encoder.model_dir, self.class_set_id)

    def _check_model(self, cal: dict) -> None:
        """The model on disk must be the one calibrated: same source weights,
        and the probe texts must score as they did."""
        want = (cal.get("model") or {}).get("source") or {}
        have = self.encoder.meta.get("source") or {}
        if want and want != have:
            raise GaiusError("ColBERT-Zero weights differ from the calibrated model")
        for p in cal.get("probes") or []:
            q = self.encoder.encode_query(p["text"])
            d = self.encoder.encode_document(p["class_text"])
            got = float((q @ d.T).max(axis=1).mean())
            if abs(got - p["score"]) > PROBE_TOLERANCE:
                raise GaiusError(f"model behaviour differs from calibration on probe {p['text']!r}: "
                                 f"{got:.5f} vs {p['score']:.5f}")

    # ---- reporting --------------------------------------------------------
    def config_summary(self) -> Dict:
        return {
            "impl": self.IMPL,
            "version": VERSION,
            "class_set": self.class_set_id,
            "candidates": len(self.candidates),
            "style": self.style,
            "temperature": self.temperature,
            "accept": self.accept,
            "max_sentences": self.max_sentences,
            "model": self.encoder.meta.get("model"),
            "diagnostics": self.enable_diagnostics,
        }

    def get_metrics(self) -> Dict:
        out = super().get_metrics()
        if self.diagnostics is None:
            return out
        snap = self.diagnostics.snapshot()
        w = snap["window"]
        out["diagnostics"] = snap
        out.update({
            "ontology_entropy_mean": w["entropy_mean"],
            "ontology_unclassified_rate": w["unclassified_rate"],
            "ontology_overactivated_rate": w["overactivated_rate"],
            "ontology_class_entropy": w["class_entropy"],
            "ontology_top1_share": w["top1_share"],
        })
        return out

    def emittable_classes(self) -> List[str]:
        """Every class this head can put in a key: the retrievable classes,
        the ancestors it can back off to, and bfo:entity."""
        reg = get_registry()
        out = {UNCLASSIFIED}
        for c in self.candidates:
            out |= reg.ancestors(c, inclusive=True)
        return sorted(out)

    # ---- classification ---------------------------------------------------
    def _scores(self, text: str) -> np.ndarray:
        hit = self._cache.get(text)
        if hit is None:
            hit = self.bank.scores(self.encoder.encode_query(text))
            self._cache.put(text, hit)
        return hit

    def _predict(self, text: Optional[str]) -> Prediction:
        if not text:
            return self.resolver.abstain()
        return self.resolver.resolve(probabilities(self._scores(text), self.temperature))

    def classify_nearest(self, text: str, spacy_doc=None) -> Optional[str]:
        """Class of one sentence span, None when it abstains (does not vote)."""
        if spacy_doc is None:
            return None
        toks = list(spacy_doc)
        if not is_real(toks):
            return None
        ph = defining_phrase(toks)
        pred = self._predict(ph.text if ph else detokenize([t.text for t in toks]))
        return None if pred.abstained else pred.cls

    def _addresses(self, pred: Prediction) -> List[SemanticAddress]:
        out = [SemanticAddress(
            head_type=self.head_type, concept=pred.cls, confidence=float(pred.confidence),
            metadata={"method": "gaius_maxsim", "backed_off": pred.backed_off,
                      "abstained": pred.abstained})]
        for cls, p in pred.top:
            if len(out) >= self.max_addresses:
                break
            if cls != pred.cls and p >= self.min_address_confidence:
                out.append(SemanticAddress(
                    head_type=self.head_type, concept=cls, confidence=float(p),
                    metadata={"method": "gaius_maxsim", "runner_up": True}))
        return out

    def _diagnose(self, pred: Prediction, meta: Dict) -> None:
        if self.diagnostics is None:
            return
        diag = compute_diagnostics(
            pred.probs, abstained=pred.abstained,
            unknown_threshold=self.unknown_threshold,
            collapse_entropy_threshold=self.collapse_entropy_threshold,
            min_active_for_collapse=self.min_active_for_collapse,
            active_floor=self.active_floor)
        meta["diagnostics"] = diag
        self.diagnostics.record(diag, pred.cls)

    def _encode(self, context: Optional[Dict], mode: str) -> EncodingResult:
        doc = (context or {}).get("spacy_doc")
        meta: Dict = {"mode": mode}
        if doc is None or not hasattr(doc, "sents"):
            meta["abstained"] = "no_parse"
            pred = self.resolver.abstain()
            self._diagnose(pred, meta)
            return EncodingResult(head_type=self.head_type, addresses=self._addresses(pred),
                                  metadata=meta)
        rt = record_text(doc)
        lead = rt.lead
        query = rt.query_text()
        record = self._predict(query)
        sent_classes, sent_conf = {}, {}
        for i, _text in rt.sentences[:self.max_sentences]:
            if lead and i == lead[0]:
                p = record
            else:
                ph = rt.phrases.get(i)
                if ph is None:
                    continue
                p = self._predict(ph.text)
            if not p.abstained:
                sent_classes[i] = p.cls
                sent_conf[i] = round(float(p.confidence), 4)
        meta.update({
            "sentence_ontologies": sent_classes,
            "sentence_confidences": sent_conf,
            "backed_off": record.backed_off,
            "abstained": record.abstained,
            "top_confidence": float(record.confidence),
            "query_text": query,
        })
        self._diagnose(record, meta)
        return EncodingResult(head_type=self.head_type, addresses=self._addresses(record),
                              metadata=meta)

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode(context, "ingest")

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode(context, "query")
