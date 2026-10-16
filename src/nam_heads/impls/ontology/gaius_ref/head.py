"""gaius_ref: the ontology slot, typed exactly as gaius types text.

gaius (github.com/zndx/gaius @ 06cbf95d) types a text with
``agents/theta/tbox_maxsim.py`` ``maxsim_tbox_topk``:

- the class side: the 905 classes gaius's own loader reads from the certified
  SDG TBox, as ``label. definition`` (``tbox.py``; here
  ``artifacts/tbox_classes.json``, see ``tbox.py`` in this package), encoded
  with ColBERT-Zero as documents (``prefix="search_document: "``,
  tbox_maxsim.py:116-118);
- the query side: the text as given, encoded as a query
  (``prefix="search_query: "``, tbox_maxsim.py:147);
- the score: Qdrant MAX_SIM over cosine (sum over query tokens of the best
  class token), divided by the number of query token vectors
  (tbox_maxsim.py:105-114, 154-161);
- the output: the top k=8 class IRIs with those scores (tbox_maxsim.py:135-162),
  no threshold, no softmax, no back-off; an empty text gives nothing
  (tbox_maxsim.py:142-143).

**What gaius does with it, and what this head does.**

- gaius types only ``admitted_item`` rows: 512-token windows its SDG aperture
  admitted (clt_skos_admit.py:95-112; clt_incidence.py:92,153). It never types
  arbitrary text. This head types whatever text the ontology stage gives it,
  so its coverage is a property of the nam pipeline, not of gaius.
- gaius uses the top 8 as a grounding set: every hit, weighted by its score,
  pairs with other items' hits on a shared CLT feature to propose novel
  subsumptions (clt_incidence.py:26-56). It has no record class.

The slot contract (``ontology-v1``) wants one class per record, so this head
reports gaius's rank-1 hit as the record class: ``addresses`` holds that one
class, with gaius's own score as the confidence. The whole grounding set,
gaius's top 8 in gaius's order with gaius's scores, is
``metadata["top_k"]`` (``[[curie, iri, score], ...]``), which the head-result
serialization keeps (``serialization/converters.py``). Nothing is
thresholded, calibrated or backed off. An empty text, where gaius returns
nothing, is ``BFO:0000001``.

Classes are the SDG ontology's, not BFO/CCO's. IMPL_META "ontology" brings
them (``artifacts/sdg_ontology.json``): while the manifest selects gaius_ref,
every manifest reader adds them to the ontology registry
(``nam_heads/ontology/registry.py`` ``use_extension``), so the bundler keys a record
by its SDG class. A class keeps the BFO/CCO CURIE when BFO/CCO has its IRI,
else it is ``sdg:<name>`` (``xsd:<name>``); no class is mapped to another.
gaius types items, not sentences, so ``sentence_ontologies`` is empty.

The encoder and the class-bank MaxSim are the gaius head's
(``impls/ontology/gaius/encoder.py``, ``scoring.ClassBank``), run on the
``lightonai/ColBERT-Zero`` weights gaius loads (MODEL_PIN). Against gaius's own
``tbox_maxsim.py`` on gaius's stack, the top 8 matches
(``artifacts/gaius_native_compare.json``). Any other export is refused unless
``require_colbert_zero`` is False, which the result metadata reports.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress

from . import tbox as T

logger = logging.getLogger(__name__)

VERSION = "1.0.0"
UNCLASSIFIED = "BFO:0000001"
ENV_MODEL_DIR = "NAM_GAIUS_REF_MODEL_DIR"
DEFAULT_MODEL_DIR = "/app/colbert-zero-ref-onnx"
TOP_K = 8                                   # maxsim_tbox_topk(k=8), tbox_maxsim.py:90

# lightonai/ColBERT-Zero @ 43aefda07cf4 (HF main, 2026-08-31): sha256 of the
# source files the ONNX export was made from (export.json "source"). The
# tokenizer is byte-identical to ModernColBERT-embed-base's.
MODEL_PIN = {
    "repo": "lightonai/ColBERT-Zero",
    "revision": "43aefda07cf4fea7a13dbdab0eef8ecf78580d3d",
    "source": {
        "model.safetensors": "57781fbe9089304684b85f19ac8441f19bee15ef1b36d0d2dd9bb744025cb367",
        "1_Dense/model.safetensors": "4402f4e674ce98f214b4af2745f52be913bd3d8afc02529159d3d78e9cfc28a7",
        "tokenizer.json": "898c16d22cc76424df313a088e75ffcbcfd567e87f96adf75a644eb10cb6ae19",
    },
}
# Other ColBERT checkpoints an export may hold, named in the refusal.
KNOWN_WEIGHTS = {
    "a4217337716a3f8f102082ba84fbe29396b20001486738938639183e0d9e39be":
        "lightonai/ModernColBERT-embed-base @ 0201e5750705",
}

IMPL_META = {
    "slot": "ontology",
    "name": "gaius_ref",
    "version": VERSION,
    "contract": "ontology-v1",
    "description": ("gaius's TBox MaxSim, reproduced: the 905 classes gaius's own loader reads "
                    "from the certified SDG ontology (gaius 06cbf95d, sdg-corpora c4138b4d), the "
                    "record text typed by ColBERT-Zero MaxSim / query tokens. Record class = "
                    "gaius's rank-1 hit; its top 8 in metadata top_k. No threshold."),
    "requires": ["numpy", "onnxruntime", "tokenizers"],
    "assets": [{"name": "colbert-zero-ref-onnx", "env": ENV_MODEL_DIR, "default": DEFAULT_MODEL_DIR,
                "files": ["model.onnx", "export.json", "tokenizer.json",
                          "config_sentence_transformers.json"]}],
    # The implementation's own classes: added to the ontology registry while
    # the manifest selects it (src/encoders/slots.py apply_ontology).
    "ontology": {"name": "sdg", "artifact": "artifacts/sdg_ontology.json"},
}

DEFAULT_CONFIG = {
    "model_dir": None,               # None -> $NAM_GAIUS_REF_MODEL_DIR, else DEFAULT_MODEL_DIR
    "cache_dir": None,               # class-bank disk cache; None -> $NAM_GAIUS_CACHE_DIR or memory
    "max_threads": None,             # ORT intra-op cap; None -> cpu_threads.threads()
    "top_k": TOP_K,
    "require_colbert_zero": True,    # False: run another export, reported in config_summary
    "text_cache_size": 4096,
}


class GaiusRefError(RuntimeError):
    """gaius_ref cannot start: the TBox artifact or the pinned model is missing."""


def model_dir_of(explicit: Optional[str] = None) -> Path:
    d = explicit or os.environ.get(ENV_MODEL_DIR) or DEFAULT_MODEL_DIR
    return Path(d)


def weights_problem(meta: Dict) -> Optional[str]:
    """None when export.json's source files are MODEL_PIN's; else why not."""
    have = meta.get("source") or {}
    if have == MODEL_PIN["source"]:
        return None
    name = KNOWN_WEIGHTS.get(have.get("model.safetensors", ""), "unknown weights")
    return (f"the export holds {name}, not {MODEL_PIN['repo']} @ {MODEL_PIN['revision'][:12]} "
            f"(model.safetensors sha256 {have.get('model.safetensors', '?')[:12]} != "
            f"{MODEL_PIN['source']['model.safetensors'][:12]})")


def check_available() -> Optional[str]:
    """None when gaius_ref can start here; else why not. Imports and file reads only."""
    import importlib

    for mod in IMPL_META["requires"]:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            return f"missing package {mod}: {e}"
    d = model_dir_of()
    for f in IMPL_META["assets"][0]["files"]:
        if not (d / f).is_file():
            return f"model file missing: {d / f} (set ${ENV_MODEL_DIR})"
    try:
        T.load()
        problem = weights_problem(json.loads((d / "export.json").read_text()))
    except (T.TBoxError, OSError, ValueError) as e:
        return str(e)
    return problem


# One encoder and one class bank per process.
_SHARED: Dict[tuple, object] = {}
_SHARED_LOCK = threading.Lock()


def _shared(key: tuple, build):
    with _SHARED_LOCK:
        if key not in _SHARED:
            _SHARED[key] = build()
        return _SHARED[key]


def _late_interaction_encoder(model_dir: str, max_threads):
    from nam_heads.impls.ontology.gaius.encoder import LateInteractionEncoder
    return LateInteractionEncoder(model_dir, max_threads=max_threads)


def _class_bank(encoder, texts: Dict[str, str], classes: List[str], model_id: str, cache_dir):
    from nam_heads.impls.ontology.gaius.scoring import ClassBank
    return ClassBank.build(encoder, texts, classes, model_id, cache_dir)


class GaiusRefOntologyEncoder(EncoderHead):
    """gaius's ``maxsim_tbox_topk`` as an ontology-slot implementation."""

    HEAD_TYPE = "ontology"
    IMPL = "gaius_ref"

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **(self.config or {})}
        self.top_k = max(1, int(cfg["top_k"]))
        try:
            doc = T.load()
            ext = T.load_extension()
            to_curie = T.iri_to_curie()
        except T.TBoxError as e:
            raise GaiusRefError(str(e)) from e
        rows = doc["classes"]
        # class order is owlready2's enumeration order (gaius's upsert order)
        self.iris: List[str] = [r["iri"] for r in rows]
        self.classes: List[str] = [to_curie[r["iri"]] for r in rows]
        if len(set(self.classes)) != len(self.classes) or set(self.classes) != set(ext["emittable"]):
            raise GaiusRefError("the TBox classes and the ontology artifact's emittable classes differ")
        self.ontology_name = ext["name"]
        # True: a BFO/CCO class (the built-in registry has its IRI)
        self.bfo_cco = np.array([c not in ext["classes"] for c in self.classes])
        texts = {c: r["text"].strip() for c, r in zip(self.classes, rows)}   # tbox_maxsim.py:67

        model_dir = model_dir_of(cfg["model_dir"])
        if not (model_dir / "model.onnx").is_file():
            raise GaiusRefError(f"ColBERT-Zero ONNX export not found in {model_dir} (set ${ENV_MODEL_DIR})")
        self.export = json.loads((model_dir / "export.json").read_text())
        problem = weights_problem(self.export)
        self.model_substitute = None
        if problem:
            if cfg["require_colbert_zero"]:
                raise GaiusRefError(problem)
            self.model_substitute = problem
            logger.warning("gaius_ref.model_substitute %s", problem)
        max_threads = cfg["max_threads"]
        self.encoder = _shared(("encoder", str(model_dir), max_threads),
                               lambda: _late_interaction_encoder(str(model_dir), max_threads))
        if Path(self.encoder.model_dir).resolve() != model_dir.resolve():
            raise GaiusRefError(f"encoder loaded {self.encoder.model_dir}, not {model_dir}")
        model_id = f"{self.export.get('onnx_sha256')}"
        cache_dir = cfg["cache_dir"] or os.environ.get("NAM_GAIUS_CACHE_DIR") or None
        self.bank = _shared(("bank", str(model_dir), T.file_sha256(T.TBOX_PATH)),
                            lambda: _class_bank(self.encoder, texts, self.classes, model_id, cache_dir))
        self._cache = _TextCache(cfg["text_cache_size"])
        from nam_heads.ontology.registry import active_extension
        if active_extension() != self.ontology_name:
            logger.warning("gaius_ref.ontology_not_active registry_extension=%s -- this process keys "
                           "%s classes bfo:entity until the manifest selecting gaius_ref is applied",
                           active_extension(), self.ontology_name)
        logger.info("GaiusRefOntologyEncoder initialized: classes=%d (BFO/CCO %d, %s %d) top_k=%d model=%s%s",
                    len(self.classes), int(self.bfo_cco.sum()), self.ontology_name,
                    int((~self.bfo_cco).sum()), self.top_k, model_dir,
                    " SUBSTITUTE" if self.model_substitute else "")

    # ---- reporting --------------------------------------------------------
    def config_summary(self) -> Dict:
        return {
            "impl": self.IMPL,
            "version": VERSION,
            "classes": len(self.classes),
            "ontology": self.ontology_name,
            "classes_bfo_cco": int(self.bfo_cco.sum()),
            "top_k": self.top_k,
            "tbox": T.load()["source"],
            "model_source": self.export.get("source"),
            "model_substitute": self.model_substitute,
        }

    def emittable_classes(self) -> List[str]:
        """Every class gaius can return (bfo:entity is one of them)."""
        return sorted(set(self.classes) | {UNCLASSIFIED})

    # ---- typing -----------------------------------------------------------
    def scores(self, text: str) -> np.ndarray:
        """Per class: summed best cosine over query tokens / query tokens."""
        hit = self._cache.get(text)
        if hit is None:
            hit = self.bank.scores(self.encoder.encode_query(text))
            self._cache.put(text, hit)
        return hit

    def topk(self, text: Optional[str]) -> List[Tuple[int, float]]:
        """gaius's ``maxsim_tbox_topk``: [(class index, score)], best first."""
        if not (text or "").strip() or self.top_k < 1:                 # tbox_maxsim.py:94-95
            return []
        s = self.scores(text)
        order = np.lexsort((np.arange(len(s)), -s))[:self.top_k]
        return [(int(i), float(s[i])) for i in order]

    def classify_nearest(self, text: str, spacy_doc=None) -> Optional[str]:
        """gaius's top class for ``text`` (the span's own text), None when empty."""
        top = self.topk(text)
        return self.classes[top[0][0]] if top else None

    def _result(self, text: Optional[str], mode: str) -> EncodingResult:
        top = self.topk(text)
        meta: Dict = {"mode": mode, "sentence_ontologies": {}, "sentence_confidences": {},
                      "backed_off": False}
        if not top:
            meta.update({"abstained": "no_text", "top_confidence": 0.0, "top_k": []})
            return EncodingResult(head_type=self.head_type, metadata=meta, addresses=[SemanticAddress(
                head_type=self.head_type, concept=UNCLASSIFIED, confidence=0.0,
                metadata={"method": "gaius_maxsim_tbox", "abstained": True})])
        best, best_score = top[0]
        addresses = [SemanticAddress(
            head_type=self.head_type, concept=self.classes[best], confidence=best_score,
            metadata={"method": "gaius_maxsim_tbox", "iri": self.iris[best], "rank": 0,
                      "bfo_cco": bool(self.bfo_cco[best])})]
        meta.update({"abstained": False, "top_confidence": best_score, "top_iri": self.iris[best],
                     "outside_bfo_cco": not bool(self.bfo_cco[best]),
                     # gaius's grounding set (maxsim_tbox_topk): top k in its order
                     "top_k": [[self.classes[i], self.iris[i], score] for i, score in top]})
        if self.model_substitute:
            meta["model_substitute"] = True
        return EncodingResult(head_type=self.head_type, addresses=addresses, metadata=meta)

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._result(text, "ingest")

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._result(query, "query")


class _TextCache:
    """Bounded text -> scores cache."""

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
