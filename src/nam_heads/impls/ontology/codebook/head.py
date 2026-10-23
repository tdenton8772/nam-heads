"""Ontology head: BFO 2020 / CCO v2.2 classes from a trained codebook.

Answers "what kind of thing is this text about?" with a class from the
certified BFO/CCO TBox (``nam_heads/ontology``). At ingest that class becomes the
address key's type segment, so every address of a record carries the class of
what the record is about: all of Einstein's addresses -- including the one
anchored on "Nobel Prize" -- are cco:Person.

How it classifies (design and the lessons behind it:
docs/plans/ontology-bfo-cco.md):
- Each sentence of the shared ParsedDocument becomes a length-invariant cue
  vector (``features.py``) scored against the codebook (``codebook.py``): IDF
  weighting, discriminatively trained centroids, a calibrated softmax.
- An uncertain sentence or record backs off to the deepest ancestor class that
  carries the probability, then to bfo:entity (BFO:0000001).
- The record's class is the vote of its sentences; the per-sentence classes
  are published as ``sentence_ontologies`` / ``sentence_confidences`` for the
  bundler.

Queries are classified the same way, as a soft preference only: which
partitions a query entity is actually stored under comes from the index, not
from the wording of the question.

It never parses: it reads the shared parse from ``context["spacy_doc"]``, and
without one it abstains (bfo:entity) with ``abstained: "no_parse"`` in its
metadata, so NAM_SKIP_HEAD_NLP cannot silently degrade it. It refuses to start
without a valid codebook.

Diagnostics (``diagnostics.py``): every encode carries
``metadata["diagnostics"]`` -- entropy of the deciding class distribution, its
top probability, active class count and the ``ontology_unclassified`` /
``ontology_overactivated`` alerts -- and ``get_metrics()`` publishes a rolling
readout of them. They read the distribution and never change the emitted class.
Thresholds come from the head config, else ``NAM_ONTOLOGY_UNKNOWN_THRESHOLD``,
``NAM_ONTOLOGY_COLLAPSE_ENTROPY``, ``NAM_ONTOLOGY_MIN_ACTIVE``,
``NAM_ONTOLOGY_ACTIVE_FLOOR``, ``NAM_ONTOLOGY_DIAG_WINDOW``;
``NAM_ONTOLOGY_DIAGNOSTICS=0`` turns them off.
"""

import logging
import os
from typing import Dict, List, Optional

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress

from nam_heads.ontology.registry import get_registry

from .codebook import Codebook, Prediction
from .diagnostics import DiagnosticsWindow, compute_diagnostics

logger = logging.getLogger(__name__)

IMPL_META = {
    "slot": "ontology",
    "name": "codebook",
    "version": "1.0.1",
    "contract": "ontology-v1",
    "output": "bfo-cco-2.2",
    "resources": {"cpu": 2.0, "memory_mb": 2048, "workers": 1},
    "dependencies": [],
    # What host stages read from it besides the head (nam_heads.sdk.exports).
    "provides": {"merge_diagnostics": "diagnostics:merge_snapshots"},
    "description": ("BFO 2020 / CCO v2.2 class from a trained codebook over the host parse's "
                    "sentence cues; backs off to an ancestor, abstains to bfo:entity."),
    "requires": ["numpy"],
}

DEFAULT_CONFIG = {
    "codebook_dir": None,          # None -> this package's artifacts/
    # Compare the host parser's cues on the probe set with the ones the codebook
    # was trained on (codebook.parser_fingerprint); a difference is reported as
    # parser_mismatch (metric ontology_parser_mismatch), never a refusal. False
    # only where no host parser exists (tests of this package on its own).
    "check_parser": True,
    "max_addresses": 5,            # resolved class + runner-up trained classes
    "min_address_confidence": 0.05,
}

# Diagnostics: config key -> (env override, default, type). Explicit head
# config wins over the environment, the environment over the default.
DIAGNOSTICS_CONFIG = {
    "enable_diagnostics": ("NAM_ONTOLOGY_DIAGNOSTICS", True, bool),
    # top-class probability below which a record counts as unclassified
    "unknown_threshold": ("NAM_ONTOLOGY_UNKNOWN_THRESHOLD", 0.25, float),
    # normalized entropy above which (with enough active classes) a record is
    # overactivated
    "collapse_entropy_threshold": ("NAM_ONTOLOGY_COLLAPSE_ENTROPY", 0.80, float),
    "min_active_for_collapse": ("NAM_ONTOLOGY_MIN_ACTIVE", 5, int),
    # probability at which a class counts as active
    "active_floor": ("NAM_ONTOLOGY_ACTIVE_FLOOR", 0.02, float),
    # The rolling window behind get_metrics(): the last N encodes or the last
    # S seconds, whichever holds fewer (short, so the readout reacts quickly).
    "diagnostics_window": ("NAM_ONTOLOGY_DIAG_WINDOW", 1000, int),
    "diagnostics_window_seconds": ("NAM_ONTOLOGY_DIAG_WINDOW_SECONDS", 300.0, float),
    # Head status from the window's aggregate alert RATES (share of recent
    # encodes that raised each alert): amber / red at or above these.
    "unclassified_rate_amber": ("NAM_ONTOLOGY_RATE_UNCLASSIFIED_AMBER", 0.25, float),
    "unclassified_rate_red": ("NAM_ONTOLOGY_RATE_UNCLASSIFIED_RED", 0.50, float),
    "overactivated_rate_amber": ("NAM_ONTOLOGY_RATE_OVERACTIVATED_AMBER", 0.25, float),
    "overactivated_rate_red": ("NAM_ONTOLOGY_RATE_OVERACTIVATED_RED", 0.50, float),
}


def _diag_setting(cfg: Dict, key: str):
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


class OntologyEncoder(EncoderHead):
    """Types text with a BFO/CCO class."""

    HEAD_TYPE = "ontology"

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **(self.config or {})}
        self.max_addresses = int(cfg["max_addresses"])
        self.min_address_confidence = float(cfg["min_address_confidence"])
        kwargs = {"directory": cfg["codebook_dir"]} if cfg["codebook_dir"] else {}
        self.check_parser = bool(cfg["check_parser"])
        self.codebook = Codebook.load(check_parser=self.check_parser, **kwargs)
        self.codebook_version = self.codebook.meta.get("created", "unknown")
        user_cfg = self.config or {}
        d = {k: _diag_setting(user_cfg, k) for k in DIAGNOSTICS_CONFIG}
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
        logger.info("OntologyEncoder(cco) initialized: classes=%d centroids=%d codebook=%s",
                    len(self.codebook.classes), len(self.codebook.centroids),
                    self.codebook_version)

    def config_summary(self) -> Dict:
        return {
            "codebook": self.codebook_version,
            "classes": len(self.codebook.classes),
            "accept": self.codebook.accept,
            "parser_checked": self.check_parser,
            "parser_mismatch": getattr(self.codebook, "parser_mismatch", None),
            "diagnostics": self.enable_diagnostics,
            "unknown_threshold": self.unknown_threshold,
            "collapse_entropy_threshold": self.collapse_entropy_threshold,
            "min_active_for_collapse": self.min_active_for_collapse,
        }

    def get_metrics(self) -> Dict:
        """Base counters plus the diagnostics readout.

        ``diagnostics`` is the nested readout (``DiagnosticsWindow.snapshot``);
        the flat ``ontology_*`` scalars repeat its headline numbers so the
        heartbeat's gauge publisher (numeric top-level fields only) gives them
        history.
        """
        out = super().get_metrics()
        # 1 while the host's parser differs from the one the codebook was trained on (an alert, not a stop)
        out["ontology_parser_mismatch"] = 1 if getattr(self.codebook, "parser_mismatch", None) else 0
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

    def emittable_classes(self) -> List[str]:
        """Every class this head can put in a key: its trained classes, the
        ancestors it can back off to, and bfo:entity. The query planner scans
        these partitions."""
        reg = get_registry()
        out = set()
        for c in self.codebook.classes:
            out |= reg.ancestors(c, inclusive=True)
        return sorted(out)

    # ---- classification ---------------------------------------------------
    def classify_nearest(self, text: str, spacy_doc=None) -> Optional[str]:
        """Class of one sentence span (the NATS stage's per-sentence call).

        Returns None when the sentence abstains, so it does not vote.
        """
        if spacy_doc is None:
            return None
        pred = self.codebook.predict_span(spacy_doc)
        return None if pred.abstained else pred.cls

    def _addresses(self, pred: Prediction) -> List[SemanticAddress]:
        out = [SemanticAddress(
            head_type=self.head_type, concept=pred.cls, confidence=float(pred.confidence),
            metadata={"method": "cco_codebook", "backed_off": pred.backed_off,
                      "abstained": pred.abstained})]
        for cls, p in pred.top:
            if len(out) >= self.max_addresses:
                break
            if cls != pred.cls and p >= self.min_address_confidence:
                out.append(SemanticAddress(
                    head_type=self.head_type, concept=cls, confidence=float(p),
                    metadata={"method": "cco_codebook", "runner_up": True}))
        return out

    def _encode(self, context: Optional[Dict], mode: str) -> EncodingResult:
        doc = (context or {}).get("spacy_doc")
        meta: Dict = {"mode": mode, "codebook": self.codebook_version}
        if doc is None or not hasattr(doc, "sents"):
            meta["abstained"] = "no_parse"
            pred = self.codebook.resolve(None)
            self._diagnose(pred, meta)
            return EncodingResult(head_type=self.head_type,
                                  addresses=self._addresses(pred),
                                  metadata=meta)
        record, per = self.codebook.predict_spans(list(doc.sents))
        sent_classes, sent_conf = {}, {}
        for i, p in enumerate(per):
            if not p.abstained:
                sent_classes[i] = p.cls
                sent_conf[i] = round(float(p.confidence), 4)
        meta.update({
            "sentence_ontologies": sent_classes,
            "sentence_confidences": sent_conf,
            "backed_off": record.backed_off,
            "abstained": record.abstained,
            "top_confidence": float(record.confidence),
        })
        self._diagnose(record, meta)
        return EncodingResult(head_type=self.head_type, addresses=self._addresses(record),
                              metadata=meta)

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode(context, "ingest")

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode(context, "query")
