"""
Affordance encoder head.

Answers: "What intentional action or interaction defines how this memory should be navigated?"
Emits one primary affordance and optionally one secondary affordance,
or __null__ if none are suitable.

When ontology context is available (phase-1 ontology result in ``ctx["ontology_result"]``),
verb scores are nudged by ontology-category relevance *before* primary/secondary selection
so that ontology can influence which verbs survive the cut, not just their ranking.

NER entity types from the parse's ``.ents`` provide an additional scoring signal
via per-category multipliers.

The parse is the record's own (``context["spacy_doc"]``, the rule backend's
ParsedDocument). A record without one is parsed with the same rule parser
(``src.encoders.record_parse``) before extraction, in both emit modes.
"""
import dataclasses
import json
import logging
import re
from typing import Dict, Optional, List, Set, Tuple

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from .config import DEFAULT_CONFIG
from .definitions import (
    get_ontology_relevance,
    get_best_category_for_ontology,
    FIREWALL_ONTOLOGY_EXEMPTIONS,
    NER_ENTITY_TYPE_RELEVANCE,
    VERB_SYNONYMS,
    ONTOLOGY_KEYS,
)
from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import get_registry
from .storage.factory import create_category_store
from .definitions import (
    FUNCTION_VERBS_DEFAULT, ONTOLOGY_FUNCTION_VERBS, ontology_class, ontology_lookup,
)
from nam_heads.sdk.learner import parked_result

# What this default plugin declares (docs/design-head-plugins.md section 3), read with ast
# (nam_heads.sdk.impl_meta.read_module_literal): literals only. "resources" are the per-container
# limits and the containers per ingest pod it runs with on the reference deployment:
# the chart's ingest.axisHeadResources.limits (cpu "1", memory 1Gi) and
# ingest.affordanceWorkers (1).
# "learner" is its learner part (learner.py, nam_heads.sdk.learner; docs/design-head-learning.md
# section 7). The gpu entries are the learner this head was built against; the chart's
# headLearning.learners.affordance overrides them.
IMPL_META = {
    "slot": "affordance",
    "name": "default",
    "version": "1.0.1",
    "contract": "affordance-v1",
    "description": "Affordance axis: the predicates the text links its entities to, as canonical "
                   "verb values (synonym fold, or the verb sense table), with a learner part for "
                   "unknown verbs.",
    "output": "affordance-senses-v1",
    "resources": {"cpu": 1.0, "memory_mb": 1024, "workers": 1},
    "dependencies": [],
    "learner": {
        "module": "learner",
        "version": "r5",
        "gpu": {"image": "tdenton8772/nam-head-learner:v0.0.1",
                "model_id": "head-learner-affordance-r5",
                "port": 50051, "disk_gb": 40,
                "gpu_types": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                              "NVIDIA L4", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX A6000"]},
    },
}
from .predicates import BE_FORMS, _SEED_UNFOLDED, extract_predicates
from . import senses as _senses
from .predicate_store import PendingObservations, StoreLexicon, supports_docs
from src.encoders.sketch_slots import action_slots, resolved_sketch
from src.nlp.sketch.model import RESOLVED_CONTRACT as SKETCH_CONTRACT
from src.encoders.record_parse import with_parse

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

import os


def _sentence_index(bounds: List[Tuple[int, int]]):
    """token index -> the index of the first sentence (start, end) holding
    it, else 0: a binary search when the ranges are ordered and disjoint (as
    a parse yields them), else the linear scan."""
    import bisect
    ordered = all(bounds[k][1] <= bounds[k + 1][0] for k in range(len(bounds) - 1)) and \
        all(a <= b for a, b in bounds)
    if not ordered:
        return lambda i: next((k for k, (a, b) in enumerate(bounds) if a <= i < b), 0)
    starts = [a for a, _ in bounds]

    def find(i):
        k = bisect.bisect_right(starts, i) - 1
        return k if k >= 0 and bounds[k][0] <= i < bounds[k][1] else 0
    return find


def _senses_on() -> bool:
    return os.getenv("NAM_AFFORDANCE_SENSES", "0") == "1"


class AffordanceEncoder(EncoderHead):
    HEAD_TYPE = "affordance"

    def create_axis_encoder(self):
        """This head's own encoder, from its own package (``encoder/charcnn``); with verb senses on, its sense table."""
        from pathlib import Path as _Path
        from src.encoders.axis_encoder import package_axis_encoder
        if _senses_on():
            from .senses import SenseTableAxisEncoder, table
            return SenseTableAxisEncoder(table())
        return package_axis_encoder(self.head_type, _Path(__file__).resolve().parent / "encoder" / "charcnn")

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **self.config}

        # Emission mode. "predicate": one address per distinct predicate in the
        # text, each carrying the token indices it occurs at, so the bundler
        # can link every entity to ITS predicate (the bundler needs a value
        # per entity, not per record). "selected": the legacy primary +
        # optional secondary verb per record.
        self.emit_mode = str(cfg.get("emit_mode", "predicate")).lower()
        self.emit_copular = bool(cfg.get("emit_copular", False))
        # Predicate lexicon in the category store (predicate_store.py):
        # learn = ingest writes observations; use_learned = learned entries
        # (past their thresholds) take part in extraction.
        self.learn_predicates = bool(cfg.get("learn_predicates", True))
        self.use_learned_predicates = bool(cfg.get("use_learned_predicates", True))
        # Commit window for lexicon observations, in batches
        # (predicate_store.PendingObservations): 1 commits every batch; N sums
        # N batches' observations and writes each touched doc once.
        every = int(cfg.get("lexicon_commit_batches",
                            os.environ.get("NAM_AFF_LEXICON_COMMIT_BATCHES", "1")) or 1)
        self._lex_pending = PendingObservations(every) if every > 1 else None
        self.max_predicate_values = int(cfg.get("max_predicate_values", 24))
        # Verb senses (senses.py, NAM_AFFORDANCE_SENSES=1): values are word@cell from the sense table, no
        # synonym fold. With ``self.parking`` (set by the factory, slots.create_head: an ingest stage with
        # a durable hand-off, learning on for this slot) a record with a verb the table does not know is
        # parked for the learning queue (docs/design-head-learning.md).
        self.senses = _senses.enabled()
        self._learned_cache: Dict[str, dict] = {}       # learned sense docs seen (positive only)

        # Hard cap (selected mode only)
        self.max_affordances = min(int(cfg.get("max_affordances", 2)), 2)

        # Scoring + selection thresholds
        self.weights = cfg.get("weights", {})
        self.abs_min = cfg.get("absolute_min_confidence", 0.25)
        self.soft_min = cfg.get("soft_min_confidence", 0.08)

        # Secondary affordance controls
        self.secondary_delta = cfg.get("secondary_delta", 0.15)
        self.secondary_min = cfg.get("secondary_min_confidence", 0.45)

        # Firewalls
        self.event_verbs = set(cfg.get("event_verbs", []))
        self.result_verbs = set(cfg.get("result_verbs", []))
        self.causal_verbs = set(cfg.get("causal_verbs", []))
        self.light_verbs = set(cfg.get("light_verbs", []))
        self.frame_verbs = (frozenset(cfg.get("frame_verbs", [])),
                            frozenset(cfg.get("frame_verbs_positional", [])))

        # Synonym folding
        self.synonyms = VERB_SYNONYMS

        # Ontology-aware scoring
        self.ontology_aware = bool(cfg.get("ontology_aware", True))
        self.ontology_relevance_weight = float(cfg.get("ontology_relevance_weight", 1.0))
        self.ner_scoping_weight = float(cfg.get("ner_scoping_weight", 0.5))

        # Firewall exemptions and NER relevance from definitions
        self.firewall_ontology_exemptions = FIREWALL_ONTOLOGY_EXEMPTIONS
        self.ner_entity_type_relevance = NER_ENTITY_TYPE_RELEVANCE

        # Category store -- maps verb lemmas to semantic categories
        self.category_store = create_category_store(
            backend=cfg.get("category_store_backend", "sqlite"),
            fallback=cfg.get("category_store_fallback", "memory"),
            path=cfg.get("category_store_path"),
            seed=cfg.get("category_store_seed", True),
            kv_on_cache_miss=self.kv_on_cache_miss,
        )

        logger.info(
            "AffordanceEncoder initialized "
            f"(emit_mode={self.emit_mode}, max_affordances={self.max_affordances}, "
            f"ontology_aware={self.ontology_aware}, "
            f"category_store={type(self.category_store).__name__}({self.category_store.count()} entries))"
        )

    def get_metrics(self) -> Dict:
        base = super().get_metrics()
        base["category_store_size"] = self.category_store.count()
        return base

    # Which selection knobs each emit mode applies (config_summary, README).
    # Predicate mode keeps every predicate that clears soft_min: the bundler
    # needs a value per ENTITY, and abs_min / primary-secondary /
    # elbow / cluster / max_affordances choose 1-2 verbs per RECORD, which
    # is what left 56% of bundles empty. Its fan-out bounds are
    # max_predicate_values per record (head) and
    # predicate_linking.MAX_AFFS_PER_BUNDLE per bundle (bundler).
    SELECTION_KNOBS = {
        "selected": ("absolute_min_confidence", "soft_min_confidence", "secondary_delta",
                     "secondary_min_confidence", "max_affordances"),
        "predicate": ("soft_min_confidence", "max_predicate_values", "max_affs_per_bundle"),
    }

    def config_summary(self) -> Dict:
        from src.addressing.predicate_linking import MAX_AFFS_PER_BUNDLE, PRED_WINDOW
        from .predicate_store import (LEARN_BASE_MIN_BARE, LEARN_BASE_MIN_INFLECTED,
                                      LEARN_PHRASAL_MIN, LEARN_PHRASAL_RATIO, ONT_FREQ_MIN)
        return {
            "emit_mode": self.emit_mode,
            "selection_knobs_applied": list(self.SELECTION_KNOBS.get(self.emit_mode, ())),
            "max_affordances": self.max_affordances,
            "soft_min_confidence": self.soft_min,
            "absolute_min_confidence": self.abs_min,
            "max_predicate_values": self.max_predicate_values,
            "max_affs_per_bundle": MAX_AFFS_PER_BUNDLE,
            "link_window_tokens": PRED_WINDOW,
            "emit_copular": self.emit_copular,
            "learn_predicates": self.learn_predicates,
            "use_learned_predicates": self.use_learned_predicates,
            "learn_thresholds": {
                "base_min_bare": LEARN_BASE_MIN_BARE, "base_min_inflected": LEARN_BASE_MIN_INFLECTED,
                "phrasal_min": LEARN_PHRASAL_MIN, "phrasal_ratio": LEARN_PHRASAL_RATIO,
                "ont_freq_min": ONT_FREQ_MIN,
            },
            "ontology_aware": self.ontology_aware,
            "category_store_backend": type(self.category_store).__name__,
        }

    # --------------------------------------------------
    # Canonicalization
    # --------------------------------------------------
    def _canonicalize(self, lemma: str) -> str:
        """Apply synonym folding to a verb lemma."""
        return self.synonyms.get(lemma, lemma)

    # --------------------------------------------------
    # Ontology context
    # --------------------------------------------------
    def _get_ontology_type(self, context: Optional[Dict]) -> Optional[str]:
        """Extract top ontology concept from context (set by registry accumulation)."""
        if not context:
            return None
        ont_result = context.get("ontology_result")
        if not ont_result or not getattr(ont_result, "addresses", None):
            return None
        first = ont_result.addresses[0]
        concept = getattr(first, "concept", None)
        if concept and concept != "__null__":
            return get_registry().nearest_key(concept, ONTOLOGY_KEYS)
        return None

    # --------------------------------------------------
    # NER extraction
    # --------------------------------------------------
    def _get_ner_entity_types(self, spacy_doc) -> Set[str]:
        """Extract unique NER labels from the parse's .ents."""
        if spacy_doc is None:
            return set()
        return {ent.label_ for ent in spacy_doc.ents}

    def _compute_ner_relevance(self, ner_labels: Set[str], category: str) -> float:
        """Aggregate NER relevance for a category across detected entity types.

        Uses max() across labels to avoid inflation. Returns 1.0 if no label
        has an opinion or NER scoping is disabled.
        """
        if not ner_labels or not category or self.ner_scoping_weight <= 0.0:
            return 1.0

        multipliers = []
        for label in ner_labels:
            cat_weights = self.ner_entity_type_relevance.get(label, {})
            if category in cat_weights:
                multipliers.append(cat_weights[category])

        if not multipliers:
            return 1.0

        raw = max(multipliers)
        # Scale by ner_scoping_weight: 0.0 = neutral, 1.0 = full multiplier
        return 1.0 + (raw - 1.0) * self.ner_scoping_weight

    # --------------------------------------------------
    # Shared by the legacy and predicate paths
    # --------------------------------------------------
    @staticmethod
    def _role_features(token) -> Tuple[str, bool, bool, bool, bool]:
        """(role, has_subj, has_obj, passive, is_purpose) of a verb token."""
        dep = token.dep_
        if dep == "ROOT":
            role = "root"
        elif dep == "conj" and token.head.dep_ == "ROOT":
            role = "conj_root"
        elif dep in ("relcl", "acl", "ccomp", "xcomp", "advcl", "pcomp"):
            role = dep
        else:
            role = "other"
        has_subj = any(c.dep_ in {"nsubj", "nsubjpass"} for c in token.children)
        has_obj = any(c.dep_ in {"dobj", "obj"} for c in token.children)
        morph = getattr(token, "morph", None)
        passive = bool(morph is not None and hasattr(morph, "get") and morph.get("Voice") == ["Pass"])
        is_purpose = dep == "xcomp" or (dep == "advcl" and any(
            c.dep_ == "mark" and c.text.lower() in {"so", "so that"} for c in token.children))
        return role, has_subj, has_obj, passive, is_purpose

    def _firewall_status(self, words, ontology_type: Optional[str]) -> str:
        """"pass", "exempt" or "block" for a verb (any of ``words`` firewalled)."""
        fw = [w for w in words if w in self.event_verbs or w in self.result_verbs
              or w in self.causal_verbs]
        if not fw:
            return "pass"
        exempt = ontology_lookup(self.firewall_ontology_exemptions, ontology_type, set()) or set()
        return "exempt" if any(w in exempt for w in fw) else "block"

    def _score_candidate(self, canon, role, has_subj, has_obj, is_purpose, passive,
                         firewall_exempted, ontology_type, ner_labels, cat_result):
        """Config weights + ontology relevance + NER relevance -> (score,
        category, inferred). The one scoring rule for both paths."""
        w = self.weights
        base = w.get(role, 0.45)
        if has_subj:
            base += w.get("subj_bonus", 0.1)
        if has_obj:
            base += w.get("obj_bonus", 0.1)
        if is_purpose:
            base += w.get("purpose_bonus", 0.25)
        if passive:
            # Passive voice lowers salience but takes at most half the base:
            # a passive verb outside the main clause (role "other", 0.2) is
            # still what happened to its subject, and a full 0.2 penalty put
            # it at 0, under soft_min ("is recognized", "was transformed").
            base -= min(w.get("passive_penalty", 0.2), base / 2)
        if firewall_exempted:
            base *= 0.6
        if cat_result:
            category, _confidence = cat_result
            inferred = False
        else:
            category = self._infer_category(canon, ontology_type)
            inferred = True
        score = base
        if ontology_type and self.ontology_aware and category != "uncategorized":
            raw_relevance = get_ontology_relevance(ontology_type, category)
            score *= 1.0 + (raw_relevance - 1.0) * self.ontology_relevance_weight
        if ner_labels and category != "uncategorized":
            score *= self._compute_ner_relevance(ner_labels, category)
        return max(0.0, min(1.0, score)), category, inferred

    # --------------------------------------------------
    # Auto-classification
    # --------------------------------------------------
    def _infer_category(self, lemma: str, ontology_type: Optional[str]) -> str:
        """Infer a category for an unknown verb WITHOUT persisting to the store.

        If ontology context is available, pick the highest-weighted category
        for that ontology type.  Otherwise return ``"uncategorized"``.
        """
        if ontology_type:
            category = get_best_category_for_ontology(ontology_type)
            if category:
                return category
        return "uncategorized"

    def _auto_classify(self, lemma: str, ontology_type: Optional[str]) -> str:
        """Classify an unknown verb and persist it to the store.

        Calls ``_infer_category`` then writes the result to the store.
        Auto-classified entries get confidence 0.5 to distinguish from seed data.
        """
        category = self._infer_category(lemma, ontology_type)
        self.category_store.put_category_weighted(
            lemma, category, confidence=0.5, ontology_type=None,
        )
        return category

    # --------------------------------------------------
    # Cross-text batch encode -- called by MultiHeadEncoder.encode_batch
    # --------------------------------------------------
    def encode_batch_texts(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[Dict]]] = None,
    ) -> List[EncodingResult]:
        """Encode multiple texts, batching ALL verb category KV lookups across texts."""
        if not texts:
            return []
        n = len(texts)
        if contexts is None:
            contexts = [None] * n
        contexts = [self._with_doc(texts[i], contexts[i]) for i in range(n)]

        if self.emit_mode == "predicate":
            return self._predicate_results(contexts, "ingest")

        # Phase 1: Extract verb candidates for ALL texts (CPU-only)
        per_text_data = []  # (candidates, ontology_type, ner_labels)
        for i in range(n):
            ctx = contexts[i]
            spacy_doc = ctx.get("spacy_doc")
            ontology_type = self._get_ontology_type(ctx)
            ner_labels = self._get_ner_entity_types(spacy_doc)

            if spacy_doc is not None:
                candidates = self._extract_from_doc(spacy_doc, ontology_type=ontology_type)
            else:
                candidates = self._extract_regex(texts[i])

            per_text_data.append((candidates, ontology_type, ner_labels))

        # Phase 2: Collect all unique (canon, ontology_type) queries
        all_queries = set()
        for candidates, ontology_type, _ in per_text_data:
            seen_canon: Set[str] = set()
            for lemma, role, has_subj, has_obj, is_purpose, passive, order, firewall_exempted in candidates:
                canon = self._canonicalize(lemma)
                if canon not in seen_canon:
                    seen_canon.add(canon)
                    all_queries.add((canon, ontology_type))

        # Phase 3: ONE batch get for all category lookups
        cat_cache = self.category_store.get_categories_weighted_batch(list(all_queries))

        # Phase 4: Score and select per text using cached categories
        results = []
        for i, (candidates, ontology_type, ner_labels) in enumerate(per_text_data):
            selected = self._score_and_select(
                candidates, ontology_type=ontology_type, ner_labels=ner_labels,
                cat_cache=cat_cache,
            )

            if not selected:
                addresses = [SemanticAddress(
                    head_type=self.head_type, concept="__null__",
                    confidence=1.0, metadata={"method": "null"},
                )]
            else:
                addresses = [
                    SemanticAddress(
                        head_type=self.head_type, concept=lemma_out,
                        confidence=score, metadata=meta,
                    )
                    for lemma_out, score, meta in selected
                ]

            metadata = {
                "mode": "ingest",
                "extraction": "parse" if contexts[i].get("spacy_doc") is not None else "regex",
                "count": len(addresses),
            }
            if ontology_type:
                metadata["ontology_type"] = ontology_type
            if ner_labels:
                metadata["ner_labels"] = sorted(ner_labels)
            results.append(EncodingResult(self.head_type, addresses, metadata))

        return results

    # --------------------------------------------------
    # Public encode APIs
    # --------------------------------------------------
    def _with_doc(self, text: str, context: Optional[Dict]) -> Dict:
        """Context with a parsed doc: a text that arrives without one is
        parsed with the rule parser (record_parse.with_parse), so every value
        comes from the same extractor -- one value space. A text that still
        has no parse gives __null__ in predicate mode and the regex reading
        in selected mode."""
        return with_parse(text, context, self.head_type)

    def _sentence_label(self, context: Optional[Dict]):
        """si -> the sentence's ontology label (or CURIE).

        ``ontology_result.metadata["sentence_ontologies"]`` is written by the
        ontology stage at ingest and by MultiHeadEncoder at query
        (encoders/sentence_ontology.py); a sentence without one falls back to
        the record ontology (``addresses[0]``)."""
        record = self._get_ontology_type(context)
        per_sent = {}
        ont = (context or {}).get("ontology_result")
        meta = getattr(ont, "metadata", None) or {}
        for k, v in (meta.get("sentence_ontologies") or {}).items():
            try:
                per_sent[int(k)] = v
            except (TypeError, ValueError):
                continue

        def label(si: int):
            lab = per_sent.get(si)
            # No per-sentence class (the head abstained on it): the record's.
            return record if (not lab or lab == B.UNCLASSIFIED) else lab
        return label

    def _sentence_profiles(self, context: Optional[Dict]):
        """Per-sentence extraction profile from ONTOLOGY_FUNCTION_VERBS
        (resolved once per distinct sentence ontology of the record)."""
        if not self.ontology_aware:
            return None
        label = self._sentence_label(context)
        by_label: Dict = {}

        def profile(si):
            lab = label(si)
            hit = by_label.get(lab)
            if hit is None:
                hit = by_label[lab] = {"function_verbs": ontology_lookup(
                    ONTOLOGY_FUNCTION_VERBS, lab, FUNCTION_VERBS_DEFAULT)}
            return hit
        return profile

    def _lexicon(self, mode: str):
        """Store-backed lexicon (seed + learned); query mode never writes."""
        store = getattr(self, "category_store", None)
        if store is not None and supports_docs(store):
            ingest = mode == "ingest" and self.learn_predicates
            return StoreLexicon(store, learn=ingest, use_learned=self.use_learned_predicates,
                                pending=self._lex_pending if ingest else None, fold=not self.senses)
        return None

    def flush_lexicon(self) -> int:
        """Commit the open commit window now and send the store's queued
        writes (worker shutdown). Returns the docs committed."""
        if self._lex_pending is None:
            return 0
        lex = self._lexicon("ingest")
        n = lex.commit(force=True) if lex is not None else 0
        flush = getattr(getattr(self, "category_store", None), "_flush_pending", None)
        if flush is not None:
            flush()
        return n

    def reset_session_state(self) -> Dict[str, int]:
        """Cold session reset: observations not yet committed came from the
        flushed bucket's state and must not land in the emptied one."""
        dropped = self._lex_pending.clear() if self._lex_pending is not None else 0
        return {"lexicon_pending": dropped}

    @staticmethod
    def _sent_bounds(doc) -> List[Tuple[int, int]]:
        return [(x.start, x.end) for x in (getattr(doc, "sents", None) or [])] or [(0, len(doc))]

    def _predicate_results(self, contexts: List[Dict], mode: str) -> List[EncodingResult]:
        """Predicate-mode encode for a batch.

        Per predicate the extractor settles: firewall (+ ontology
        exemptions), category lookup, the shared score (config weights,
        ONTOLOGY_RELEVANCE, NER relevance) and the soft_min threshold.
        Store traffic: ONE batch read (prefetch: lexicon docs and the
        vcat:: category keys) and, at ingest, the batch's writes (lexicon
        observations + auto-classified categories, relayed to LMDB).
        """
        lex = self._lexicon(mode)
        labels = [self._sentence_label(c) for c in contexts]
        docs = [c.get("spacy_doc") for c in contexts]
        bounds_of = [self._sent_bounds(d) if d is not None else [] for d in docs]
        if lex is not None:
            lex.prefetch(docs, sentence_labels=[
                {lab(si) for si in range(len(b))} if d is not None else set()
                for d, lab, b in zip(docs, labels, bounds_of)])
        extracted = []
        queries = set()
        light = frozenset(self.light_verbs)
        for doc, ctx, lab, bounds in zip(docs, contexts, labels, bounds_of):
            preds = [] if doc is None else extract_predicates(
                doc, emit_copular=self.emit_copular, light=light,
                sentence_profile=self._sentence_profiles(ctx),
                lexicon=lex if lex is not None else (_SEED_UNFOLDED if self.senses else None),
                frame=self.frame_verbs)
            sentence_of = _sentence_index(bounds)
            sis = [sentence_of(p.token_i) for p in preds]
            extracted.append((preds, sis))
            for p, si in zip(preds, sis):
                queries.add((p.value, lab(si)))
        if lex is not None:
            cat_cache = lex.categories(queries)
        else:
            cat_cache = self.category_store.get_categories_weighted_batch(sorted(queries, key=str))

        pending = []
        to_classify = set()
        for (preds, sis), doc, ctx, lab in zip(extracted, docs, contexts, labels):
            ner = self._get_ner_entity_types(doc)
            # The resolved sentence sketch (NLP2), when the record carries
            # one: each value is a clause's action, attached to the chains
            # of its actor and patient (sketch_slots.action_slots).
            sk = resolved_sketch(doc)
            slots = action_slots(sk) if sk is not None else None
            kept = []
            for p, si in zip(preds, sis):
                ont = lab(si)
                fw = self._firewall_status((p.value, p.lemma), ont)
                if fw == "block":
                    continue
                if p.kind == "copular":
                    role, subj, obj, passive, purpose = "copular", True, False, False, False
                else:
                    role, subj, obj, passive, purpose = self._role_features(doc[p.token_i])
                slot = slots.get(p.token_i) if slots is not None else None
                if slot is not None and p.kind != "copular":
                    passive = slot[0] == "passive"      # the clause's voice
                score, category, inferred = self._score_candidate(
                    p.value, role, subj, obj, purpose, passive, fw == "exempt",
                    ont, ner, cat_cache.get((p.value, ont)))
                if score < self.soft_min:
                    continue
                if inferred and p.kind != "copular":
                    to_classify.add((p.value, ont))
                if lex is not None and lex.learn:
                    # the store learns from every predicate, with a slot or not
                    lex.record_value(p, ontology_class(ont))
                if slots is not None and slot is None:
                    continue        # no clause has it as its action: empty over wrong
                kept.append((p, score, category, slot[1] if slot is not None else None))
            pending.append((doc, ctx, kept, sk))
        if self.senses:
            results = self._sense_results(pending, mode, lex)
        else:
            results = [self._predicate_encoding(doc, ctx, mode, kept, lex, sketch=sk is not None)
                       for doc, ctx, kept, sk in pending]
        if lex is not None:
            lex.commit()
        if mode == "ingest":
            for value, ont in sorted(to_classify, key=str):
                self._auto_classify(value, ont)
        return results

    # ---- verb senses (senses.py) ------------------------------------------------------------------
    def _sense_results(self, pending, mode: str, lex) -> List[EncodingResult]:
        """Resolve every kept predicate to its sense (word@cell). A record with a verb nothing knows is
        PARKED at ingest when the factory set ``self.parking``: its result carries metadata "parked" (the
        values and where they occur) and no addresses, and the join takes the record out of the pipeline.
        At query, or with parking off, unknown verbs are left out."""
        tbl = _senses.table()
        plans, need = [], set()
        for doc, _ctx, kept, _sk in pending:
            rows = []
            for p, score, category, chains in kept:
                if p.kind == "copular":
                    rows.append((p, score, category, chains, None, None))
                    continue
                word = p.lemma if p.kind == "phrasal" else p.value
                frame = p.particle if (p.kind == "phrasal" and p.particle) else _senses.frame_after(doc, p.token_i)
                if tbl.seed_cell(word, frame) is None and word not in tbl.not_verb:
                    need.add(word)
                rows.append((p, score, category, chains, word, frame))
            plans.append(rows)
        learned = self._learned_senses(sorted(need))
        results = []
        for (doc, ctx, _kept, sk), rows in zip(pending, plans):
            new_kept, parked, ont_key = [], {}, {}
            for p, score, category, chains, word, frame in rows:
                if word is None:
                    new_kept.append((p, score, category, chains))
                    continue
                kind, cell = tbl.resolve(word, frame, learned.get(word))
                if kind == _senses.UNKNOWN and p.kind != "phrasal":
                    # The repaired base can be a junk base the lexicon learned earlier ("changing" ->
                    # "chang"); a base candidate of the token that the sense table knows wins first.
                    alt = self._known_base(doc[p.token_i].text, word, tbl)
                    if alt is not None:
                        word = alt
                        kind, cell = tbl.resolve(word, frame)
                if kind == _senses.CELL:
                    v = _senses.value_of(word, cell)
                    ont_key[v] = p.value
                    new_kept.append((dataclasses.replace(p, value=v), score, category, chains))
                elif kind == _senses.UNKNOWN and word not in parked:
                    parked[word] = self._occurrence(doc, p, word, frame)
            if parked and mode == "ingest" and self.parking:
                results.append(parked_result(self.head_type, list(parked.values()), mode=mode,
                                             extraction="predicate", lexicon="senses"))
            else:
                results.append(self._predicate_encoding(doc, ctx, mode, new_kept, lex,
                                                        sketch=sk is not None, ont_key=ont_key))
        return results

    @staticmethod
    def _known_base(token: str, word: str, tbl) -> Optional[str]:
        """A base candidate of ``token`` (predicates.surface_candidates) the sense table's seed knows."""
        from .predicates import surface_candidates
        for c in surface_candidates(token):
            if c != word and tbl.seed_cell(c) is not None:
                return c
        return None

    def _learned_senses(self, words: List[str]) -> Dict[str, dict]:
        """Learned sense docs (vcat::sense:<word>) for words the seed does not know: a positive-only cache,
        then ONE direct session read for the rest. Not through the store's caches: they cache misses,
        and a word learned after its first miss would park every later record."""
        out = {w: self._learned_cache[w] for w in words if w in self._learned_cache}
        missing = [w for w in words if w not in out]
        store = getattr(self, "category_store", None)
        kv = getattr(store, "_kv", None)
        if not missing or kv is None:
            return out
        keys = [store._key(_senses.LEARNED_PREFIX + w) for w in missing]
        try:
            got = kv.pipeline_get(keys)
        except Exception as e:  # unreadable: those words count as unknown this batch (they park, not lost)
            logger.warning("affordance.learned_read_failed n=%d err=%s", len(keys), e)
            return out
        for w, k in zip(missing, keys):
            v = got.get(k)
            raw = v[0] if isinstance(v, tuple) else v
            if raw:
                try:
                    doc = json.loads(raw)
                except Exception:
                    continue
                out[w] = self._learned_cache[w] = doc
        return out

    def _occurrence(self, doc, p, word: str, frame: str) -> Dict:
        """Where a parked verb occurs, for the learner: the token, its sentence and the frame."""
        tok = doc[p.token_i]
        sent = None
        for start, end in self._sent_bounds(doc):
            if start <= p.token_i < end:
                sent = " ".join(doc[k].text for k in range(start, end))
                break
        return {"value": word, "token": tok.text, "frame": frame, "sentence": (sent or "")[:1000]}

    def _predicate_result(self, doc, context: Optional[Dict], mode: str) -> EncodingResult:
        ctx = dict(context or {})
        ctx["spacy_doc"] = doc
        return self._predicate_results([ctx], mode)[0]

    def _predicate_encoding(self, doc, context, mode, kept, lex, sketch: bool = False,
                            ont_key: Optional[Dict[str, str]] = None) -> EncodingResult:
        """One address per distinct value (best score), with its token
        positions; at most max_predicate_values per record, ordered by score
        (addresses[0] is the strongest, as in the legacy primary). Metadata is kept small: tokens always; lemma/kind only when
        they differ from the value / from a plain verb; ont_freq top 3.

        ``sketch``: the values were read from the resolved sketch's clauses;
        each carries ``links``, [chain, action token] per chain its clause's
        actor or patient carries, in text order -- the bundler gives an
        entity the values linked to its chains. A value whose clause has no
        chained participant carries no links (it reaches no bundle)."""
        by_value: Dict[str, Dict] = {}
        best: Dict[str, float] = {}
        for p, score, category, chains in kept:
            m = by_value.get(p.value)
            if m is None:
                m = {"method": "predicate", "tokens": [p.token_i]}
                if p.kind != "verb":
                    m["kind"] = p.kind
                if p.lemma != p.value:
                    m["lemma"] = p.lemma
                if category and category != "uncategorized":
                    m["category"] = category          # existing metadata contract
                if lex is not None:
                    of = lex.ont_freq((ont_key or {}).get(p.value, p.value))
                    if of:
                        m["ont_freq"] = dict(sorted(of.items(), key=lambda kv: (-kv[1], kv[0]))[:3])
                if sketch:
                    m["links"] = []
                by_value[p.value] = m
                best[p.value] = score
            else:
                m["tokens"].append(p.token_i)
                best[p.value] = max(best[p.value], score)
            for c in chains or ():
                if [c, p.token_i] not in m["links"]:
                    m["links"].append([c, p.token_i])
        order = sorted(by_value, key=lambda v: (-best[v], by_value[v]["tokens"][0]))
        order = order[: self.max_predicate_values]
        if order:
            addresses = [SemanticAddress(head_type=self.head_type, concept=v,
                                         confidence=round(best[v], 3), metadata=by_value[v])
                         for v in order]
        else:
            addresses = [SemanticAddress(head_type=self.head_type, concept="__null__",
                                         confidence=1.0, metadata={"method": "null"})]
        metadata = {"mode": mode, "extraction": "predicate", "count": len(addresses),
                    "lexicon": "store" if lex is not None else "seed"}
        if sketch:
            metadata["sketch"] = SKETCH_CONTRACT     # values carry their slots' chains
        ontology_type = self._get_ontology_type(context)
        if ontology_type:
            metadata["ontology_type"] = ontology_type
        ner_labels = self._get_ner_entity_types(doc) if doc is not None else set()
        if ner_labels:
            metadata["ner_labels"] = sorted(ner_labels)
        return EncodingResult(self.head_type, addresses, metadata)

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        context = self._with_doc(text, context)
        if self.emit_mode == "predicate":
            return self._predicate_results([context], "ingest")[0]
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        ner_labels = self._get_ner_entity_types(spacy_doc)
        addresses = self._extract_affordances(
            text, spacy_doc=spacy_doc, ontology_type=ontology_type, ner_labels=ner_labels,
        )
        metadata = {
            "mode": "ingest",
            "extraction": "parse" if spacy_doc is not None else "regex",
            "count": len(addresses),
        }
        if ontology_type:
            metadata["ontology_type"] = ontology_type
        if ner_labels:
            metadata["ner_labels"] = sorted(ner_labels)
        return EncodingResult(self.head_type, addresses, metadata)

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        context = self._with_doc(query, context)
        if self.emit_mode == "predicate":
            return self._predicate_results([context], "query")[0]
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        ner_labels = self._get_ner_entity_types(spacy_doc)
        addresses = self._extract_affordances(
            query, spacy_doc=spacy_doc, ontology_type=ontology_type, ner_labels=ner_labels,
        )
        metadata = {
            "mode": "query",
            "extraction": "parse" if spacy_doc is not None else "regex",
            "count": len(addresses),
        }
        if ontology_type:
            metadata["ontology_type"] = ontology_type
        if ner_labels:
            metadata["ner_labels"] = sorted(ner_labels)
        return EncodingResult(self.head_type, addresses, metadata)

    # --------------------------------------------------
    def _extract_affordances(
        self,
        text: str,
        spacy_doc=None,
        ontology_type: Optional[str] = None,
        ner_labels: Optional[Set[str]] = None,
    ) -> List[SemanticAddress]:
        logger.debug("=== Affordance extraction start ===")
        logger.debug(f"Text: {text}")

        # The record's parse; regex only when there is none (record_parse)
        if spacy_doc is not None:
            candidates = self._extract_from_doc(spacy_doc, ontology_type=ontology_type)
        else:
            candidates = self._extract_regex(text)

        logger.debug(f"Raw verb candidates ({len(candidates)}):")
        for c in candidates:
            logger.debug(f"  {c}")

        selected = self._score_and_select(
            candidates, ontology_type=ontology_type, ner_labels=ner_labels,
        )

        if not selected:
            logger.debug("No affordances selected -> __null__")
            return [
                SemanticAddress(
                    head_type=self.head_type,
                    concept="__null__",
                    confidence=1.0,
                    metadata={"method": "null"},
                )
            ]

        logger.debug("Final affordances:")
        for s in selected:
            logger.debug(f"  {s}")
        logger.info("affordance.selected %s", selected)

        return [
            SemanticAddress(
                head_type=self.head_type,
                concept=lemma,
                confidence=score,
                metadata=meta,
            )
            for lemma, score, meta in selected
        ]

    # --------------------------------------------------
    def _extract_copular_affordance(self, token) -> Optional[tuple]:
        """Detect copular construction: subject + be-verb + predicate complement.

        Returns (concept, role, confidence) or None.
        - NOUN predicate (attr): "Lincoln is a president" → "classify"
        - ADJ predicate (acomp): "The sky is blue" → "describe"
        """
        for child in token.children:
            if child.dep_ in ("attr", "acomp", "oprd"):
                if child.pos_ == "NOUN":
                    return ("classify", "copular", 0.60)
                elif child.pos_ == "ADJ":
                    return ("describe", "copular", 0.60)
                elif child.pos_ == "PROPN":
                    return ("classify", "copular", 0.55)
        return None

    def _extract_from_doc(self, doc, ontology_type: Optional[str] = None) -> List[tuple]:
        """Extract affordances from the record's parse.

        Returns tuples:
        (lemma, role, has_subj, has_obj, is_purpose, passive, order, firewall_exempted)

        Firewall check is ontology-aware: light verbs are always filtered,
        but event/result/causal verbs may be exempted in specific ontology contexts.
        """
        cand = []

        for i, token in enumerate(doc):
            lemma = token.lemma_.lower()

            # Copular detection: AUX be-verbs with a predicate complement.
            # The tagger may tag copular "is/was/are" AUX rather than VERB.
            if token.pos_ in ("AUX", "VERB") and lemma in ("be", "is", "was", "were", "am", "are", "been"):
                copular = self._extract_copular_affordance(token)
                if copular:
                    cop_lemma, cop_role, cop_conf = copular
                    has_subj = any(c.dep_ in {"nsubj", "nsubjpass"} for c in token.children)
                    cand.append((cop_lemma, cop_role, has_subj, False, False, False, i, False))
                    logger.debug(f"Copular affordance '{cop_lemma}' from '{token.text}' (conf={cop_conf})")
                    continue

            if token.pos_ != "VERB":
                continue

            # Light verbs: ALWAYS filtered (never exempt)
            if lemma in self.light_verbs:
                logger.debug(f"Filtered verb '{lemma}' (light verb, never exempt)")
                continue

            # Event/result/causal firewalls: check ontology exemptions
            fw = self._firewall_status((lemma,), ontology_type)
            if fw == "block":
                logger.debug(f"Filtered verb '{lemma}' (firewall)")
                continue
            firewall_exempted = fw == "exempt"

            # Structural role
            if token.dep_ == "ROOT":
                role = "root"
            elif token.dep_ == "conj" and token.head.dep_ == "ROOT":
                role = "conj_root"
            elif token.dep_ == "relcl":
                role = "relcl"
            elif token.dep_ == "acl":
                role = "acl"
            elif token.dep_ == "ccomp":
                role = "ccomp"
            elif token.dep_ == "xcomp":
                role = "xcomp"
            elif token.dep_ == "advcl":
                role = "advcl"
            elif token.dep_ == "pcomp":
                role = "pcomp"
            else:
                role = "other"

            has_subj = any(c.dep_ in {"nsubj", "nsubjpass"} for c in token.children)
            has_obj = any(c.dep_ in {"dobj", "obj"} for c in token.children)
            passive = token.morph.get("Voice") == ["Pass"]

            # Purpose detection
            is_purpose = False
            if token.dep_ == "advcl":
                if any(
                    c.dep_ == "mark" and c.text.lower() in {"so", "so that"}
                    for c in token.children
                ):
                    is_purpose = True
            if token.dep_ == "xcomp":
                is_purpose = True

            cand.append((lemma, role, has_subj, has_obj, is_purpose, passive, i, firewall_exempted))

        return cand

    def _extract_regex(self, text: str) -> List[tuple]:
        tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        cand = []
        for i, tok in enumerate(tokens):
            lemma = tok.lower()
            if lemma in self.light_verbs:
                continue
            if lemma.endswith(("ed", "ing")) or lemma in {"play", "repair", "restart", "bring", "take"}:
                cand.append((lemma, "regex", False, False, False, False, i, False))
        return cand

    # --------------------------------------------------
    def _score_and_select(
        self,
        cand: List[tuple],
        ontology_type: Optional[str] = None,
        ner_labels: Optional[Set[str]] = None,
        cat_cache: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """Score and select top affordances.

        Ontology relevance and NER scoping are integrated into scoring BEFORE
        primary/secondary selection so they influence which verbs survive the cut.

        Returns list of (concept, score, metadata_dict).
        """
        logger.debug("--- Scoring affordance candidates ---")

        # Step 1: canonicalize + dedup + score all candidates
        scored = []
        seen_canon: Set[str] = set()
        w = self.weights

        for lemma, role, has_subj, has_obj, is_purpose, passive, order, firewall_exempted in cand:
            canon = self._canonicalize(lemma)

            # Dedup on canonical form
            if canon in seen_canon:
                continue
            seen_canon.add(canon)

            if cat_cache is not None:
                cat_result = cat_cache.get((canon, ontology_type))
            else:
                cat_result = self.category_store.get_category_weighted(canon, ontology_type)
            score, category, inferred = self._score_candidate(
                canon, role, has_subj, has_obj, is_purpose, passive, firewall_exempted,
                ontology_type, ner_labels, cat_result)

            logger.debug(
                f"Verb '{lemma}'->'{canon}': role={role}, subj={has_subj}, obj={has_obj}, "
                f"purpose={is_purpose}, passive={passive}, category={category}, "
                f"exempted={firewall_exempted} -> score={score:.3f}"
            )

            scored.append((canon, score, role, is_purpose, order, category, inferred))

        if not scored:
            return []

        scored.sort(key=lambda x: (-x[1], x[4]))

        # --------------------------------------------------
        # Step 2: primary affordance (purpose override)
        purpose_scored = [s for s in scored if s[3] and s[1] >= self.abs_min]

        if purpose_scored:
            primary = purpose_scored[0]
            primary_is_purpose = True
        else:
            primary = next((s for s in scored if s[1] >= self.abs_min), None)
            primary_is_purpose = False

        if not primary:
            if scored[0][1] >= self.soft_min:
                primary = scored[0]
                primary_is_purpose = primary[3]
            else:
                return []

        selected = [primary]

        logger.debug(f"Primary affordance: {primary}")

        # --------------------------------------------------
        # Step 3: optional secondary affordance
        if self.max_affordances > 1:
            for cand_item in scored:
                if cand_item == primary:
                    continue

                canon, score, role, is_purpose, _, _cat, _inf = cand_item

                # Structural gate
                if role not in {"root", "conj_root", "xcomp", "advcl", "relcl", "acl", "ccomp", "pcomp"}:
                    continue

                # Absolute strength gate
                if score < self.secondary_min:
                    continue

                if primary_is_purpose:
                    # Purpose primary -> allow one strong, non-purpose support verb
                    if is_purpose:
                        continue
                    logger.debug(f"Secondary affordance accepted (purpose-primary): {cand_item}")
                    selected.append(cand_item)
                    break
                else:
                    # Non-purpose primary -> enforce score proximity
                    if (primary[1] - score) > self.secondary_delta:
                        continue
                    logger.debug(f"Secondary affordance accepted: {cand_item}")
                    selected.append(cand_item)
                    break

        # --------------------------------------------------
        # Step 4: persist auto-classification only for selected verbs
        for canon, _score, _role, _purp, _ord, _cat, inferred in selected:
            if inferred:
                self._auto_classify(canon, ontology_type)

        return [
            (
                canon,
                score,
                {"method": role, "category": category} if category and category != "uncategorized" else {"method": role},
            )
            for canon, score, role, _, _, category, _ in selected
        ]
