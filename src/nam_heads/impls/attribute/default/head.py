"""
Attribute encoder head.

Answers: "What qualities or properties are explicitly asserted in the text?"
Emits a bounded set of attributes (ADJ/adjectival phrases) or a single "__null__".

When ontology context is available (phase-1 ontology result in ``ctx["ontology_result"]``),
attribute scores are nudged by ontology-category relevance *before* adaptive selection
so that ontology can influence which attributes survive the cut, not just their ranking.

NER entity types from the parse's ``.ents`` provide an additional scoring signal
via per-category multipliers.

The parse is the record's own (``context["spacy_doc"]``, the rule backend's
ParsedDocument). A record without one is parsed with the same rule parser
(``src.encoders.record_parse``) before extraction.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from .config import DEFAULT_CONFIG
from .definitions import (
    get_ontology_relevance,
    get_best_category_for_ontology,
    evidence_category,
    excluded_categories,
    ontology_lookup,
    STOPWORD_ONTOLOGY_EXEMPTIONS,
    NER_ENTITY_TYPE_RELEVANCE,
    ONTOLOGY_KEYS,
)
from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import get_registry
from .storage.factory import create_category_store
from .predication import attribute_values, candidate_values
from . import learning as L
from .predication import LINK_WEIGHT as P_LINK_WEIGHT
from src.encoders.sketch_slots import MentionIndex, resolved_sketch, span_chains
from src.nlp.sketch.model import RESOLVED_CONTRACT as SKETCH_CONTRACT
from src.encoders.record_parse import with_parse

# What this default plugin declares (docs/design-head-plugins.md section 3), read with ast
# (nam_heads.sdk.impl_meta.read_module_literal): literals only. "resources" are the per-container
# limits and the containers per ingest pod it runs with on the reference deployment:
# the chart's ingest.axisHeadResources.limits (cpu "1", memory 1Gi) and
# ingest.attributeWorkers (2).
IMPL_META = {
    "slot": "attribute",
    "name": "default",
    "version": "1.0.2",
    "contract": "attribute-v1",
    "description": "Attribute axis: the qualities the text predicates of its entities (adjectives "
                   "and adjectival phrases), as store-normalized attribute values.",
    "output": "attribute-values-v1",
    "resources": {"cpu": 1.0, "memory_mb": 1024, "workers": 2},
    "dependencies": [],
    # What host stages read from it besides the head (nam_heads.sdk.exports).
    "provides": {
        "entity_admits": "definitions:entity_admits",
        "evidence_exclusions_enabled": "definitions:evidence_exclusions_enabled",
        "title_spans": "predication:title_spans",
        "seed": "exports:seed",
        "axis_encoder": "exports:axis_encoder",
    },
}

logger = logging.getLogger(__name__)

import os


class AttributeEncoder(EncoderHead):
    """
    Deterministic attribute extractor (lexical, low fanout, null-safe).
    """

    HEAD_TYPE = "attribute"

    def create_axis_encoder(self):
        """This head's own encoder (exports.axis_encoder, the one host stages read too)."""
        from .exports import axis_encoder
        return axis_encoder()

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **self.config}
        self.max_attributes = int(cfg.get("max_attributes", 3))
        self.min_len = int(cfg.get("min_len", 2))
        self.wordlist = self._load_wordlist(cfg.get("wordlist"))
        self.synonyms = {k.lower(): v.lower() for k, v in cfg.get("synonyms", {}).items()}
        self.stopwords = set(w.lower() for w in cfg.get("stopwords", []))
        self.weights = cfg.get("weights", {})
        self.ontology_aware = bool(cfg.get("ontology_aware", True))
        self.ontology_relevance_weight = float(cfg.get("ontology_relevance_weight", 1.0))
        self.ner_scoping_weight = float(cfg.get("ner_scoping_weight", 0.5))
        # Predication mode: values are what the text predicates of each
        # subject (copular, appositive, title, as-role, modifier, pronoun),
        # each carrying the subject span it belongs to, so the bundler can
        # link it per entity. No per-record cap: the bundler needs a value
        # per ENTITY.
        self.predication_mode = bool(cfg.get(
            "predication_mode",
            os.environ.get("NAM_ATTRIBUTE_PREDICATION", "1") != "0"))
        # Ontology steer: the ontology of the sentence a predication is read
        # from (ontology_result.metadata["sentence_ontologies"], falling back
        # to the record ontology) decides which kinds of evidence count
        # through ONTOLOGY_RELEVANCE. NAM_ATTRIBUTE_SENTENCE_STEER=0 steers by
        # the record ontology instead.
        # Per subject, the same adaptive selection as the legacy per-record
        # cut, bounded generously (the bundler needs a value per ENTITY).
        self.max_attributes_per_subject = int(cfg.get("max_attributes_per_subject", 8))
        # Predication values per record (after the per-subject cut). The
        # legacy record-level selection (max_attributes) still governs the
        # legacy extraction sources, which run alongside.
        self.max_predicated_per_record = int(cfg.get("max_predicated_per_record", 16))
        # Explicit hard filter: EVIDENCE_EXCLUSIONS (definitions.py).
        # OFF by default (definitions.evidence_exclusions_enabled).
        from .definitions import evidence_exclusions_enabled
        self.evidence_exclusions = bool(cfg.get(
            "evidence_exclusions", evidence_exclusions_enabled()))
        self.sentence_steering = bool(cfg.get(
            "sentence_steering",
            os.environ.get("NAM_ATTRIBUTE_SENTENCE_STEER", "1") != "0"))

        # Stopword exemptions and NER relevance from definitions
        self.stopword_ontology_exemptions = STOPWORD_ONTOLOGY_EXEMPTIONS
        self.ner_entity_type_relevance = NER_ENTITY_TYPE_RELEVANCE

        # Category store -- maps lemmas to semantic categories
        self.category_store = create_category_store(
            backend=cfg.get("category_store_backend", "sqlite"),
            fallback=cfg.get("category_store_fallback", "memory"),
            path=cfg.get("category_store_path"),
            seed=cfg.get("category_store_seed", True),
            kv_on_cache_miss=self.kv_on_cache_miss,
        )

        logger.info(
            f"AttributeEncoder initialized: max_attributes={self.max_attributes}, "
            f"ontology_aware={self.ontology_aware}, "
            f"ner_scoping_weight={self.ner_scoping_weight}, "
            f"category_store={type(self.category_store).__name__}({self.category_store.count()} entries)"
        )

    def get_metrics(self) -> Dict:
        base = super().get_metrics()
        base["category_store_size"] = self.category_store.count()
        return base

    def config_summary(self) -> Dict:
        return {
            "max_attributes": self.max_attributes,
            "ontology_aware": self.ontology_aware,
            "category_store_backend": type(self.category_store).__name__,
        }

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
    # Auto-classification
    # --------------------------------------------------
    def _infer_category(self, lemma: str, ontology_type: Optional[str]) -> str:
        """Infer a category for an unknown adjective WITHOUT persisting to the store.

        If ontology context is available, pick the highest-weighted category
        for that ontology type.  Otherwise return ``"uncategorized"``.
        """
        if ontology_type:
            category = get_best_category_for_ontology(ontology_type)
            if category:
                return category
        return "uncategorized"

    def _auto_classify(self, lemma: str, ontology_type: Optional[str]) -> str:
        """Classify an unknown adjective and persist it to the store.

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
        """Encode multiple texts, batching ALL category KV lookups across texts."""
        if not texts:
            return []
        n = len(texts)
        if contexts is None:
            contexts = [None] * n
        # A record without a parse is parsed here (rule parser, record_parse).
        contexts = [with_parse(texts[i], contexts[i], self.head_type) for i in range(n)]

        # Phase 1: Extract raw attributes for ALL texts (CPU-only)
        per_text_data = []  # (attrs, ontology_type, ner_labels)
        pred_idx: List[int] = []  # texts settled by the predication path
        for i in range(n):
            ctx = contexts[i]
            spacy_doc = ctx.get("spacy_doc")
            ontology_type = self._get_ontology_type(ctx)
            ner_labels = self._get_ner_entity_types(spacy_doc)

            attrs: List[tuple] = []
            if spacy_doc is not None and self.predication_mode:
                pred_idx.append(i)
            if spacy_doc is not None:
                attrs = self._extract_from_doc(spacy_doc)
            else:
                attrs = self._extract_regex(texts[i])

            if self.wordlist:
                wl_hits = self._wordlist_hits(texts[i])
                attrs.extend([(w, "wordlist", len(attrs) + j, w.lower()) for j, w in enumerate(wl_hits)])

            per_text_data.append((attrs, ontology_type, ner_labels))

        # Phase 2: Collect all unique (canon, ontology_type) queries
        all_queries = set()
        for i, (attrs, ontology_type, _) in enumerate(per_text_data):
            seen: Set[str] = set()
            for text_val, source, order, lemma in attrs:
                canon = self._canonicalize(text_val, lemma=lemma)
                if len(canon) < self.min_len or canon in seen:
                    continue
                seen.add(canon)
                all_queries.add((canon, ontology_type))

        # Phase 3: ONE batch get for all category lookups
        cat_cache = self.category_store.get_categories_weighted_batch(list(all_queries))

        # Phase 4: Score and select per text using cached categories
        results = []
        for i, (attrs, ontology_type, ner_labels) in enumerate(per_text_data):
            scored_attrs = self._score_and_select(
                attrs, ontology_type=ontology_type, ner_labels=ner_labels,
                cat_cache=cat_cache,
            )

            if not scored_attrs:
                addresses = [SemanticAddress(
                    head_type=self.head_type, concept="__null__",
                    confidence=1.0, metadata={"method": "null"},
                )]
            else:
                addresses = []
                for concept, score, source, category in scored_attrs:
                    meta = {"method": source}
                    if category:
                        meta["category"] = category
                    addresses.append(SemanticAddress(
                        head_type=self.head_type, concept=concept,
                        confidence=score, metadata=meta,
                    ))

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

        # Phase 5: predication values, merged with the legacy selection
        # (after it, so rows _auto_classify just wrote are read and kept).
        if pred_idx:
            preds = self._settle_predications(
                [texts[i] for i in pred_idx], [contexts[i] or {} for i in pred_idx], mode="ingest")
            for i, pr in zip(pred_idx, preds):
                ctx = contexts[i]
                results[i] = self._merge_with_legacy(results[i], pr, ctx.get("spacy_doc"), ctx.get("title") or "")
        return results

    # --------------------------------------------------
    # Predication mode
    # --------------------------------------------------
    def _merge_with_legacy(self, legacy: EncodingResult, pred: EncodingResult, doc,
                           title: str = "") -> EncodingResult:
        """One result: the legacy selection (its sources, its record-level cap)
        plus the predication values. A legacy value gains anchors from token
        adjacency (the noun phrase it modifies) unless it is junk by the
        structural rules, so the bundler can link it per entity; the
        dependency walk still links it either way. Order: confidence desc,
        legacy first on ties."""
        from .predication import legacy_anchors
        merged: Dict[str, SemanticAddress] = {}
        order: List[str] = []
        anchor_map = legacy_anchors(doc, [a.concept for a in legacy.addresses if a.concept != "__null__"],
                                    self._canonicalize, title=title)
        for a in legacy.addresses:
            if a.concept == "__null__":
                continue
            meta = dict(a.metadata or {})
            anchors = anchor_map.get(a.concept)
            if anchors:
                meta["anchors"] = anchors
            merged[a.concept] = SemanticAddress(self.head_type, a.concept, a.confidence, meta)
            order.append(a.concept)
        for a in pred.addresses:
            if a.concept == "__null__":
                continue
            prev = merged.get(a.concept)
            if prev is None:
                merged[a.concept] = a
                order.append(a.concept)
                continue
            pm = prev.metadata
            for an in a.metadata.get("anchors", []):
                pm.setdefault("anchors", [])
                if an not in pm["anchors"]:
                    pm["anchors"].append(an)
            for r in a.metadata.get("resolved", []):
                pm.setdefault("resolved", [])
                if r not in pm["resolved"]:
                    pm["resolved"].append(r)
            pm["kinds"] = sorted(set(pm.get("kinds", [])) | set(a.metadata.get("kinds", [])))
            if a.confidence > prev.confidence:
                prev.confidence = a.confidence
                pm["method"] = a.metadata.get("method", pm.get("method"))
        rank = {c: k for k, c in enumerate(order)}
        addrs = sorted(merged.values(), key=lambda a: (-a.confidence, rank[a.concept]))
        meta = dict(legacy.metadata or {})
        sk = resolved_sketch(doc)
        if sk is not None:
            self._bind_chains(addrs, doc, sk)
            meta["sketch"] = SKETCH_CONTRACT
        if not addrs:
            addrs = [SemanticAddress(self.head_type, "__null__", 1.0, {"method": "null"})]
        meta.update({"extraction": "legacy+predication", "count": len(addrs)})
        return EncodingResult(self.head_type, addrs, meta)

    @staticmethod
    def _bind_chains(addrs, doc, sk) -> None:
        """Each value's ``chains``: the resolved sketch's chain of every
        subject it was predicated of (its anchors: the subject's own span,
        sketch_slots.span_chains). The bundler gives an entity the values
        whose chains hold its chain; a value with none reaches no bundle.

        A pronoun or anaphor subject NLP2 leaves in no chain binds through
        the main-subject span the head's own resolution gave it
        (``resolved``), when there is one: NLP2 abstains, the head's
        title-subject rule decides, as it does without the sketch ("If you're
        wondering who X was, he was a ... economist": NLP2 leaves "he"
        unresolved; "The village's location": no mention for "the village").
        Where NLP2 did resolve the subject, its chain stands. Measured on the
        2,000-passage answer key (scripts/axes/head_gold_eval.py,
        2026-10-07): falling back also for subjects NLP2 put in a chain of
        common-noun mentions only raised recall 0.164 -> 0.167 and lowered
        precision 0.591 -> 0.574."""
        index = MentionIndex(sk)
        memo: Dict[Tuple[int, int], Tuple[int, ...]] = {}

        def chains_of(key):
            if key not in memo:
                memo[key] = span_chains(sk, doc, key, index)
            return memo[key]
        for a in addrs:
            meta = a.metadata if a.metadata is not None else {}
            via = {(int(r[0]), int(r[1])): (int(r[2]), int(r[3])) for r in meta.get("resolved") or ()}
            chains: List[int] = []
            for an in meta.get("anchors") or ():
                key = (int(an[0]), int(an[1]))
                got = chains_of(key)
                if not got and key in via:
                    got = chains_of(via[key])
                for c in got:
                    if c not in chains:
                        chains.append(c)
            meta["chains"] = chains
            a.metadata = meta

    # --------------------------------------------------
    def _sentence_ontology(self, context: Optional[Dict]):
        """sentence index -> ontology label (the sentence's, else the
        record's, else None)."""
        ont = (context or {}).get("ontology_result")
        meta = getattr(ont, "metadata", None) or {}
        sent = {int(k): v for k, v in (meta.get("sentence_ontologies") or {}).items()}
        record = self._get_ontology_type(context)
        if record == B.UNCLASSIFIED:
            record = None

        def label_of(si):
            lab = sent.get(si)
            return record if (not lab or lab == B.UNCLASSIFIED) else lab
        return label_of

    def _steer_label(self, context: Optional[Dict]):
        """sentence index -> the ontology that steers evidence read there:
        the sentence's own ontology (sentence_steering) else the record's.
        Ingest and query compute sentence ontologies the same way
        (encoders/sentence_ontology.py)."""
        label_of = self._sentence_ontology(context)
        if self.sentence_steering:
            return label_of
        record = self._get_ontology_type(context)
        record = None if record == B.UNCLASSIFIED else record
        return lambda si: record

    def _stopword_base(self, value: str, base: float, ontology) -> float:
        if value not in self.stopwords:
            return base
        pen = self.weights.get("stop_penalty", 0.3)
        exempt = ontology_lookup(self.stopword_ontology_exemptions, ontology, set()) or set()
        return max(0.05, base - (pen * 0.33 if value in exempt else pen))

    def _score_occurrence(self, value: str, link: str, kind: str, ontology,
                          lemma_category: Optional[str], ner_labels: Set[str]) -> float:
        """The legacy scoring, per reading: link weight (config weights),
        ontology-aware stopword penalty, ONTOLOGY_RELEVANCE for the evidence
        category under the steering ontology, NER relevance."""
        base = float(self.weights.get(link, P_LINK_WEIGHT.get(link, 0.5)))
        score = self._stopword_base(value, base, ontology)
        category = evidence_category(kind, lemma_category)
        if category:
            if ontology and self.ontology_aware:
                raw = get_ontology_relevance(ontology, category)
                score *= 1.0 + (raw - 1.0) * self.ontology_relevance_weight
            if ner_labels:
                score *= self._compute_ner_relevance(ner_labels, category)
        return max(0.05, min(1.0, score))

    def _wordlist_values(self, doc):
        """Wordlist hits as predication values anchored on their own token."""
        if not self.wordlist:
            return []
        from .predication import AttributeValue
        out = {}
        for tok in doc:
            w = tok.text.lower()
            if w in self.wordlist and len(w) >= self.min_len:
                av = out.get(w)
                if av is None:
                    av = out[w] = AttributeValue(w, "wordlist", P_LINK_WEIGHT.get("wordlist", 0.5))
                anchor = (tok.i, tok.i + 1)
                av.anchors.append(anchor)
                av.token_indices.append(tok.i)
                av.occurrences.append((anchor, "wordlist", "adj", 0))
                if "adj" not in av.kinds:
                    av.kinds.append("adj")
        return list(out.values())

    def _select_values(self, values, label_of, ner_labels):
        """Score every reading, then run the adaptive selection per subject
        and sentence -- one predication's reading of one subject. Returns [(value, score, link, anchors, kinds)] in emission
        order (score desc, first occurrence)."""
        groups: Dict[Tuple[int, int], list] = {}
        order = 0
        # Category rows read as the legacy path reads them: the
        # ontology-scoped row (acat::lemma::ontology, ONTOLOGY_CATEGORY_
        # OVERRIDES) first, then the global row -- one batched lookup.
        queries = sorted({(self._canonicalize(av.value), label_of(o[3]))
                          for av in values for o in av.occurrences}, key=str)
        cats = self.category_store.get_categories_weighted_batch(queries) if queries else {}
        for av in values:
            concept = self._canonicalize(av.value)
            for anchor, link, kind, si in av.occurrences:
                label = label_of(si)
                # Only a curated category (seed, confidence 1.0) weighs the
                # score. An inferred one is the best category of an ontology
                # the word was seen with, so weighing it by that ontology is
                # circular -- it made extraction drift with store state (cold
                # vs warm: 132 values dropped on wiki_sample_2000).
                cr = cats.get((concept, label))
                lemma_cat = cr[0] if cr and float(cr[1]) >= 1.0 else None
                if self.evidence_exclusions and \
                        evidence_category(kind, lemma_cat) in excluded_categories(label):
                    continue
                sc = self._score_occurrence(concept, link, kind, label, lemma_cat, ner_labels)
                groups.setdefault((tuple(anchor), si), []).append((concept, sc, link, order, kind))
                order += 1
        chosen: Dict[str, list] = {}
        for gkey in sorted(groups):
            anchor = gkey[0]
            best = {}
            for item in groups[gkey]:   # one reading per value per subject: the best
                if item[0] not in best or item[1] > best[item[0]][1]:
                    best[item[0]] = item
            items = sorted(best.values(), key=lambda x: (-x[1], x[3]))
            for concept, sc, link, o, kind in self._adaptive_select(items, self.max_attributes_per_subject,
                                                                    keep_one_weak=False):
                c = chosen.get(concept)
                if c is None:
                    chosen[concept] = c = [concept, sc, link, [], [], o]
                elif sc > c[1]:
                    c[1], c[2] = sc, link
                c[5] = min(c[5], o)
                if list(anchor) not in c[3]:
                    c[3].append(list(anchor))
                for k2 in [it[4] for it in groups[gkey] if it[0] == concept]:
                    if k2 not in c[4]:
                        c[4].append(k2)
        return [tuple(c[:5]) for c in sorted(chosen.values(), key=lambda c: (-c[1], c[5]))]

    def _settle_predications(self, texts: List[str], contexts: List[Dict], mode: str):
        """Predication-mode results for texts that have a parsed doc.

        One batched store read for every candidate value. At ingest every
        kept word (before the ontology steer) is observed and merged into
        its store row (learning.merge); changed rows are written once, at
        the end, through put_doc -- the deferred-KV + LMDB-relay path. The
        store's role flag (seeded from person_nouns.txt, grown by learning)
        is what extraction asks when it needs to know whether a noun names
        a kind of person. Queries read the same rows and never write.
        """
        from .definitions import entity_class
        min_len = self.min_len
        learn = mode == "ingest"
        is_query = mode == "query"
        docs = [(c or {}).get("spacy_doc") for c in contexts]
        # The record title (ctx["title"], ingest only): its mention is the
        # main subject that pronouns and kind-noun anaphors resolve to.
        titles = [((c or {}).get("title") or "") for c in contexts]
        keys = sorted({self._canonicalize(v) for d, t in zip(docs, titles) if d is not None
                       for v in candidate_values(d, query=is_query, title=t)})
        rows = dict(self.category_store.get_docs_batch(keys)) if keys else {}
        original = dict(rows)
        results = []
        for text, ctx, doc, title in zip(texts, contexts, docs, titles):
            ctx = ctx or {}
            label_of = self._sentence_ontology(ctx)
            steer_of = self._steer_label(ctx)
            person_tok = {t.i for e in (getattr(doc, "ents", None) or []) if e.label_ == "PERSON" for t in e}
            fp = L.fingerprint(text)
            obs: Dict[str, L.Observation] = {}

            def is_person(v, _rows=rows):
                return L.is_role(_rows.get(self._canonicalize(v)))

            def is_kind(v, _rows=rows):
                return L.kind_status(_rows.get(self._canonicalize(v)))

            def observe(value, kind, p, w, _obs=obs):
                o = _obs.get(self._canonicalize(value))
                if o is None:
                    o = _obs[self._canonicalize(value)] = L.Observation()
                o.kinds.add(L.structural_kind(kind))
                label = label_of(p.sentence)
                cls = entity_class(label)
                if label:
                    o.ontologies.add(str(label))
                    o.classes.add(cls)
                followed_by_of = w.i + 1 < len(doc) and doc[w.i + 1].text.lower() == "of"
                if L.person_evidence(p.link, w.role, kind, cls,
                                     any(k in person_tok for k in range(*p.anchor)),
                                     p.person_cue, followed_by_of):
                    o.support.add(fp)
                relational = w.i + 1 < len(doc) and doc[w.i + 1].text.lower() in L.RELATIONAL_PREPS
                if L.kind_evidence(p.link, w.role, kind, p.named_subject, p.of_complement, relational):
                    o.kind_support.add(fp)

            values = attribute_values(doc, query=is_query, is_person=is_person,
                                      observe=observe if learn else None, min_len=min_len,
                                      skip_links=("modifier",), title=title, is_kind=is_kind,
                                      sketch_subjects=resolved_sketch(doc) is not None)
            values += self._wordlist_values(doc)
            if learn:
                for key, o in obs.items():
                    rows[key] = L.merge(rows.get(key), o)
            ner_labels = self._get_ner_entity_types(doc)
            selected = self._select_values(values, steer_of, ner_labels)[: self.max_predicated_per_record]
            resolved: Dict[str, Dict] = {}
            for av in values:
                if av.resolved:
                    resolved.setdefault(self._canonicalize(av.value), {}).update(av.resolved)
            results.append(self._predication_result(
                selected, self._get_ontology_type(ctx), ner_labels, rows, mode, resolved))
        if learn:
            for key in sorted(rows):
                if rows[key] is not None and rows[key] != original.get(key):
                    self.category_store.put_doc(key, rows[key])
        return results

    def _predication_result(self, selected, ontology_type, ner_labels, rows, mode, resolved=None):
        """EncodingResult for selected predication values, each with the
        subject spans (anchors) it was selected for and its evidence kinds;
        ``resolved`` (sketch only): concept -> {own subject span: the
        main-subject span the head's resolution gives it}, kept as
        ``resolved`` [own start, own end, main start, main end] for
        _bind_chains."""
        addresses = []
        for concept, score, link, anchors, kinds in selected:
            meta = {"method": link, "anchors": anchors, "kinds": kinds}
            res = (resolved or {}).get(concept) or {}
            rs = [[a[0], a[1], res[tuple(a)][0], res[tuple(a)][1]] for a in anchors if tuple(a) in res]
            if rs:
                meta["resolved"] = rs
            category = (rows.get(concept) or {}).get("category")
            if category and category != "uncategorized":
                meta["category"] = category
            addresses.append(SemanticAddress(
                head_type=self.head_type, concept=concept, confidence=score, metadata=meta))
        if not addresses:
            addresses = [SemanticAddress(
                head_type=self.head_type, concept="__null__",
                confidence=1.0, metadata={"method": "null"},
            )]
        metadata = {"mode": mode, "extraction": "predication", "count": len(addresses)}
        if ontology_type:
            metadata["ontology_type"] = ontology_type
        if ner_labels:
            metadata["ner_labels"] = sorted(ner_labels)
        return EncodingResult(self.head_type, addresses, metadata)

    # --------------------------------------------------
    # Public encode APIs
    # --------------------------------------------------
    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        context = with_parse(text, context, self.head_type)
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        ner_labels = self._get_ner_entity_types(spacy_doc)
        addresses = self._extract_attributes(
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
        result = EncodingResult(self.head_type, addresses, metadata)
        if spacy_doc is not None and self.predication_mode:
            pred = self._settle_predications([text], [context], mode="ingest")[0]
            result = self._merge_with_legacy(result, pred, spacy_doc, (context or {}).get("title") or "")
        return result

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        context = with_parse(query, context, self.head_type)
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        ner_labels = self._get_ner_entity_types(spacy_doc)
        addresses = self._extract_attributes(
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
        result = EncodingResult(self.head_type, addresses, metadata)
        if spacy_doc is not None and self.predication_mode:
            pred = self._settle_predications([query], [context], mode="query")[0]
            result = self._merge_with_legacy(result, pred, spacy_doc)
        return result

    # --------------------------------------------------
    # Attribute extraction
    # --------------------------------------------------
    def _extract_attributes(
        self,
        text: str,
        spacy_doc=None,
        ontology_type: Optional[str] = None,
        ner_labels: Optional[Set[str]] = None,
    ) -> List[SemanticAddress]:
        attrs: List[tuple] = []  # (text, source, order, lemma)

        if spacy_doc is not None:
            attrs = self._extract_from_doc(spacy_doc)
        else:
            attrs = self._extract_regex(text)

        # Merge with wordlist matches (if any)
        if self.wordlist:
            wl_hits = self._wordlist_hits(text)
            attrs.extend([(w, "wordlist", len(attrs) + i, w.lower()) for i, w in enumerate(wl_hits)])

        scored_attrs = self._score_and_select(
            attrs, ontology_type=ontology_type, ner_labels=ner_labels,
        )

        if not scored_attrs:
            return [
                SemanticAddress(
                    head_type=self.head_type,
                    concept="__null__",
                    confidence=1.0,
                    metadata={"method": "null"}
                )
            ]

        addresses = []
        for concept, score, source, category in scored_attrs:
            meta = {"method": source}
            if category:
                meta["category"] = category
            addresses.append(
                SemanticAddress(
                    head_type=self.head_type,
                    concept=concept,
                    confidence=score,
                    metadata=meta,
                )
            )
        return addresses

    def _extract_from_doc(self, doc) -> List[tuple]:
        """Extract attributes from the record's parse.

        Returns list of (text, source, order, lemma) tuples.
        Includes single-token and multi-token compound phrases.
        """
        attrs: List[tuple] = []

        for i, token in enumerate(doc):
            lemma = token.lemma_.lower()

            # Simple ADJ
            if token.pos_ == "ADJ":
                attrs.append((token.text, "adj", i, lemma))

            # Predicate adjective via copula (attr/acomp)
            if token.dep_ in {"acomp", "attr"} and token.pos_ == "ADJ":
                attrs.append((token.text, "predicate", i, lemma))

            # amod modifiers on nouns (collect modifier)
            if token.dep_ == "amod" and token.head.pos_ in {"NOUN", "PROPN"}:
                attrs.append((token.text, "amod", i, lemma))

            # Participial adjectives: past participle used as modifier
            if token.tag_ == "VBN" and token.dep_ == "amod":
                attrs.append((token.text, "participial", i, lemma))

            # Compound modifiers: hyphenated tokens acting as amod
            if token.dep_ == "amod" and "-" in token.text:
                attrs.append((token.text, "compound_mod", i, token.text.lower()))

            # NOUN compound modifier: "stone" in "stone bridge"
            if token.pos_ == "NOUN" and token.dep_ == "compound":
                attrs.append((token.text, "noun_compound", i, lemma))

            # Appositive: "president" in "Lincoln, the president"
            if token.pos_ in {"NOUN", "PROPN"} and token.dep_ == "appos":
                attrs.append((token.text, "appositive", i, lemma))

            # Participial acl: "destroyed" in "the city destroyed by war"
            if token.tag_ == "VBN" and token.dep_ == "acl":
                attrs.append((token.text, "acl_participial", i, lemma))

        # Second pass: multi-token compound adjective phrases
        for token in doc:
            if token.pos_ != "ADJ":
                continue
            if token.dep_ not in {"amod", "acomp", "attr"}:
                continue

            for child in token.children:
                # advmod + ADJ: "well known", "deeply rooted"
                if child.dep_ == "advmod" and child.pos_ == "ADV":
                    phrase = f"{child.text} {token.text}".lower()
                    lemma_phrase = f"{child.lemma_} {token.lemma_}".lower()
                    attrs.append((phrase, "compound_phrase", token.i, lemma_phrase))

                # ADJ + ADJ compounds: "blood red", "ice cold"
                if child.pos_ == "ADJ" and child.dep_ in {"amod", "compound"}:
                    phrase = f"{child.text} {token.text}".lower()
                    lemma_phrase = f"{child.lemma_} {token.lemma_}".lower()
                    attrs.append((phrase, "compound_phrase", token.i, lemma_phrase))

        return attrs

    def _extract_regex(self, text: str) -> List[tuple]:
        candidates = re.findall(r"\b[A-Za-z]{3,}\b", text)
        suffixes = ("ive", "ous", "ful", "less", "able", "ible", "ic", "al", "y", "ish")
        attrs = [
            (w, "regex", idx, w.lower())
            for idx, w in enumerate(candidates)
            if any(w.lower().endswith(s) for s in suffixes)
        ]
        return attrs

    def _wordlist_hits(self, text: str) -> List[str]:
        if isinstance(self.wordlist, set):
            hits = []
            for tok in re.findall(r"\b\w+\b", text.lower()):
                if tok in self.wordlist:
                    hits.append(tok)
            return hits
        return []

    def _canonicalize(self, term: str, lemma: Optional[str] = None) -> str:
        """Canonicalize an attribute term.

        Prefers *lemma* (from the parse) over raw *term*, then applies synonym folding.
        """
        t = (lemma or term).strip().lower()
        t = self.synonyms.get(t, t)
        return t

    def _score_and_select(
        self,
        attrs: List[tuple],
        ontology_type: Optional[str] = None,
        ner_labels: Optional[Set[str]] = None,
        cat_cache: Optional[Dict] = None,
    ) -> List[Tuple[str, float, str, Optional[str]]]:
        """Score and select top attributes.

        Ontology relevance and NER scoping are integrated into scoring BEFORE
        adaptive selection so they influence which attributes survive the cut.

        Args:
            attrs: list of (text, source, order, lemma)
            ontology_type: optional ontology concept for relevance weighting
            ner_labels: optional set of the parse's NER labels for NER scoping
            cat_cache: optional pre-fetched category cache from batch lookup

        Returns:
            list of (concept, score, source, category) sorted by score desc, order asc
        """
        seen: Set[str] = set()
        scored = []
        w = self.weights
        stop_pen = w.get("stop_penalty", 0.3)

        for text, source, order, lemma in attrs:
            canon = self._canonicalize(text, lemma=lemma)
            if len(canon) < self.min_len:
                continue
            if canon in seen:
                continue
            seen.add(canon)

            base = w.get(source, 0.5)

            # Ontology-aware stopword modulation
            if canon in self.stopwords:
                if (
                    ontology_type
                    and ontology_type in self.stopword_ontology_exemptions
                    and canon in self.stopword_ontology_exemptions[ontology_type]
                ):
                    # Reduced penalty for ontology-exempt stopwords
                    base = max(0.05, base - stop_pen * 0.33)
                else:
                    base = max(0.05, base - stop_pen)

            # Get category from store (or pre-fetched cache), scoped by ontology type
            if cat_cache is not None:
                cat_result = cat_cache.get((canon, ontology_type))
            else:
                cat_result = self.category_store.get_category_weighted(canon, ontology_type)
            if cat_result:
                category, _confidence = cat_result
                inferred = False
            else:
                category = self._infer_category(canon, ontology_type)
                inferred = True

            # Apply ontology relevance BEFORE selection
            score = base
            if ontology_type and self.ontology_aware and category != "uncategorized":
                raw_relevance = get_ontology_relevance(ontology_type, category)
                relevance = 1.0 + (raw_relevance - 1.0) * self.ontology_relevance_weight
                score *= relevance

            # Apply NER relevance BEFORE selection
            if ner_labels and category != "uncategorized":
                ner_mult = self._compute_ner_relevance(ner_labels, category)
                score *= ner_mult

            score = max(0.05, min(1.0, score))
            scored.append((canon, score, source, order, category, inferred))

        # Sort by score desc, then order for determinism
        scored.sort(key=lambda x: (-x[1], x[3]))

        selected = self._adaptive_select(scored, self.max_attributes)

        # Persist auto-classification only for selected attributes
        for c, _s, _src, _ord, _cat, inf in selected:
            if inf:
                self._auto_classify(c, ontology_type)

        return [
            (c, s, src, cat if cat != "uncategorized" else None)
            for c, s, src, _, cat, _ in selected
        ]

    def _adaptive_select(self, scored: list, cap: int, keep_one_weak: bool = True) -> list:
        """Adaptive selection on COMPOSITE scores (ontology+NER integrated).

        ``scored``: tuples (concept, score, source, order, ...) sorted by
        score desc, order asc. Absolute floor, top cluster, elbow cut, hard
        cap. Used per record (legacy) and per subject (predication mode).
        """
        abs_min = self.weights.get("absolute_min_confidence", 0.2)
        soft_min = self.weights.get("soft_min_confidence", 0.05)
        elbow_gap = self.weights.get("elbow_gap", 0.15)
        cluster_tol = self.weights.get("cluster_tolerance", 0.05)

        candidates = [x for x in scored if x[1] >= abs_min]

        if not candidates:
            # All weak; keep best one if above soft_min. Record level only:
            # per subject (predication mode) a reading below the floor is one
            # ONTOLOGY_RELEVANCE rejected for that ontology, and keeping it
            # would undo the steer.
            if keep_one_weak and scored and scored[0][1] >= soft_min:
                candidates = [scored[0]]
            else:
                return []

        # Top cluster within tolerance of top score
        top_score = candidates[0][1]
        cluster = [x for x in candidates if (top_score - x[1]) <= cluster_tol]

        # Elbow cut: find largest gap
        gap_cut = len(candidates)
        gaps = []
        for i in range(len(candidates) - 1):
            gaps.append((candidates[i][1] - candidates[i+1][1], i))
        if gaps:
            max_gap, idx = max(gaps, key=lambda x: x[0])
            if max_gap > elbow_gap:
                gap_cut = idx + 1

        selected = candidates[:gap_cut]

        # Union with cluster (to keep tight cluster near top)
        selected_ids = set(id(x) for x in selected)
        for x in cluster:
            if id(x) not in selected_ids:
                selected.append(x)

        # Re-sort and cap hard max
        selected.sort(key=lambda x: (-x[1], x[3]))
        return selected[:cap]

    def _load_wordlist(self, wordlist_cfg):
        if not wordlist_cfg:
            return None
        if isinstance(wordlist_cfg, (list, tuple, set)):
            return set(w.strip().lower() for w in wordlist_cfg if w.strip())
        path = None
        try:
            path = Path(wordlist_cfg)
            if path.exists():
                if path.suffix.lower() in {".txt"}:
                    with open(path, "r") as f:
                        return set(line.strip().lower() for line in f if line.strip())
                elif path.suffix.lower() in {".json"}:
                    import json
                    with open(path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        return set(w.strip().lower() for w in data if isinstance(w, str))
        except Exception as e:
            logger.warning(f"Failed to load wordlist {wordlist_cfg}: {e}")
        return None
