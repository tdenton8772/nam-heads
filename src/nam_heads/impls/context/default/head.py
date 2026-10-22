"""
Context encoder head.

Answers: "Under what circumstances does this memory apply?"

Two extractions, one head (one EncoderHead subclass, as discover_encoders
requires):

* ``particular`` (default) -- the particular-evidence extractor
  (``src/context_axis``): temporal paths ``t:C19/D1860/Y1865``, gazetteer
  places ``g:NA/US/US-IL/...``, institutions ``i:<org>``, fields ``d:<field>``,
  each with the sentences it came from (the bundler scopes values to
  entities through the profile the entity's own evidence gives it --
  context_axis/profiles.py; the record's class never removes a dimension),
  and each named place's container. Reads and grows the context
  store (``self.store``, relayed to LMDB by nats_worker run_head). It never
  derives a value from the record's type (DOMAIN_SEEDS) or a default
  (INSTITUTIONAL_DEFAULT 'commercial').
* ``legacy`` -- the NER-driven extraction below: 4 fixed dimensions
  normalized via store-backed lookups, 1 address per dimension with signal.
  Kept for rollback: ``NAM_CONTEXT_HEAD=legacy`` (or config ``head``).

Both read the record's parse (``context["spacy_doc"]``, the rule backend's
ParsedDocument); a record without one is parsed with the same rule parser
(``src.encoders.record_parse``).
"""

import logging
import re
from typing import Dict, Optional, List, Set

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from .config import DEFAULT_CONFIG
from .definitions import (
    DIMENSIONS,
    DOMAIN_NER_OVERRIDES,
    DOMAIN_ORG_KEYWORDS,
    DOMAIN_SEEDS,
    INSTITUTIONAL_KEYWORDS,
    INSTITUTIONAL_DEFAULT,
    normalize_temporal_rules,
    ONTOLOGY_KEYS,
)
from nam_heads.ontology import bindings as B
from nam_heads.ontology.registry import get_registry
from .normalizers import (
    normalize_temporal,
    normalize_geographic,
    normalize_domain,
    normalize_institutional,
)
from .storage.factory import create_context_store
from src.encoders.record_parse import with_parse

# What this default plugin declares (docs/design-head-plugins.md section 3), read with ast
# (nam_heads.sdk.impl_meta.read_module_literal): literals only. "resources" are the per-container
# limits and the containers per ingest pod it runs with on the reference deployment:
# the chart's ingest.axisHeadResources.limits (cpu "1", memory 1Gi) and
# ingest.contextWorkers (1).
IMPL_META = {
    "slot": "context",
    "name": "default",
    "version": "1.0.1",
    "contract": "context-v1",
    "description": "Context axis: the circumstances a record applies under (time, place, "
                   "institution, field) from its particular evidence, or the legacy NER extraction.",
    "output": "context-values-v1",
    "resources": {"cpu": 1.0, "memory_mb": 1024, "workers": 1},
    "dependencies": [],
}

logger = logging.getLogger(__name__)

import os


class ContextEncoder(EncoderHead):
    HEAD_TYPE = "context"

    def create_axis_encoder(self):
        """This head's own encoder, from its own package (``encoder/charcnn``)."""
        from pathlib import Path as _Path
        from src.encoders.axis_encoder import package_axis_encoder
        return package_axis_encoder(self.head_type, _Path(__file__).resolve().parent / "encoder" / "charcnn")

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **(self.config or {})}
        mode = str(cfg.get("head") or os.environ.get("NAM_CONTEXT_HEAD", "particular")).strip().lower()
        if mode not in ("particular", "legacy"):
            raise ValueError(f"NAM_CONTEXT_HEAD must be 'particular' or 'legacy', got {mode!r}")
        self.head_mode = mode
        self._impl = None
        if mode == "particular":
            self._init_particular()
            return
        self.default_null = cfg.get("default_null", "__null__")
        self.dimensions = cfg.get("dimensions", DIMENSIONS)
        self.max_per_dimension = int(cfg.get("max_per_dimension", 1))
        self.split_fragments = bool(cfg.get("split_fragments", True))

        # Context store
        self.store = create_context_store(
            backend=cfg.get("context_store_backend", "sqlite"),
            fallback=cfg.get("context_store_fallback", "memory"),
            path=cfg.get("context_store_path"),
            seed=cfg.get("context_store_seed", True),
            kv_on_cache_miss=self.kv_on_cache_miss,
        )

        logger.info(
            f"ContextEncoder initialized: head=legacy dimensions={self.dimensions}, "
            f"store={type(self.store).__name__}({self.store.count()} entries)"
        )

    def _init_particular(self):
        """Delegate to the particular-evidence extractor. Imported here, not
        at module level: discover_encoders requires exactly one EncoderHead
        subclass in this module's namespace."""
        from .context_axis.encoder import ParticularContextEncoder
        cfg = {k: v for k, v in (self.config or {}).items() if k != "head"}
        cfg.setdefault("kv_on_cache_miss", self.kv_on_cache_miss)
        self._impl = ParticularContextEncoder(head_type=self.head_type, config=cfg)
        self.store = self._impl.store
        self.default_null = self._impl.default_null
        self.dimensions = list(self._impl.dimensions)
        self.max_per_dimension = self._impl.max_per_dimension
        self.split_fragments = self._impl.split_fragments
        logger.info("ContextEncoder initialized: head=particular store=%s",
                    type(self.store).__name__)

    def get_metrics(self) -> Dict:
        base = super().get_metrics()
        if self._impl is not None:
            extra = self._impl.get_metrics()
            for k in ("records_processed", "errors", "avg_encode_ms", "head_type"):
                extra.pop(k, None)
            base.update(extra)
            base["context_head"] = self.head_mode
            return base
        base["context_store_size"] = self.store.count()
        base["context_head"] = self.head_mode
        return base

    def config_summary(self) -> Dict:
        if self._impl is not None:
            return {"head": self.head_mode, **self._impl.config_summary()}
        return {
            "head": self.head_mode,
            "dimensions": list(self.dimensions),
            "max_per_dimension": self.max_per_dimension,
            "context_store_backend": type(self.store).__name__,
        }

    # --------------------------------------------------
    # Context from other heads
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

    def _get_entity_result(self, context: Optional[Dict]):
        """Extract entity result from context."""
        if not context:
            return None
        return context.get("entity_result")

    # --------------------------------------------------
    # NER extraction helpers
    # --------------------------------------------------
    def _get_ner_entities(self, spacy_doc) -> Dict[str, List[str]]:
        """Extract NER entities grouped by label from the parse's .ents."""
        grouped: Dict[str, List[str]] = {}
        if spacy_doc is None:
            return grouped
        for ent in spacy_doc.ents:
            grouped.setdefault(ent.label_, []).append(ent.text)
        return grouped

    def _get_ner_labels(self, ner_entities: Dict[str, List[str]]) -> Set[str]:
        """Get the set of NER labels present."""
        return set(ner_entities.keys())

    # --------------------------------------------------
    # Fragment splitting
    # --------------------------------------------------
    def _fragments(self, text: str, spacy_doc=None) -> List[str]:
        if not self.split_fragments:
            return [text]
        if spacy_doc is not None:
            return [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]
        parts = re.split(r"[.!?;\n]+", text)
        return [p.strip() for p in parts if p.strip()]

    # --------------------------------------------------
    # Cross-text batch encode -- called by MultiHeadEncoder.encode_batch
    # --------------------------------------------------
    def encode_batch_texts(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[Dict]]] = None,
    ) -> List[EncodingResult]:
        """Encode multiple texts, batching ALL store lookups across texts."""
        if not texts:
            return []
        if self._impl is not None:
            return self._impl.encode_batch_texts(texts, contexts)
        n = len(texts)
        if contexts is None:
            contexts = [None] * n

        # Phase 1: Extract NER entities + ontology types for ALL texts (CPU-only)
        per_text_data = []  # (ner_entities, ner_labels, ontology_type, spacy_doc)
        all_store_queries: Set[tuple] = set()  # (phrase, dimension)

        for i in range(n):
            ctx = with_parse(texts[i], contexts[i], self.head_type)
            doc = ctx.get("spacy_doc")
            ontology_type = self._get_ontology_type(ctx)

            ner_entities = self._get_ner_entities(doc)
            ner_labels = self._get_ner_labels(ner_entities)
            per_text_data.append((ner_entities, ner_labels, ontology_type, doc))

            # Collect store query keys for pre-fetch
            for label in ("DATE", "TIME", "EVENT"):
                for ent_text in ner_entities.get(label, []):
                    all_store_queries.add((ent_text.lower().strip(), "temporal"))
            for label in ("GPE", "LOC", "FAC"):
                for ent_text in ner_entities.get(label, []):
                    all_store_queries.add((ent_text.lower().strip(), "geographic"))
            if ontology_type:
                all_store_queries.add((f"domain::{ontology_type}", "domain"))
            for ent_text in ner_entities.get("ORG", []):
                all_store_queries.add((ent_text.lower().strip(), "institutional"))

        # Phase 2: ONE batch_get for all store queries
        store_cache = self.store.get_batch(list(all_store_queries)) if all_store_queries else {}

        # Phase 3: Process each text using cached store results
        results = []
        for i, (ner_entities, ner_labels, ontology_type, doc) in enumerate(per_text_data):
            addresses = self._extract_context_cached(
                ner_entities, ner_labels, ontology_type, store_cache,
            )
            metadata = {
                "mode": "ingest",
                "dimensions": len([a for a in addresses if a.concept != self.default_null]),
                "address_count": len(addresses),
            }
            results.append(EncodingResult(self.head_type, addresses, metadata))

        return results

    def _extract_context_cached(
        self,
        ner_entities: Dict[str, List[str]],
        ner_labels: Set[str],
        ontology_type: Optional[str],
        store_cache: Dict,
    ) -> List[SemanticAddress]:
        """Extract context using pre-fetched store cache (no individual KV calls)."""
        dim_results: Dict[str, List[SemanticAddress]] = {d: [] for d in self.dimensions}

        # Temporal
        if "temporal" in dim_results:
            for label in ("DATE", "TIME", "EVENT"):
                for ent_text in ner_entities.get(label, []):
                    key = ent_text.lower().strip()
                    cached = store_cache.get((key, "temporal"))
                    if cached is not None:
                        canonical = cached
                    else:
                        canonical = normalize_temporal_rules(ent_text)
                        if canonical:
                            self.store.put(key, "temporal", canonical, confidence=0.9)
                    if canonical:
                        dim_results["temporal"].append(SemanticAddress(
                            head_type=self.head_type, concept=canonical,
                            confidence=0.85,
                            metadata={"dimension": "temporal", "raw": ent_text,
                                      "source": f"ner_{label.lower()}"},
                        ))

        # Geographic
        if "geographic" in dim_results:
            for label in ("GPE", "LOC", "FAC"):
                for ent_text in ner_entities.get(label, []):
                    key = ent_text.lower().strip()
                    cached = store_cache.get((key, "geographic"))
                    if cached is not None:
                        dim_results["geographic"].append(SemanticAddress(
                            head_type=self.head_type, concept=cached,
                            confidence=0.85,
                            metadata={"dimension": "geographic", "raw": ent_text,
                                      "source": f"ner_{label.lower()}"},
                        ))

        # Domain
        if "domain" in dim_results:
            domain = None
            for ner_label, d in DOMAIN_NER_OVERRIDES.items():
                if ner_label in ner_labels:
                    domain = d
                    break
            if domain is None and ontology_type == B.ORGANIZATION and "ORG" in ner_labels:
                # ORG keyword refinement would need text, skip for batch
                pass
            if domain is None and ontology_type:
                cached = store_cache.get((f"domain::{ontology_type}", "domain"))
                if cached is not None:
                    domain = cached
                elif ontology_type in DOMAIN_SEEDS:
                    domain = DOMAIN_SEEDS[ontology_type]
            if domain:
                dim_results["domain"].append(SemanticAddress(
                    head_type=self.head_type, concept=domain,
                    confidence=0.80,
                    metadata={"dimension": "domain", "raw": ontology_type or "",
                              "source": "ontology_ner"},
                ))

        # Institutional
        if "institutional" in dim_results:
            for ent_text in ner_entities.get("ORG", []):
                key = ent_text.lower().strip()
                cached = store_cache.get((key, "institutional"))
                if cached is not None:
                    dim_results["institutional"].append(SemanticAddress(
                        head_type=self.head_type, concept=cached,
                        confidence=0.80,
                        metadata={"dimension": "institutional", "raw": ent_text,
                                  "source": "ner_org"},
                    ))
                else:
                    inst_type = None
                    for it, keywords_str in INSTITUTIONAL_KEYWORDS.items():
                        for keyword in keywords_str.split():
                            if keyword in key:
                                inst_type = it
                                break
                        if inst_type:
                            break
                    if inst_type is None:
                        inst_type = INSTITUTIONAL_DEFAULT
                    self.store.put(key, "institutional", inst_type, confidence=0.85 if inst_type != INSTITUTIONAL_DEFAULT else 0.5)
                    dim_results["institutional"].append(SemanticAddress(
                        head_type=self.head_type, concept=inst_type,
                        confidence=0.80,
                        metadata={"dimension": "institutional", "raw": ent_text,
                                  "source": "ner_org"},
                    ))

        # Deduplicate and cap per dimension
        addresses: List[SemanticAddress] = []
        for dim in self.dimensions:
            results_dim = dim_results.get(dim, [])
            seen: Dict[str, SemanticAddress] = {}
            for addr in results_dim:
                existing = seen.get(addr.concept)
                if existing is None or addr.confidence > existing.confidence:
                    seen[addr.concept] = addr
            deduped = sorted(seen.values(), key=lambda a: -a.confidence)
            deduped = deduped[: self.max_per_dimension]
            if deduped:
                addresses.extend(deduped)
            else:
                addresses.append(SemanticAddress(
                    head_type=self.head_type, concept=self.default_null,
                    confidence=1.0,
                    metadata={"dimension": dim, "raw": "", "source": "null"},
                ))

        return addresses

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        if self._impl is not None:
            return self._impl.encode(text, context)
        context = with_parse(text, context, self.head_type)
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        addresses = self._extract_context(text, spacy_doc, ontology_type)
        metadata = {
            "mode": "ingest",
            "dimensions": len([a for a in addresses if a.concept != self.default_null]),
            "address_count": len(addresses),
        }
        return EncodingResult(self.head_type, addresses, metadata)

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        if self._impl is not None:
            return self._impl.encode_query(query, context)
        context = with_parse(query, context, self.head_type)
        spacy_doc = context.get("spacy_doc")
        ontology_type = self._get_ontology_type(context)
        addresses = self._extract_context(query, spacy_doc, ontology_type)
        metadata = {
            "mode": "query",
            "dimensions": len([a for a in addresses if a.concept != self.default_null]),
            "address_count": len(addresses),
        }
        return EncodingResult(self.head_type, addresses, metadata)

    # --------------------------------------------------
    # Core extraction
    # --------------------------------------------------
    def _extract_context(
        self,
        text: str,
        spacy_doc,
        ontology_type: Optional[str],
    ) -> List[SemanticAddress]:
        doc = spacy_doc

        ner_entities = self._get_ner_entities(doc)
        ner_labels = self._get_ner_labels(ner_entities)

        # Collect results per dimension
        dim_results: Dict[str, List[SemanticAddress]] = {d: [] for d in self.dimensions}

        # Temporal: DATE, TIME, EVENT entities
        if "temporal" in dim_results:
            for label in ("DATE", "TIME", "EVENT"):
                for ent_text in ner_entities.get(label, []):
                    canonical = normalize_temporal(ent_text, self.store)
                    if canonical:
                        dim_results["temporal"].append(
                            SemanticAddress(
                                head_type=self.head_type,
                                concept=canonical,
                                confidence=0.85,
                                metadata={
                                    "dimension": "temporal",
                                    "raw": ent_text,
                                    "source": f"ner_{label.lower()}",
                                },
                            )
                        )

        # Geographic: GPE, LOC, FAC entities
        if "geographic" in dim_results:
            for label in ("GPE", "LOC", "FAC"):
                for ent_text in ner_entities.get(label, []):
                    canonical = normalize_geographic(ent_text, self.store)
                    if canonical:
                        dim_results["geographic"].append(
                            SemanticAddress(
                                head_type=self.head_type,
                                concept=canonical,
                                confidence=0.85,
                                metadata={
                                    "dimension": "geographic",
                                    "raw": ent_text,
                                    "source": f"ner_{label.lower()}",
                                },
                            )
                        )

        # Domain: ontology_type + NER labels
        if "domain" in dim_results:
            domain = normalize_domain(ontology_type, ner_labels, text, self.store)
            if domain:
                dim_results["domain"].append(
                    SemanticAddress(
                        head_type=self.head_type,
                        concept=domain,
                        confidence=0.80,
                        metadata={
                            "dimension": "domain",
                            "raw": ontology_type or "",
                            "source": "ontology_ner",
                        },
                    )
                )

        # Institutional: ORG entities
        if "institutional" in dim_results:
            for ent_text in ner_entities.get("ORG", []):
                canonical = normalize_institutional(ent_text, self.store)
                if canonical:
                    dim_results["institutional"].append(
                        SemanticAddress(
                            head_type=self.head_type,
                            concept=canonical,
                            confidence=0.80,
                            metadata={
                                "dimension": "institutional",
                                "raw": ent_text,
                                "source": "ner_org",
                            },
                        )
                    )

        # Keyword fallback: if NER/ontology produced no domain, scan tokens
        if "domain" in dim_results and not dim_results["domain"]:
            kw_domain = self._keyword_domain_fallback(text, doc)
            if kw_domain:
                dim_results["domain"].append(
                    SemanticAddress(
                        head_type=self.head_type,
                        concept=kw_domain,
                        confidence=0.65,
                        metadata={
                            "dimension": "domain",
                            "raw": "",
                            "source": "keyword_fallback",
                        },
                    )
                )

        # Keyword fallback: if NER produced no temporal signal, scan for era keywords
        if "temporal" in dim_results and not dim_results["temporal"]:
            kw_temporal = self._keyword_temporal_fallback(text, doc)
            if kw_temporal:
                dim_results["temporal"].append(
                    SemanticAddress(
                        head_type=self.head_type,
                        concept=kw_temporal,
                        confidence=0.60,
                        metadata={
                            "dimension": "temporal",
                            "raw": "",
                            "source": "keyword_fallback",
                        },
                    )
                )

        # Deduplicate within each dimension and cap
        addresses: List[SemanticAddress] = []
        for dim in self.dimensions:
            results = dim_results.get(dim, [])
            # Deduplicate by concept, keep highest confidence
            seen: Dict[str, SemanticAddress] = {}
            for addr in results:
                existing = seen.get(addr.concept)
                if existing is None or addr.confidence > existing.confidence:
                    seen[addr.concept] = addr
            deduped = sorted(seen.values(), key=lambda a: -a.confidence)
            deduped = deduped[: self.max_per_dimension]

            if deduped:
                addresses.extend(deduped)
            else:
                # No signal for this dimension -> null
                addresses.append(
                    SemanticAddress(
                        head_type=self.head_type,
                        concept=self.default_null,
                        confidence=1.0,
                        metadata={
                            "dimension": dim,
                            "raw": "",
                            "source": "null",
                        },
                    )
                )

        return addresses

    def _keyword_domain_fallback(self, text: str, doc) -> Optional[str]:
        """Scan text tokens against domain keyword dictionaries.

        Requires >= 2 matches from the same domain to activate.
        Returns the best-matching domain or None.
        """
        from .definitions import DOMAIN_KEYWORD_PATTERNS

        min_hits = 2
        text_lower = text.lower()
        # Use lemmatized tokens if doc available, else split on whitespace
        if doc is not None:
            tokens = {tok.lemma_.lower() for tok in doc}
        else:
            tokens = set(text_lower.split())

        best_domain = None
        best_count = 0
        for domain, keywords in DOMAIN_KEYWORD_PATTERNS.items():
            count = sum(1 for kw in keywords if kw in tokens or kw in text_lower)
            if count >= min_hits and count > best_count:
                best_count = count
                best_domain = domain

        return best_domain

    def _keyword_temporal_fallback(self, text: str, doc) -> Optional[str]:
        """Scan text for era/period keywords and year patterns.

        Returns a temporal concept or None.
        """
        import re
        from .definitions import TEMPORAL_KEYWORD_PATTERNS

        text_lower = text.lower()

        # Check era keywords
        for keyword, concept in TEMPORAL_KEYWORD_PATTERNS.items():
            if keyword in text_lower:
                return concept

        # Check for year patterns (e.g., "1865", "in 1492")
        year_match = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
        if year_match:
            return year_match.group(1)

        return None
