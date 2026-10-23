# src/encoders/heads/entity/head.py

import hashlib
import logging
import os
import re
from typing import Dict, Optional, List, Set, Tuple

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from .config import DEFAULT_CONFIG
from .storage.factory import create_entity_store
from .normalization import extract_candidates, normalize_mention, EntityCandidate, _sentence_starts
from .temporal_filter import is_fragment, is_non_entity_mention, is_temporal_value
from .comparators import get_comparator
from .codebook.lookup import get_codebook, classify_entity, REJECTION_TYPES
from .codebook.compound_filter import is_meaningful_compound
from .generic_nouns import is_generic_noun
from .concepts import (ConceptLearner, gate_enabled, in_seed, learnable, learning_enabled,
                       needs_concept, record_fingerprint)
from src.nlp import get_nlp_service

# What this default plugin declares (docs/design-head-plugins.md section 3), read with ast
# (nam_heads.sdk.impl_meta.read_module_literal): literals only. "resources" are the per-container
# limits and the containers per ingest pod it runs with on the reference deployment:
# the chart's ingest.entityResources.limits (cpu "1", memory 1Gi) and
# ingest.entityWorkers (2).
IMPL_META = {
    "slot": "entity",
    "name": "default",
    "version": "1.0.2",
    "contract": "entity-v1",
    "description": "Entity resolution: the record's name mentions as deterministic entity ids "
                   "(a hash of the normalized mention), with store-backed aliases.",
    "output": "entity-ids-v1",
    "resources": {"cpu": 1.0, "memory_mb": 1024, "workers": 2},
    "dependencies": [],
    # What host stages read from it besides the head (nam_heads.sdk.exports).
    "provides": {
        "normalize_mention": "normalization:normalize_mention",
        "comparator": "comparators:get_comparator",
        "open_store": "storage.couchbase_kv_store:CouchbaseKVEntityStore",
    },
}

logger = logging.getLogger(__name__)


# Word endings that are never a plural "s": business, campus, crisis.
_NOT_PLURAL_ENDINGS = ("ss", "us", "is")


def _hash_entity_id(entity_type: str, normalized: str) -> str:
    """Deterministic hash-based entity ID. Same input = same ID across pods.

    Entity type is NOT included in the ID because the ontology head can classify
    the same canonical form differently depending on input text length (full
    article vs single query word).  The type stays in address metadata and is
    used by the bundler for ontology assignment; the ontology dimension is
    expanded at query time to cover all types.
    """
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return h


def _ingest_entity_id(entity_type: str, canonical: str,
                      store_hit: Optional[str]) -> Tuple[str, bool]:
    """The id ingest emits for *canonical*, and whether the store already has it.

    Always ``_hash_entity_id(entity_type, canonical)`` -- the value the shared
    create logic produces on a miss -- so the emitted id does not depend on
    what the store holds, which pod asks, or how far LMDB sync has got.

    A store hit counts as "exists" only when it equals that id. A different
    hit is an alias owned by another canonical (``surf::abraham_lincoln``
    written by "President Abraham Lincoln") and must not change the id;
    the caller treats it as a miss and creates the entry.
    """
    eid = _hash_entity_id(entity_type, canonical)
    return eid, store_hit == eid



_TITLE_DISAMBIGUATOR_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _record_title(title: Optional[str]) -> str:
    """The name a record title gives its subject.

    The title is the record's (``ctx["title"]``, what ingest carries on the
    record), not a guess from the text. A trailing disambiguator goes ("Yves
    Rocher (company)" -> "Yves Rocher"), and so does a single comma qualifier
    ("Santa Margarita, California" -> "Santa Margarita", "Francesco II
    d'Este, Duke of Modena" -> "Francesco II d'Este"): other passages name
    the subject without it. A title with two or more commas is kept whole.
    """
    t = _TITLE_DISAMBIGUATOR_RE.sub("", (title or "").strip()).strip()
    if t.count(",") == 1:
        head = t.split(",", 1)[0].strip()
        if head:
            t = head
    return t


def _title_prefix_tokens(doc, title: Optional[str]) -> int:
    """How many leading tokens are the ingest adapter's "<title>. " prefix.

    The adapter puts the title in front of the payload unless the payload
    already leads with it (ingest/adapters/dcp/adapter.py). Those tokens are
    the title, which the head adds as one candidate of its own; their chunks
    ("Battle", "Politics", "company") are cut-off pieces of it. 0 when the
    text does not start with the title followed by a period.
    """
    if not title or doc is None:
        return 0
    target = re.sub(r"\s+", "", title)
    if not target:
        return 0
    toks = list(doc)
    acc = ""
    for k, t in enumerate(toks):
        acc += re.sub(r"\s+", "", t.text)
        if acc == target:
            return k + 2 if k + 1 < len(toks) and toks[k + 1].text == "." else 0
        if acc == target + ".":
            return k + 1          # "F.C." + "." read as one token
        if not target.startswith(acc):
            return 0
    return 0


def _ends_capitalised(text: str) -> bool:
    """The last word of *text* is capitalised or a number ("Armed Forces",
    "Apollo 15", "F . C ."): a name, whose form is kept in the entity id."""
    for w in reversed(text.split()):
        ch = next((c for c in w if c.isalnum()), "")
        if ch:
            return ch.isupper() or ch.isdigit()
    return False


def _with_title(candidates: List[EntityCandidate], title: Optional[str]) -> List[EntityCandidate]:
    """*candidates* with the record title as a name candidate in front.

    A mention with the title's normalized form is promoted and moved to the
    front (its surface is kept: it matches the parse token by token);
    otherwise the title is added in front. Either way it is a name (conf
    0.6, PROPN): the record is about it, so no codebook rejection, compound
    filter or generic-noun list drops it, and max_entities_per_record never
    cuts it.
    """
    name = _record_title(title)
    if not name:
        return candidates
    norm = normalize_mention(name)
    if not norm:
        return candidates
    for ci, c in enumerate(candidates):
        if c.normalized == norm:
            # Moved to the front: max_entities_per_record never cuts the title.
            title_cand = EntityCandidate(
                text=c.text, normalized=norm, confidence=max(c.confidence, 0.6),
                pos_pattern="PROPN" if not c.is_name else c.pos_pattern,
                token_count=c.token_count, is_propn=True, has_digit=c.has_digit,
                has_hyphen=c.has_hyphen, ner_labels=c.ner_labels, is_name=True,
                start=c.start,
            )
            return [title_cand] + list(candidates[:ci]) + list(candidates[ci + 1:])
    return [EntityCandidate(
        text=name, normalized=norm, confidence=0.6, pos_pattern="PROPN",
        token_count=len(name.split()), is_propn=True,
        has_digit=any(ch.isdigit() for ch in name), has_hyphen="-" in name,
        is_name=True,
    )] + list(candidates)


def _name_type(cand: EntityCandidate, entity_type: str) -> str:
    """A name is never a codebook rejection: the codebook reads shape, not
    case, and rejected "Moon", "LDS Church" and "NFL Draft" as generic nouns.
    A rejection type on a name becomes ``proper_noun``."""
    if cand.is_name and entity_type in REJECTION_TYPES:
        return "proper_noun"
    return entity_type


def _drop_non_entities(candidates: List[EntityCandidate]) -> List[EntityCandidate]:
    """Candidates minus temporal values and 1-2 character fragments.

    Dates, years, decades and centuries are temporal context, never entities
    (the context head carries them); "st", "ii" are fragments. Applied to
    ingest and query candidates alike (temporal_filter.py).
    """
    return [c for c in candidates
            if not is_non_entity_mention(c.text, c.normalized, c.ner_labels)]


class EntityEncoder(EncoderHead):
    """
    Entity encoder.

    Answers: "Which specific things are being referred to?"
    """

    HEAD_TYPE = "entity"

    def _initialize(self):
        cfg = {**DEFAULT_CONFIG, **self.config}

        self.enable_diagnostics = cfg["enable_diagnostics"]
        self.max_entities_per_record = cfg["max_entities_per_record"]
        self.create_new_entities = cfg["create_new_entities"]
        self.min_confidence = cfg.get("min_entity_confidence", 0.35)
        self.enable_fuzzy = cfg.get("enable_fuzzy_resolution", True)
        self.fuzzy_max_candidates = cfg.get("fuzzy_resolution_max_candidates", 50)
        self._rule_nlp = None  # lazy-init rule NLP backend

        # Store creation (fallback to memory handled internally)
        self.store = create_entity_store(
            backend=cfg["db_backend"],
            fallback=cfg["db_fallback"],
            kv_on_cache_miss=self.kv_on_cache_miss,
        )

        logger.info("EntityEncoder initialized (fuzzy=%s)", self.enable_fuzzy)

    def get_metrics(self) -> Dict:
        base = super().get_metrics()
        store_stats = {}
        if hasattr(self.store, '_surf_cache'):
            store_stats["surf_cache_size"] = len(self.store._surf_cache)
        if hasattr(self.store, '_cand_cache'):
            store_stats["cand_cache_size"] = len(self.store._cand_cache)
        if store_stats:
            base["store_stats"] = store_stats
        return base

    def config_summary(self) -> Dict:
        return {
            "max_entities_per_record": self.max_entities_per_record,
            "min_entity_confidence": self.min_confidence,
            "enable_fuzzy_resolution": self.enable_fuzzy,
            "storage_backend": type(self.store).__name__,
        }

    # -- Config: skip fuzzy during ingest (aliases handle most cross-references) --
    _FUZZY_INGEST = os.environ.get("NAM_ENTITY_FUZZY_INGEST", "1").lower() in ("1", "true", "yes")

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._encode_internal(text, mode="ingest", context=context)

    # Common query prefixes that obscure the entity mention for NLP parsing.
    # "tell me about Manufacturing Consent" -> NLP sees "Manufacturing" as VERB.
    # Stripping yields "Manufacturing Consent" -> NLP sees compound noun.
    _QUERY_PREFIXES = re.compile(
        r"^(?:tell\s+me\s+about|what\s+(?:is|are)|who\s+(?:is|are|was)|"
        r"where\s+is|show\s+me|find|search\s+for|look\s+up|describe)\s+",
        re.IGNORECASE,
    )

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        result = self._encode_internal(query, mode="query", context=context)

        # If query has a common prefix, also try the bare entity mention.
        # Merge any additional entities found from the stripped version.
        stripped = self._QUERY_PREFIXES.sub("", query)
        if stripped != query and stripped:
            bare_result = self._encode_internal(stripped, mode="query", context=context)
            existing_concepts = {a.concept for a in result.addresses}
            for addr in bare_result.addresses:
                if addr.concept not in existing_concepts:
                    result.addresses.append(addr)
                    existing_concepts.add(addr.concept)

        return result

    # ------------------------------------------------------------------
    # Cross-text batch encode -- called by MultiHeadEncoder.encode_batch
    # ------------------------------------------------------------------
    _phase_log_every = int(os.environ.get("NAM_ENTITY_PHASE_LOG_EVERY", "10"))
    _phase_batch_count = 0
    _phase_totals: Dict[str, float] = {}
    _phase_counts: Dict[str, int] = {}

    def encode_batch_texts(
        self,
        texts: List[str],
        contexts: Optional[List[Optional[Dict]]] = None,
    ) -> List[EncodingResult]:
        """Encode multiple texts, batching ALL KV operations across texts."""
        if not texts:
            return []
        if contexts is None:
            contexts = [None] * len(texts)

        # Lazy-init: tests bypass __init__ via __new__, so ensure the attr exists.
        if not hasattr(self, "_ensured_entities"):
            self._ensured_entities: set = set()

        n = len(texts)
        use_fuzzy_ingest = self.enable_fuzzy and self._FUZZY_INGEST
        import time as _time
        _t0 = _time.perf_counter()

        # ---- Phase 1: Prepare candidates for ALL texts ----
        # Extract candidates and canonicalize, then batch-classify types
        per_text = []  # per_text[i] = list of (cand, canonical, entity_type, lemma_cache)
        # Collect features for batch classification
        all_features = []   # flat list of feature vectors
        all_feature_idx = []  # (text_idx, cand_idx_in_prepared) for mapping back
        codebook = get_codebook()

        # First pass: extract candidates, build canonical IDs, collect features
        per_text_untyped = []  # per_text[i] = list of (cand, canonical, lemma_cache)
        for i in range(n):
            ctx = contexts[i] or {}
            spacy_doc = ctx.get("spacy_doc")
            # The record title (ctx["title"], run_head) is the subject: a
            # name candidate of its own, in front.
            candidates = _drop_non_entities(
                self._record_candidates(texts[i], spacy_doc, ctx.get("title")))

            lemma_cache: Dict[str, str] = {}
            propn_tokens: Set[str] = set()
            if spacy_doc is not None:
                for token in spacy_doc:
                    lemma_cache[token.text.lower()] = token.lemma_.lower()
                    if token.pos_ == "PROPN":
                        propn_tokens.add(token.text.lower())

            # Collect more candidates than max_entities_per_record because
            # some will be filtered as rejection types after batch classify.
            # The single-encode path skips rejections before counting toward
            # the limit; match that behavior by over-collecting here and
            # trimming after the rejection filter.
            collect_limit = self.max_entities_per_record * 2
            prepared = []
            for cand in candidates:
                if cand.confidence < self.min_confidence:
                    continue
                if len(prepared) >= collect_limit:
                    break
                canonical = self._canonical_for(cand, lemma_cache, propn_tokens)
                if is_generic_noun(cand, canonical):
                    continue
                prepared.append((cand, canonical, lemma_cache))
                # Collect feature vector for batch classify
                feat = codebook._build_features(
                    cand.normalized, cand.pos_pattern, cand.text,
                    cand.token_count, cand.is_propn, cand.has_digit,
                    cand.has_hyphen, cand.confidence, 0.0,
                )
                all_features.append(feat)
                all_feature_idx.append((i, len(prepared) - 1))

            per_text_untyped.append(prepared)

        # Batch classify all candidates at once (single vectorized numpy op)
        if all_features:
            all_types = codebook.classify_batch(all_features)
        else:
            all_types = []

        # Map types back and build final per_text structure, filtering rejections
        for fi, (ti, ci) in enumerate(all_feature_idx):
            cand, canonical, lemma_cache = per_text_untyped[ti][ci]
            per_text_untyped[ti][ci] = (cand, canonical, _name_type(cand, all_types[fi]), lemma_cache)

        # Filter out rejection types, then trim to max_entities_per_record.
        # This matches the single-encode path where rejections are skipped
        # before counting toward the limit.
        per_text = []
        for i in range(n):
            filtered = [
                entry for entry in per_text_untyped[i]
                if len(entry) == 4 and not codebook.is_rejection_type(entry[2])
            ]
            per_text.append(filtered[:self.max_entities_per_record])

        # Meaningful before any store work: a candidate that is not gets an
        # address (the sketch resolves mentions through it) but no store
        # entry, so a description never resolves as an entity later.
        verdicts = self._concept_verdicts(texts, contexts, per_text, learn=True)
        meaningful_of = [
            [self._meaningful(entry[0], entry[2], verdicts[i][ci])
             for ci, entry in enumerate(per_text[i])]
            for i in range(n)
        ]

        _t1 = _time.perf_counter()

        # ---- Phase 2: Fuzzy-first batch resolve for ALL candidates ----
        resolved_entities = [[None] * len(per_text[i]) for i in range(n)]

        if use_fuzzy_ingest:
            # Build fuzzy queries for all candidates
            fuzzy_queries = []  # (canonical, entity_type, comparator)
            fuzzy_index = []    # (text_idx, cand_idx)
            for i in range(n):
                for ci, (cand, canonical, entity_type, _lc) in enumerate(per_text[i]):
                    if not meaningful_of[i][ci]:
                        continue
                    comp = get_comparator(entity_type)
                    fuzzy_queries.append((canonical, entity_type, comp))
                    fuzzy_index.append((i, ci))

            if fuzzy_queries:
                fuzzy_results = self.store.resolve_fuzzy_batch(
                    fuzzy_queries, max_candidates=self.fuzzy_max_candidates,
                )
                for qi, (ti, ci) in enumerate(fuzzy_index):
                    _cand, canonical, entity_type, _lc = per_text[ti][ci]
                    eid, exists = _ingest_entity_id(
                        entity_type, canonical, fuzzy_results.get(qi))
                    if exists:
                        resolved_entities[ti][ci] = eid

        _t2 = _time.perf_counter()
        _t3 = _time.perf_counter()

        # ---- Phase 3: Batch create all new entities ----
        # Also ensure ent:: records exist for Phase-2-resolved entities.
        # After a session flush, LMDB may retain stale surf:: keys that cause
        # Phase 2 to resolve entities whose ent:: records no longer exist.
        # Track which entity_ids we've already ensured this process lifetime.
        create_items = []  # (entity_id, surface_forms, metadata)
        create_index = []  # (text_idx, cand_idx)
        fuzzy_bulk = []     # for candidate key storage
        ensure_items = []   # Phase-2-resolved entities needing ent:: write

        for i in range(n):
            for ci, (cand, canonical, entity_type, lemma_cache) in enumerate(per_text[i]):
                if not meaningful_of[i][ci]:
                    resolved_entities[i][ci] = _hash_entity_id(entity_type, canonical)
                    continue
                if resolved_entities[i][ci] is not None:
                    # Phase 2 resolved — ensure ent:: record exists if not
                    # already ensured in this process lifetime.
                    eid = resolved_entities[i][ci]
                    if eid not in self._ensured_entities:
                        self._ensured_entities.add(eid)
                        surface_forms = self._surface_forms_with_aliases(
                            canonical, extra_aliases=[cand.normalized],
                            lemma_cache=lemma_cache, entity_type=entity_type,
                        )
                        meta = {"source": "entity_encoder", "entity_type": entity_type}
                        ensure_items.append((eid, surface_forms, meta))
                    continue
                entity_id = _hash_entity_id(entity_type, canonical)
                resolved_entities[i][ci] = entity_id
                self._ensured_entities.add(entity_id)
                surface_forms = self._surface_forms_with_aliases(
                    canonical, extra_aliases=[cand.normalized], lemma_cache=lemma_cache,
                    entity_type=entity_type,
                )
                meta = {"source": "entity_encoder", "entity_type": entity_type}
                create_items.append((entity_id, surface_forms, meta))
                create_index.append((i, ci))

                if use_fuzzy_ingest:
                    comp = get_comparator(entity_type)
                    for sf in surface_forms:
                        keys = comp.generate_keys(sf)
                        if keys:
                            fuzzy_bulk.append((entity_id, sf, keys))

        if create_items:
            self.store.create_entities_batch(create_items)
        if ensure_items:
            self.store.create_entities_batch(ensure_items)
        if fuzzy_bulk and hasattr(self.store, "store_candidate_keys_bulk"):
            self.store.store_candidate_keys_bulk(fuzzy_bulk)

        _t4 = _time.perf_counter()

        # ---- Phase 4: Build per-text EncodingResults ----
        results = []
        for i in range(n):
            addresses: List[SemanticAddress] = []
            for ci, (cand, canonical, entity_type, _lc) in enumerate(per_text[i]):
                entity_id = resolved_entities[i][ci]
                if entity_id is None:
                    continue
                meaningful = meaningful_of[i][ci]
                addresses.append(SemanticAddress(
                    head_type=self.head_type,
                    concept=entity_id,
                    confidence=cand.confidence,
                    metadata={
                        "surface_form": cand.text,
                        "normalized": cand.normalized,
                        "entity_type": entity_type,
                        "meaningful": meaningful,
                    },
                ))

            # Fallback: if no addresses, emit first Title-Case bigram or first token
            if self.create_new_entities and not addresses:
                fallback_norm = self._fallback_entity(texts[i])
                if fallback_norm:
                    ctx = contexts[i] or {}
                    spacy_doc = ctx.get("spacy_doc")
                    lemma_cache_fb: Dict[str, str] = {}
                    propn_tokens_fb: Set[str] = set()
                    if spacy_doc is not None:
                        for token in spacy_doc:
                            lemma_cache_fb[token.text.lower()] = token.lemma_.lower()
                            if token.pos_ == "PROPN":
                                propn_tokens_fb.add(token.text.lower())
                    canonical_fb = self._canonical_entity_id(fallback_norm, lemma_cache_fb, propn_tokens_fb)
                    etype_fb = self._detect_type(fallback_norm)
                    entity_id_fb = _hash_entity_id(etype_fb, canonical_fb)
                    surface_forms_fb = self._surface_forms_with_aliases(
                        canonical_fb, extra_aliases=[fallback_norm],
                        lemma_cache=lemma_cache_fb, entity_type=etype_fb,
                    )
                    self.store.create_entity(
                        entity_id=entity_id_fb, surface_forms=surface_forms_fb,
                        metadata={"fallback": True, "entity_type": etype_fb},
                    )
                    addresses.append(SemanticAddress(
                        head_type=self.head_type,
                        concept=entity_id_fb,
                        confidence=0.1,
                        metadata={"fallback": True, "normalized": fallback_norm, "entity_type": etype_fb},
                    ))

            results.append(EncodingResult(
                head_type=self.head_type,
                addresses=addresses,
                metadata={"mode": "ingest", "entity_count": len(addresses)},
            ))

        _t5 = _time.perf_counter()
        # Accumulate phase timings
        phases = {
            "p1_prepare": (_t1 - _t0) * 1000,
            "p2_resolve": (_t2 - _t1) * 1000,
            "p3_fuzzy": (_t3 - _t2) * 1000,
            "p4_create": (_t4 - _t3) * 1000,
            "p5_build": (_t5 - _t4) * 1000,
        }
        EntityEncoder._phase_batch_count += 1
        for k, v in phases.items():
            EntityEncoder._phase_totals[k] = EntityEncoder._phase_totals.get(k, 0) + v
            EntityEncoder._phase_counts[k] = EntityEncoder._phase_counts.get(k, 0) + 1
        if EntityEncoder._phase_batch_count % EntityEncoder._phase_log_every == 0:
            parts = []
            for k in ("p1_prepare", "p2_resolve", "p3_fuzzy", "p4_create", "p5_build"):
                avg = EntityEncoder._phase_totals.get(k, 0) / max(EntityEncoder._phase_counts.get(k, 1), 1)
                parts.append(f"{k}={avg:.1f}ms")
            unresolved = len([1 for items in unresolved_items]) if 'unresolved_items' in dir() else 0
            logger.info("entity.phase_timing batches=%d n=%d created=%d fuzzy=%s %s",
                        EntityEncoder._phase_batch_count, n, len(create_items),
                        use_fuzzy_ingest, " ".join(parts))

        return results

    # ------------------------------------------------------------------
    # Single-text encode (used by query path and per-text fallback)
    # ------------------------------------------------------------------
    def _encode_internal(self, text: str, mode: str, context: Optional[Dict] = None) -> EncodingResult:
        # Use shared doc from context if available; fall back to rule NLP
        spacy_doc = context.get("spacy_doc") if context else None
        if spacy_doc is None:
            spacy_doc = self._parse_text(text)
        # The record title, if the caller passed one (see the batch path).
        candidates = self._record_candidates(text, spacy_doc, (context or {}).get("title"))

        # Query mode: inject the full query text as a candidate if NLP
        # didn't produce it as a noun chunk.  "acid rain" → NLP only yields
        # "rain", missing the compound.  The user's query IS their intent.
        # Skip injection when NLP already found a proper noun — the full text
        # would include verbs/prepositions that pollute the entity hash
        # (e.g. "assassinated Abraham Lincoln" → "assassinated_abraham_lincoln").
        has_propn = any(c.is_propn for c in candidates)
        if mode == "query" and len(text.split()) >= 2 and not has_propn:
            full_norm = normalize_mention(text)
            if full_norm and not any(c.normalized == full_norm for c in candidates):
                # Build POS from available tokens
                pos_parts = []
                if spacy_doc is not None:
                    for token in spacy_doc:
                        pos_parts.append(token.pos_)
                pos_pattern = "_".join(pos_parts) if pos_parts else "UNK"
                candidates.insert(0, EntityCandidate(
                    text=text, normalized=full_norm,
                    confidence=0.5, pos_pattern=pos_pattern,
                    token_count=len(text.split()), is_propn=False,
                ))

        # Same rule as ingest: a query about "the 1980s" must not look for a
        # 1980s partition, which ingest never writes.
        candidates = _drop_non_entities(candidates)

        min_conf = 0.15 if mode == "query" else self.min_confidence

        lemma_cache = {}
        propn_tokens: Set[str] = set()
        if spacy_doc is not None:
            for token in spacy_doc:
                lemma_cache[token.text.lower()] = token.lemma_.lower()
                if token.pos_ == "PROPN":
                    propn_tokens.add(token.text.lower())

        addresses: List[SemanticAddress] = []
        create_new = self.create_new_entities and mode != "query"
        use_fuzzy = self.enable_fuzzy and (mode == "query" or self._FUZZY_INGEST)

        # Prepare candidates
        codebook = get_codebook()
        prepared = []
        for cand in candidates:
            if cand.confidence < min_conf:
                continue
            if len(prepared) >= self.max_entities_per_record:
                break
            canonical = self._canonical_for(cand, lemma_cache, propn_tokens)
            # Generic nouns are never an ingest entity. A query keeps what the
            # user typed (as with codebook rejections below).
            if mode != "query" and is_generic_noun(cand, canonical):
                continue
            entity_type = _name_type(cand, self._detect_type(
                cand.normalized, cand.pos_pattern, cand.text,
                cand.token_count, cand.is_propn, cand.has_digit,
                cand.has_hyphen, cand.confidence,
            ))
            # Skip rejection types (non-entity noise from codebook).
            # Query mode: never reject — the user typed it intentionally.
            if mode != "query" and codebook.is_rejection_type(entity_type):
                continue
            prepared.append((cand, canonical, entity_type))

        # As in the batch path. A query needs no verdict: the query bundler
        # ignores "meaningful", and a phrase ingest never made an entity has
        # no store entry for the query to resolve.
        verdicts = ([None] * len(prepared) if mode == "query" else
                    self._concept_verdicts([text], [context], [prepared], learn=True)[0])
        meaningful_of = [self._meaningful(cand, entity_type, verdicts[pi])
                         for pi, (cand, _c, entity_type) in enumerate(prepared)]

        # Fuzzy-first batch resolve for all candidates
        fuzzy_resolved: Dict[int, Optional[str]] = {}
        if use_fuzzy and prepared:
            fuzzy_queries = [
                (canonical, entity_type, get_comparator(entity_type))
                for _, canonical, entity_type in prepared
            ]
            fuzzy_resolved = self.store.resolve_fuzzy_batch(
                fuzzy_queries, max_candidates=self.fuzzy_max_candidates,
            )

        for pi, (cand, canonical, entity_type) in enumerate(prepared):
            meaningful = meaningful_of[pi]
            if mode != "query" and not meaningful:
                addresses.append(SemanticAddress(
                    head_type=self.head_type,
                    concept=_hash_entity_id(entity_type, canonical),
                    confidence=cand.confidence,
                    metadata={"surface_form": cand.text, "normalized": cand.normalized,
                              "entity_type": entity_type, "meaningful": False},
                ))
                continue
            entity_id = fuzzy_resolved.get(pi)
            alt_ids: List[str] = []
            if mode == "query":
                if entity_id is None:
                    entity_id = self._resolve_exact(canonical, cand.normalized, entity_type)
                # Ingest writes every record mentioning this text under
                # hash(canonical); that is the primary. A different id the
                # store resolved (an alias owned by another canonical) is
                # carried as an extra candidate.
                if entity_id is not None:
                    pure_id = _hash_entity_id(entity_type, canonical)
                    if entity_id != pure_id:
                        alt_ids.append(entity_id)
                        entity_id = pure_id
            else:
                pure_id, exists = _ingest_entity_id(entity_type, canonical, entity_id)
                entity_id = pure_id if exists else None

            if entity_id is None and create_new:
                entity_id = _hash_entity_id(entity_type, canonical)
                surface_forms = self._surface_forms_with_aliases(
                    canonical, extra_aliases=[cand.normalized], lemma_cache=lemma_cache,
                    entity_type=entity_type,
                )
                self.store.create_entity(
                    entity_id=entity_id,
                    surface_forms=surface_forms,
                    metadata={"source": "entity_encoder", "entity_type": entity_type},
                )
                if use_fuzzy:
                    comparator = get_comparator(entity_type)
                    bulk_items = []
                    for sf in surface_forms:
                        keys = comparator.generate_keys(sf)
                        if keys:
                            bulk_items.append((entity_id, sf, keys))
                    if bulk_items and hasattr(self.store, "store_candidate_keys_bulk"):
                        self.store.store_candidate_keys_bulk(bulk_items)
                    elif bulk_items:
                        for eid, sf, keys in bulk_items:
                            self.store.store_candidate_keys(eid, sf, keys)

            if entity_id is None and mode == "query":
                entity_id = _hash_entity_id(entity_type, canonical)
                addresses.append(SemanticAddress(
                    head_type=self.head_type, concept=entity_id,
                    confidence=cand.confidence * 0.5,
                    metadata={"surface_form": cand.text, "normalized": cand.normalized,
                              "entity_type": entity_type, "unresolved": True,
                              "meaningful": meaningful},
                ))
                continue

            if entity_id is None:
                continue

            meta = {"surface_form": cand.text, "normalized": cand.normalized,
                    "entity_type": entity_type, "meaningful": meaningful}
            if alt_ids:
                meta["alt_entity_ids"] = alt_ids
            addresses.append(SemanticAddress(
                head_type=self.head_type, concept=entity_id,
                confidence=cand.confidence,
                metadata=meta,
            ))

        if create_new and not addresses:
            fallback_norm = self._fallback_entity(text)
            if fallback_norm:
                canonical = self._canonical_entity_id(fallback_norm, lemma_cache, propn_tokens)
                entity_type = self._detect_type(fallback_norm)
                # Try fuzzy resolution for fallback entity
                hit = None
                if use_fuzzy:
                    comp = get_comparator(entity_type)
                    hit = self.store.resolve_fuzzy(
                        canonical, entity_type, comp,
                        max_candidates=self.fuzzy_max_candidates,
                    )
                entity_id, exists = _ingest_entity_id(entity_type, canonical, hit)
                if not exists:
                    surface_forms = self._surface_forms_with_aliases(
                        canonical, extra_aliases=[fallback_norm], lemma_cache=lemma_cache,
                        entity_type=entity_type,
                    )
                    self.store.create_entity(
                        entity_id=entity_id, surface_forms=surface_forms,
                        metadata={"fallback": True, "entity_type": entity_type},
                    )
                addresses.append(SemanticAddress(
                    head_type=self.head_type, concept=entity_id, confidence=0.1,
                    metadata={"fallback": True, "normalized": fallback_norm, "entity_type": entity_type},
                ))

        return EncodingResult(
            head_type=self.head_type, addresses=addresses,
            metadata={"mode": mode, "entity_count": len(addresses)},
        )

    def _resolve_exact(self, canonical: str, normalized: str, entity_type: str) -> Optional[str]:
        """Exact surface-form lookup: surf::{type}::{name}, then surf::{name}.

        The keys ingest writes for every surface form (create_entity). Fuzzy
        resolution runs first; this catches stored forms the comparator's
        candidate keys do not reach.
        """
        for name in dict.fromkeys((canonical, normalized)):
            if not name:
                continue
            for etype in (entity_type, None):
                eid = self.store.resolve_surface_form(name, etype)
                if eid:
                    return eid
        return None

    def resolve_query_entities(self, text: str, context: Optional[Dict] = None) -> List[Dict]:
        """Entities in *text* that exist in the store, strongest first.

        The ONE resolver for query-side entity lookup. Two sources, both
        reading only the keys ingest writes:

        1. The entity head's own query path (encode_query): same candidate
           extraction, normalization and store lookups as the address path.
           Unresolved mention hashes and the fallback guess are dropped.
        2. The query's contiguous word spans (>= 2 words, no stop word at
           either edge) looked up as exact surf:: keys in one batch. The head
           proposes noun chunks, which miss stored names it splits or
           over-extends: "What was the Apple II?" yields only "ii", and the
           chunk "Polydor Records artists" is not stored while
           "polydor_records" is.

        Then dropped: a mention covered by a longer resolved mention ("ii"
        inside "apple_ii"); a bare roman numeral, a number, or a 1-2
        character fragment (the ingest rule: "UN", "U2" stay, "II", "TV",
        "St" go); a temporal span; a mention starting or ending with a function
        or question word ("what_label" -- ingest stores such chunks too); and,
        once any specific name resolved, generic common nouns. Ordered longest first, then
        by confidence -- the first entity seeds the graph walk.

        Each item: entity_id, normalized, surface_form, entity_type, confidence;
        plus alt_entity_ids when the store resolved the mention to an id other
        than hash(canonical) (entity_id is always hash(canonical) for a
        head-path mention -- where ingest writes every record with that text).
        """
        result = self.encode_query(text, context=context)
        found: Dict[str, Dict] = {}  # normalized -> item

        def add(eid: str, normalized: str, surface: str, etype: str, conf: float,
                alt_ids: Optional[List[str]] = None) -> None:
            if not eid or not normalized or normalized in found:
                return
            found[normalized] = {"entity_id": eid, "normalized": normalized,
                                 "surface_form": surface, "entity_type": etype,
                                 "confidence": conf}
            if alt_ids:
                found[normalized]["alt_entity_ids"] = list(alt_ids)

        for addr in sorted(result.addresses, key=lambda a: -a.confidence):
            meta = addr.metadata or {}
            if meta.get("fallback") or meta.get("unresolved"):
                continue
            add(addr.concept, meta.get("normalized", ""), meta.get("surface_form", ""),
                meta.get("entity_type", ""), addr.confidence, meta.get("alt_entity_ids"))

        # A temporal span ("late_1980s", "19th_century") is never an entity,
        # even if a store written before that rule still holds it.
        spans = [sp for sp in self._text_spans(text)
                 if sp not in found and not is_temporal_value(sp)]
        if spans:
            hits = self.store.resolve_surface_forms_batch([(sp, None) for sp in spans])
            for sp in spans:
                eid = hits.get((sp, None))
                if eid:
                    ent = self.store.get_entity(eid) or {}
                    etype = (ent.get("metadata") or {}).get("entity_type", "")
                    add(eid, sp, sp.replace("_", " "), str(etype), self._SPAN_CONFIDENCE)

        edge = self._NAME_REJECT_EDGES
        items = [it for it in found.values()
                 if not self._is_weak_mention(it["normalized"], it["surface_form"])
                 and not ({it["normalized"].split("_")[0], it["normalized"].split("_")[-1]} & edge)]
        # Once a specific entity resolves, generic common nouns ("label",
        # "king", "book") only steer the graph seed and the routing reason
        # away from it.
        if any(self._is_specific(it) for it in items):
            items = [it for it in items if self._is_specific(it)]
        token_sets = [(it, it["normalized"].split("_")) for it in items]
        kept = [it for it, toks in token_sets
                if not any(len(o) > len(toks) and self._contains_run(o, toks)
                           for _, o in token_sets)]
        kept.sort(key=lambda it: (-len(it["normalized"].split("_")), -it["confidence"]))
        out: List[Dict] = []
        seen_ids: Set[str] = set()
        for it in kept:
            if it["entity_id"] not in seen_ids:
                seen_ids.add(it["entity_id"])
                out.append(it)
        return out

    _SPAN_CONFIDENCE = 0.5
    # A resolved mention starting or ending with one of these is a parse
    # fragment, not a name. Articles are NOT here: "the_beatles" and
    # "the_hague" are names, and may be stored only in that form.
    _NAME_REJECT_EDGES = frozenset({
        "who", "what", "where", "when", "why", "how", "which", "whom", "whose",
        "is", "was", "are", "were", "be", "been", "am", "do", "did", "does",
        "has", "have", "had",
    })

    # Words a looked-up span never starts or ends with: function words only.
    # NOT _COMPONENT_STOP_WORDS, which also holds adjectives ("great", "new",
    # "first") that do end or start names: alexander_the_great, new_york.
    _SPAN_EDGE_WORDS = frozenset({
        "who", "what", "where", "when", "why", "how", "which", "whom", "whose",
        "the", "a", "an", "and", "or", "but", "nor", "of", "in", "on", "at", "to",
        "for", "with", "from", "by", "as", "about", "into", "than", "that", "this",
        "is", "was", "are", "were", "be", "been", "am", "do", "did", "does",
        "has", "have", "had", "it", "its", "his", "her", "their", "our",
    })
    _MAX_SPAN_TOKENS = 6
    # A well-formed roman numeral, not any word spelled from i/v/x/l/c/d/m
    # ("civil", "mild" and "did" are words). "mix" (1009) is the known cost.
    _ROMAN_NUMERAL = re.compile(r"^m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$")

    @classmethod
    def _text_spans(cls, text: str) -> List[str]:
        """Contiguous word spans of *text* (normalized), longest first.

        At least 2 and at most _MAX_SPAN_TOKENS words; a span may not start or
        end with a function word ("the_apple_ii" is never looked up, "apple_ii"
        and "alexander_the_great" are).
        """
        parts = [p for p in normalize_mention(text or "").split("_") if p]
        stop = cls._SPAN_EDGE_WORDS
        spans: List[str] = []
        for n in range(min(len(parts), cls._MAX_SPAN_TOKENS), 1, -1):
            for i in range(len(parts) - n + 1):
                run = parts[i:i + n]
                if run[0] in stop or run[-1] in stop:
                    continue
                spans.append("_".join(run))
        return spans

    @classmethod
    def _is_weak_mention(cls, normalized: str, surface: str = "") -> bool:
        """A single token that cannot identify anything on its own: a roman
        numeral ("xiv"), a bare number, or a 1-2 character fragment by the
        rule ingest applies (temporal_filter.is_fragment): "UN", "DC", "U2"
        stay; "II", "TV", "St", lower-case "un" go. 1-2 character tokens are
        decided by that rule alone, not the numeral pattern, which would also
        take DC, MD and MI."""
        if "_" in normalized:
            return False
        if len(normalized) <= 2:
            return is_fragment(surface or "", normalized)
        if (normalized and cls._ROMAN_NUMERAL.match(normalized)) or normalized.isdigit():
            return True
        return False

    @staticmethod
    def _is_specific(item: Dict) -> bool:
        """A name rather than a common noun: several words, or written with a
        capital in the query ("Sidon", "UK"), and not a codebook rejection type."""
        if str(item.get("entity_type") or "").startswith("rejection"):
            return False
        if "_" in (item.get("normalized") or ""):
            return True
        surface = (item.get("surface_form") or "").strip()
        return bool(surface) and surface[0].isupper()

    @staticmethod
    def _contains_run(outer: List[str], inner: List[str]) -> bool:
        n = len(inner)
        return any(outer[i:i + n] == inner for i in range(len(outer) - n + 1))

    def _detect_type(
        self,
        normalized: str,
        pos_pattern: str = "UNK",
        text: str = "",
        token_count: int = 1,
        is_propn: bool = False,
        has_digit: bool = False,
        has_hyphen: bool = False,
        confidence: float = 0.0,
    ) -> str:
        """Detect entity type via codebook nearest-centroid lookup.

        Uses unsupervised VQ codebook trained on 7.5M entity candidates.
        Deterministic: same input always produces the same type.
        """
        return classify_entity(
            normalized=normalized,
            pos_pattern=pos_pattern,
            text=text,
            token_count=token_count,
            is_propn=is_propn,
            has_digit=has_digit,
            has_hyphen=has_hyphen,
            confidence=confidence,
        )

    def _fallback_entity(self, text: str) -> Optional[str]:
        """First Title-Case bigram, else the first token that is not a
        temporal value or a 1-2 character fragment (temporal_filter)."""
        import re
        title_tokens = re.findall(r"[A-Z][a-z]+", text)
        if len(title_tokens) >= 2:
            first, last = title_tokens[0], title_tokens[-1]
            return f"{first.lower()}_{last.lower()}"
        for tok in re.findall(r"[A-Za-z0-9]+", text):
            if not is_non_entity_mention(tok, tok.lower()):
                return tok.lower()
        return None

    # Honorific / title prefixes stripped from surface forms so that
    # "President Abraham Lincoln" also stores "abraham_lincoln" as a
    # surface form.  Fuzzy resolution at query time then matches
    # "abraham_lincoln" → the entity stored as "president_abraham_lincoln".
    _TITLE_PREFIXES = frozenset({
        "president", "vice", "senator", "governor", "mayor", "king", "queen",
        "prince", "princess", "emperor", "empress", "pope", "cardinal",
        "bishop", "reverend", "rabbi", "imam", "sheikh",
        "general", "admiral", "colonel", "captain", "lieutenant", "sergeant",
        "commander", "marshal", "corporal", "private",
        "dr", "prof", "professor", "sir", "dame", "lord", "lady",
        "mr", "mrs", "ms", "miss", "master", "mx",
        "chief", "secretary", "minister", "chancellor", "ambassador",
        "representative", "speaker", "justice", "judge",
    })

    # Entity types where individual name components are useful surface forms.
    # "einstein" should resolve to "albert_einstein" (full_name).
    # "system" should NOT resolve to "control_system" (common_noun).
    _COMPONENT_EXPANSION_TYPES = frozenset({
        # Legacy types (backwards compatibility with stored entities)
        "person", "organization", "group",
        # Codebook types that benefit from component expansion
        "full_name", "titled_entity", "proper_noun",
    })

    # Stop words filtered from component expansion to prevent noise entries
    # like surf::the, surf::first pointing to arbitrary entities.
    _COMPONENT_STOP_WORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
        "for", "with", "from", "by", "as", "is", "was", "are", "were",
        "not", "no", "its", "his", "her", "our", "their",
        "one", "two", "three", "four", "five", "first", "second", "third",
        "new", "old", "great", "big", "small", "large", "most", "other",
        "also", "than", "then", "very", "just", "only", "more", "some",
        "any", "each", "every", "such", "own", "same", "after", "before",
        "between", "under", "over", "about", "into", "through",
        "born", "known", "died", "lived", "called", "named", "located",
    })

    MAX_SURFACE_FORMS = 5

    def _surface_forms_with_aliases(self, normalized: str, extra_aliases: Optional[List[str]] = None,
                                    lemma_cache: Optional[Dict[str, str]] = None,
                                    entity_type: Optional[str] = None) -> List[str]:
        """
        Expand primary normalized form with deterministic aliases.

        Only proper-name entity types (person, organization, group) get
        individual component expansion. Common nouns store only the
        canonical compound form. Resolution normalizes input at query
        time, so conjugations/plurals are never stored.
        """
        forms = [normalized]
        if extra_aliases:
            for a in extra_aliases:
                if a not in forms:
                    forms.append(a)

        parts = normalized.split("_")

        # Title-stripped form: "president_abraham_lincoln" → "abraham_lincoln".
        # Stored as a surface form so fuzzy resolution matches queries that
        # omit the title.  Only strip leading title prefixes.
        if len(parts) >= 2 and entity_type in self._COMPONENT_EXPANSION_TYPES:
            i = 0
            while i < len(parts) - 1 and parts[i] in self._TITLE_PREFIXES:
                i += 1
            if i > 0:
                title_stripped = "_".join(parts[i:])
                if title_stripped not in forms:
                    forms.append(title_stripped)

        # Name shortening (first_last) — only for proper-name types.
        # Use leading-title-stripped parts so "president_abraham_lincoln" →
        # "abraham_lincoln" not "president_lincoln".
        j = 0
        while j < len(parts) - 1 and parts[j] in self._TITLE_PREFIXES:
            j += 1
        name_parts = parts[j:] if j > 0 else parts
        if len(name_parts) >= 3 and entity_type in self._COMPONENT_EXPANSION_TYPES:
            first_last = f"{name_parts[0]}_{name_parts[-1]}"
            if first_last not in forms:
                forms.append(first_last)

        # Individual name components — only for proper-name types.
        # Disabled by default: single-word components create cross-entity
        # collisions at ingest time. E.g., "lincoln" stored for
        # "abraham_lincoln" causes Blood libel article's "Hugh of Lincoln"
        # mention to resolve to Abraham Lincoln's entity hash.
        # Ambiguous single-word queries will be handled by the LLM
        # disambiguation layer instead.
        if bool(int(os.environ.get("NAM_ENTITY_COMPONENT_EXPANSION", "0"))):
            if len(parts) >= 2 and entity_type in self._COMPONENT_EXPANSION_TYPES:
                for part in parts:
                    if len(part) >= 3 and part not in forms and part not in self._COMPONENT_STOP_WORDS:
                        forms.append(part)

        return forms[:self.MAX_SURFACE_FORMS]

    @staticmethod
    def _record_candidates(text: str, spacy_doc, title: Optional[str]) -> List[EntityCandidate]:
        """The candidates of one text, the record title (if any) in front.

        With a title, the adapter's "<title>. " prefix contributes no
        candidates of its own: the title is one candidate."""
        skip = _title_prefix_tokens(spacy_doc, title) if title else 0
        candidates = extract_candidates(text, debug=False, spacy_doc=spacy_doc,
                                        skip_tokens=skip)
        return _with_title(candidates, title) if title else candidates

    def reset_session_state(self) -> Dict[str, int]:
        """Forget the head's own record of the session bucket after it is
        flushed (cold reset, src/storage/session_reset.py). That record is the
        entities whose ent:: doc this process already ensured, and the concept
        learner's maps. The store resets its caches itself."""
        ensured = getattr(self, "_ensured_entities", None)
        out = {"ensured": len(ensured) if ensured is not None else 0}
        if ensured is not None:
            ensured.clear()
        learner = getattr(self, "_concepts", None)
        if learner is not None:
            out.update({f"concept_{k}": v for k, v in learner.reset().items()})
        return out

    def _concept_learner(self) -> ConceptLearner:
        # Lazy: tests build the head with __new__.
        learner = getattr(self, "_concepts", None)
        if learner is None:
            learner = self._concepts = ConceptLearner()
        return learner

    def _concept_verdicts(self, texts: List[str], contexts: List[Optional[Dict]],
                          per_text: List[list], learn: bool) -> List[List[Optional[bool]]]:
        """The concept gate per prepared candidate (entity/concepts.py): None
        where it does not apply, True for a seed or learned concept, False
        otherwise. ``per_text[i]`` entries start (cand, canonical,
        entity_type, ...). With ``learn``, each learnable gated phrase outside
        the seed counts this record."""
        verdicts: List[List[Optional[bool]]] = [[None] * len(p) for p in per_text]
        if not gate_enabled():
            return verdicts
        observe: List[Tuple[str, str]] = []
        where: List[Tuple[int, int, str]] = []
        for i, prepared in enumerate(per_text):
            doc = (contexts[i] or {}).get("spacy_doc")
            starts = _sentence_starts(doc) if doc is not None else set()
            fp = None
            for ci, entry in enumerate(prepared):
                cand, canonical = entry[0], entry[1]
                if not needs_concept(cand, starts):
                    continue
                if in_seed(cand.text):
                    verdicts[i][ci] = True
                    continue
                if not learnable(cand, entry[2]):
                    verdicts[i][ci] = False
                    continue
                if fp is None:
                    fp = record_fingerprint(texts[i])
                observe.append((canonical, fp))
                where.append((i, ci, canonical))
        if observe:
            known = self._concept_learner().observe(
                self.store, observe, learn=learn and learning_enabled())
            for i, ci, canonical in where:
                verdicts[i][ci] = canonical in known
        return verdicts

    @staticmethod
    def _meaningful(cand: EntityCandidate, entity_type: str, concept: Optional[bool]) -> bool:
        return is_meaningful_compound(
            cand.normalized, cand.pos_pattern, cand.text,
            cand.token_count, cand.is_propn, entity_type=entity_type,
            is_name=cand.is_name, has_article=cand.has_article, concept=concept,
        )

    def _canonical_for(self, cand: EntityCandidate, lemma_cache: Optional[Dict[str, str]],
                       propn_tokens: Optional[Set[str]]) -> str:
        """The canonical form of a candidate: a name ending in a capitalised
        word or a number keeps its form ("angolan_armed_forces",
        "united_states", "alcoholics_anonymous"), anything else is
        singularised on its last word (_canonical_entity_id)."""
        if cand.is_name and _ends_capitalised(cand.text):
            return cand.normalized
        return self._canonical_entity_id(cand.normalized, lemma_cache, propn_tokens)

    def _canonical_entity_id(self, normalized: str, lemma_cache: Optional[Dict[str, str]] = None,
                             propn_tokens: Optional[Set[str]] = None) -> str:
        """
        Choose a stable, typically singular form for the entity ID.
        Proper nouns (PROPN) skip lemmatization to avoid mangling names
        (e.g., "paris" -> "pari", "united_states" -> "united_state").
        """
        parts = normalized.split("_")
        if not parts:
            return normalized

        def singularize(tok: str) -> str:
            if tok.endswith(_NOT_PLURAL_ENDINGS):
                return tok
            if tok.endswith("ies") and len(tok) > 3:
                return tok[:-3] + "y"
            if tok.endswith(("ses", "xes", "zes", "ches", "shes")) and len(tok) > 3:
                return tok[:-2]
            if tok.endswith("s") and len(tok) > 1:
                return tok[:-1]
            return tok

        propn_set = propn_tokens or set()

        # Skip lemmatization for proper nouns -- keep original form
        if parts[-1] in propn_set:
            return "_".join(parts)

        # A word ending in -ss/-us/-is is not a plural, and the rule
        # lemmatizer strips its "s" anyway (business -> "busines", campus ->
        # "campu", crisis -> "crisi"): keep it as written.
        if parts[-1].endswith(_NOT_PLURAL_ENDINGS):
            return "_".join(parts)

        # The lemma cache if it has the word, else the rule parser's lemma
        lemma_last = (lemma_cache.get(parts[-1]) if lemma_cache else None) or self._lemmatize(parts[-1]) or parts[-1]
        parts[-1] = singularize(lemma_last)
        return "_".join(parts)

    def _get_rule_nlp(self):
        """Lazy-init the rule NLP backend (singleton)."""
        if self._rule_nlp is None:
            self._rule_nlp = get_nlp_service(backend="rule", auto_start=True)
        return self._rule_nlp

    def _parse_text(self, text: str):
        """Parse text with rule NLP backend, returning a ParsedDocument."""
        try:
            return self._get_rule_nlp().parse(text)
        except Exception as e:
            logger.warning("EntityEncoder rule NLP parse failed (%s)", e)
            return None

    def _lemmatize(self, token: str) -> Optional[str]:
        try:
            doc = self._get_rule_nlp().parse(token)
            if doc and len(doc):
                return doc[0].lemma_.lower()
        except Exception:
            return None
        return None
