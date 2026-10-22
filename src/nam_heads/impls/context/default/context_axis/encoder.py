"""Context head on the particular-evidence extractor (prototype).

Drop-in for ``heads/context/head.py``'s ContextEncoder: same head type
("context"), same EncodingResult shape, and the same store
(``heads/context/storage``: session bucket, L1/LMDB caches; exposed as
``self.store`` so nats_worker's LMDB relay drains it, nats_worker.py:967).

Every call goes through ``encode_batch_texts``:
  1. one ``store.get_batch`` for every phrase the batch's docs may consult
     (``lexicon.candidate_queries``) -- one KV round-trip per batch;
  2. ``extract_context`` per doc against that lexicon;
  3. ingest only: observations, computed against the SEED part of the same
     lexicon, are folded into learned entries and written back with
     ``store.put`` (deferred, batched, relayed to LMDB by the store).
Query (``encode_query``) never writes.

Values carry the sentence indices they came from; the bundler scopes them
to entities (``bundler._context_sentence_map``) and applies the profile the
entity's own evidence gives it (``profiles.entity_profile`` / ``admit`` /
``select``: is it a place, is it the record's topic). Each named place also
yields its container (``extract.py``), metadata ``containers``.
"""

import logging
import os
from typing import Dict, List, Optional

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress
from .extract import DIMENSIONS, extract_context
from .lexicon import (Lexicon, MIN_WITNESSES, candidate_queries, learned_writes,
                      seed_tables, witness_id)

logger = logging.getLogger(__name__)
NULL = "__null__"


class ParticularContextEncoder(EncoderHead):
    HEAD_TYPE = "context"

    def _initialize(self):
        cfg = self.config or {}
        self.default_null = cfg.get("default_null", NULL)
        self.dimensions = tuple(cfg.get("dimensions", DIMENSIONS))
        # Per-sentence ontology steering at extraction time. Off by default:
        # the query side has only the record ontology, so steering would make
        # ingest and query extract differently for the same text; and the
        # 50-bundle hand check scored it slightly worse (72 vs 75 correct).
        # At bundling the entity's own evidence (is it a place, is it the
        # record's topic) conditions values (profiles.admit/select); the
        # record's class never removes a dimension.
        self.sentence_steering = bool(cfg.get("sentence_steering", False))
        # Values kept per dimension PER ENTITY (profiles.select). The current
        # head's knob meant per record; values are now scoped to entities, so
        # the cap applies where the rows are made. 0 = no cap.
        self.max_per_dimension = int(cfg.get("max_per_dimension", 1))
        # True: values scope to the sentences they occur in. False: every
        # value applies to the whole record (the current head's unsplit mode).
        self.split_fragments = bool(cfg.get("split_fragments", True))
        self.learn = bool(cfg.get("learn", True))
        self.min_witnesses = int(cfg.get("min_witnesses", MIN_WITNESSES))
        self.learned_written = 0
        self.store_lookups = 0
        self._batches = 0
        self._nlp = None
        store = cfg.get("store")
        if store is None:
            from ..storage.factory import create_context_store
            from .seed import seed_store
            store = create_context_store(
                backend=cfg.get("context_store_backend",
                                os.getenv("NAM_CONTEXT_STORE_BACKEND", "couchbase_kv")),
                fallback=cfg.get("context_store_fallback", "memory"),
                seed=False, kv_on_cache_miss=self.kv_on_cache_miss)
            # The supervisor seeds the store (services/supervisor/seeds.py); a
            # head reads the sealed seed from memory (lexicon.SeedTables), so
            # per-head seeding is opt-in.
            if cfg.get("seed_store", False):
                from src.storage.deferred_seed import run_seed
                run_seed("context-axis-store", lambda: seed_store(store))
        self.store = store
        # Decode the sealed seed now, not inside the first batch (~0.2 s).
        seed_tables()

    # --------------------------------------------------------------
    # Metrics (encoder-heads dashboard) and config
    # --------------------------------------------------------------
    def get_metrics(self) -> Dict:
        from .lexicon import seed_tables
        base = super().get_metrics()
        try:
            base["context_store_size"] = self.store.count()
        except Exception:
            base["context_store_size"] = -1
        base["context_seed_places"] = len(seed_tables().places)
        base["context_learned_writes"] = self.learned_written
        base["context_store_lookups"] = self.store_lookups
        base["context_lookups_per_batch"] = round(self.store_lookups / max(1, self._batches), 1)
        return base

    def config_summary(self) -> Dict:
        return {
            "dimensions": list(self.dimensions),
            "max_per_dimension": self.max_per_dimension,
            "split_fragments": self.split_fragments,
            "sentence_steering": self.sentence_steering,
            "learn": self.learn,
            "min_witnesses": self.min_witnesses,
            "context_store_backend": type(self.store).__name__,
        }

    def _doc(self, text: str, context: Optional[Dict]):
        doc = (context or {}).get("spacy_doc")
        if doc is None and text:
            if self._nlp is None:
                from src.nlp import get_nlp_service
                self._nlp = get_nlp_service()
            doc = self._nlp.parse(text)
        return doc

    def _addresses(self, values, n_sents: int = 0) -> List[SemanticAddress]:
        out = []
        for cv in values:
            meta = {"dimension": cv.dimension, "raw": cv.surface, "source": cv.source,
                    "sentences": list(cv.sentences) if self.split_fragments else list(range(max(1, n_sents))),
                    "spans": [list(sp) for sp in cv.spans],
                    "max_per_dimension": self.max_per_dimension}
            if cv.kind:
                meta["kind"] = cv.kind
            if cv.containers:
                meta["containers"] = list(cv.containers)
            if cv.demonym_spans:
                meta["demonym_spans"] = [list(sp) for sp in cv.demonym_spans]
            out.append(SemanticAddress(head_type=self.head_type, concept=cv.value,
                                       confidence=0.9, metadata=meta))
        if not out:
            out.append(SemanticAddress(head_type=self.head_type, concept=self.default_null,
                                       confidence=1.0,
                                       metadata={"dimension": "", "raw": "", "source": "null"}))
        return out

    def _run(self, texts: List[str], contexts, mode: str) -> List[EncodingResult]:
        contexts = contexts or [None] * len(texts)
        docs = [self._doc(t, c) for t, c in zip(texts, contexts)]
        from . import doc as D
        queries = set()
        parsed = []          # (tokens, sentences, entities) per doc, computed once
        for d in docs:
            toks = list(d) if d is not None else []
            ents = D.entities(d, toks) if d is not None else []
            parsed.append((toks, D.sentences(d, len(toks)) if toks else [], ents))
            if d is not None:
                gpe = {k for e in ents if e.label in ("GPE", "LOC", "FAC")
                       for k in range(e.start, e.end)}
                queries |= candidate_queries(toks, gpe)
        self._batches += 1
        self.store_lookups += len(queries)
        lex = Lexicon(self.store.get_batch(sorted(queries)) if queries else {},
                      min_witnesses=self.min_witnesses)
        learning = self.learn and mode == "ingest"
        seed_lex = lex.seed_only() if learning else None
        steer_for = None
        if self.sentence_steering:
            from .profiles import sentence_dimensions as steer_for
        results, observations = [], []
        for t, c, d, (toks, sents, ents) in zip(texts, contexts, docs, parsed):
            steer = steer_for((c or {}).get("ontology_result")) if steer_for else None
            # One pass when this doc can read no usable learned entry (none of
            # the batch's learned entries is reachable from its tokens) and
            # nothing restricts the dimensions: its extraction against lex IS
            # its extraction against the seed, so observations come from the
            # same pass.
            one_pass = (learning and steer is None
                        and tuple(self.dimensions) == tuple(DIMENSIONS)
                        and not lex.consults_learned(toks))
            obs: list = []
            values = extract_context(d, lex, self.dimensions, sentence_dimensions=steer,
                                     obs=obs if one_pass else None, toks=toks, sents=sents,
                                     ents=ents) if d is not None else []
            addrs = self._addresses(values, len(D.sentences(d, len(toks))) if d is not None else 0)
            results.append(EncodingResult(self.head_type, addrs, {
                "mode": mode,
                "dimensions": len({a.metadata["dimension"] for a in addrs if a.concept != self.default_null}),
                "address_count": len(addrs)}))
            if learning and d is not None:
                if not one_pass:
                    extract_context(d, seed_lex, DIMENSIONS, obs=obs, toks=toks, sents=sents, ents=ents)
                w = witness_id(t or "")
                observations.extend((k, s, v, w) for k, s, v in obs)
        if learning and observations:
            for phrase, dim, payload, conf in learned_writes(observations, dict(lex.learned_docs)):
                self.store.put(phrase, dim, payload, confidence=conf)
                self.learned_written += 1
        return results

    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._run([text], [context], "ingest")[0]

    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        return self._run([query], [context], "query")[0]

    def encode_batch_texts(self, texts: List[str],
                           contexts: Optional[List[Optional[Dict]]] = None) -> List[EncodingResult]:
        return self._run(texts, contexts, "ingest")
