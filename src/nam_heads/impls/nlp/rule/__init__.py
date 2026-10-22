"""
Rule-based NLP backend.

Fully deterministic, no model weights, no external dependencies.
Activated via NAM_NLP_BACKEND=rule.

Six-stage pipeline:
    Text -> Tokenizer -> POS Tagger -> Lemmatizer -> NER -> Dep Parser -> ParsedDocument
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.nlp.interfaces import (
    Entity,
    NLPService,
    ParsedDocument,
    Sentence,
    Token,
)

logger = logging.getLogger(__name__)


class RuleNLPService(NLPService):
    """
    Rule-based NLP service.

    Pure dict lookups and regex patterns -- no model inference.
    Deterministic: same input always produces same output.
    """

    def __init__(self, **config):
        data_dir = os.path.join(os.path.dirname(__file__), "data")

        from .tokenizer import Tokenizer
        from .pos_tagger import POSTagger
        from .lemmatizer import Lemmatizer
        from .ner import NERRecognizer
        from .dep_parser import DependencyParser

        self._tokenizer = Tokenizer()
        self._pos_tagger = POSTagger(data_dir)
        self._lemmatizer = Lemmatizer(data_dir)
        self._ner = NERRecognizer(data_dir)
        self._dep_parser = DependencyParser()
        self._running = False
        self._records_processed = 0
        self._errors = 0
        self._total_parse_ms = 0.0

    def parse(self, text: str, title: str = "") -> Optional[ParsedDocument]:
        """Parse ``text``; the document carries the sentence sketch's skeleton
        (``doc.skeleton``, NLP1: src/nlp/sketch), built after the dependency
        parse from the parse and ``title`` (the record title; "" for a query).
        The skeleton is an additional layer: no token, tag, entity or
        sentence of the parse changes."""
        if not text or not text.strip():
            doc = ParsedDocument(text=text or "", tokens=[], entities=[], sentences=[])
            doc.skeleton = self._skeleton(doc, title)
            return doc
        try:
            raw_tokens, sentences = self._tokenizer.tokenize(text)
            self._pos_tagger.tag(raw_tokens, sentences)
            self._lemmatizer.lemmatize(raw_tokens)
            entities = self._ner.recognize(raw_tokens, sentences)
            self._dep_parser.parse(raw_tokens, sentences)
            doc = self._assemble(text, raw_tokens, entities, sentences)
        except Exception:
            logger.exception("rule_nlp.parse_error")
            return None
        doc.skeleton = self._skeleton(doc, title)
        return doc

    @staticmethod
    def _skeleton(doc: ParsedDocument, title: str):
        """NLP1. A failure leaves an empty skeleton and is logged: the parse
        itself is never lost to it."""
        from src.nlp.sketch import Skeleton, build_skeleton
        try:
            return build_skeleton(doc, title or "")
        except Exception:
            logger.exception("rule_nlp.skeleton_error")
            return Skeleton.empty()

    def parse_batch(self, texts: List[str]) -> List[Optional[ParsedDocument]]:
        return [self.parse(t) for t in texts]

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def backend_name(self) -> str:
        return "rule"

    def get_metrics(self) -> dict:
        avg_ms = (self._total_parse_ms / self._records_processed) if self._records_processed > 0 else 0.0
        return {
            "head_type": "nlp",
            "backend": "rule",
            "running": self.is_running,
            "records_processed": self._records_processed,
            "errors": self._errors,
            "avg_parse_ms": round(avg_ms, 2),
        }

    def config_summary(self) -> dict:
        return {"backend": "rule", "deterministic": True}

    def _assemble(
        self,
        text: str,
        raw_tokens: list,
        entities: list,
        sent_spans: list,
    ) -> ParsedDocument:
        """Convert internal pipeline objects to ParsedDocument."""
        tokens = []
        for i, rt in enumerate(raw_tokens):
            tokens.append(Token(
                text=rt.text,
                lemma=rt.lemma,
                pos=rt.pos,
                tag=rt.tag,
                dep=rt.dep,
                ent_type=rt.ent_type,
                ent_iob=rt.ent_iob,
                head_idx=rt.head_idx,
                idx=i,
                is_stop=rt.is_stop,
                is_punct=rt.is_punct,
                is_space=rt.is_space,
                like_num=rt.like_num,
                morph_data=rt.morph_data,
            ))

        ents = [
            Entity(
                text=e["text"],
                label=e["label"],
                start_token=e["start"],
                end_token=e["end"],
            )
            for e in entities
        ]

        sents = [
            Sentence(
                start_token=s["start"],
                end_token=s["end"],
                text=s["text"],
            )
            for s in sent_spans
        ]

        return ParsedDocument(
            text=text,
            tokens=tokens,
            entities=ents,
            sentences=sents,
        )
