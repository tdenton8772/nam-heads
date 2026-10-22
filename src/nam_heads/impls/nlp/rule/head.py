"""The rule parser as the default NLP plugin (slot ``nlp``, contract ``parser-v1``;
docs/design-head-plugins.md). NLP is not a head, but it is loaded the same way: the NLP stage and every
head's ``host.parser()`` use the parser the head-slot manifest selects. This package is the default one.
"""
from nam_heads.sdk.parser import Parser

IMPL_META = {
    "slot": "nlp",
    "name": "rule",
    "version": "1.0.1",
    "contract": "parser-v1",
    "description": "NAM's rule-based parser: tokenizer, POS tagger, lemmatizer, NER and dependency parser",
    # The parse every head reads; a different parser writes a different parse.
    "output": "rule-parse-v1",
    # ingest.nlpResources.limits (cpu "1", memory 1Gi) and ingest.nlpWorkers 2 on the reference deployment.
    "resources": {"cpu": 1.0, "memory_mb": 1024, "workers": 2},
    "dependencies": [],
}


class RuleParser(Parser):
    backend_name = "rule"

    def _initialize(self) -> None:
        from . import RuleNLPService
        self._svc = RuleNLPService()

    def parse(self, text, title: str = ""):
        return self._svc.parse(text, title=title)

    def parse_batch(self, texts):
        return self._svc.parse_batch(texts)

    def fingerprint(self) -> str:
        return ""
