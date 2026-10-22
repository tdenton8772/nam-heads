"""The NLP plugin contract (slot ``nlp``, contract ``parser-v1``).

NLP is not a head, but it is loaded the same way: one plugin package fills the ``nlp`` slot, and every
other head reads its parse in ``context["spacy_doc"]`` (and through ``host.parser()``). An implementation
defines exactly one ``Parser`` subclass in its ``head.py``, with a literal ``IMPL_META`` like any head.

A parse is a document with ``.sents`` (token spans with ``.ents``), indexable tokens with ``.text``,
``.lemma_``, ``.pos_``, ``.tag_``, ``.dep_`` and ``.head`` -- the shape NAM's rule parser produces and the
heads read.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class Parser(ABC):
    """Base class of an NLP plugin. The host builds it as ``cls(config)``."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(config or {})
        self._initialize()

    def _initialize(self) -> None:
        """Load the parser's tables and models."""

    @abstractmethod
    def parse(self, text: str) -> Any:
        """The parse of one text."""

    def parse_batch(self, texts: Sequence[str]) -> List[Any]:
        """Parses of several texts, in order (override when batching is cheaper)."""
        return [self.parse(t) for t in texts]

    def fingerprint(self) -> str:
        """A stable digest of the parser's behaviour (e.g. its output on a probe set); heads trained on one
        parser compare it to warn when the parser changed. Empty when the plugin does not provide one."""
        return ""
