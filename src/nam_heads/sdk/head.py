"""The head interface: what every implementation subclasses and returns.

A host (NAM's ingest workers and query service) builds an implementation as
``cls(head_type=<slot>, config=<dict or None>)`` and calls ``encode`` on each
record and ``encode_query`` on each query. The ``context`` dict carries what
the host already computed: ``spacy_doc`` (the shared parse), and for heads
after the ontology slot, ``ontology_result``.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from . import host


@dataclass
class SemanticAddress:
    """
    A semantic address pointing to records in a specific address space.
    """
    head_type: str  # e.g., "entity", "attribute", "custom_sentiment"
    concept: str    # e.g., "ball", "red", "positive"
    confidence: float = 1.0
    metadata: Optional[Dict] = None
    # The head's own encoding of ``concept`` on its address axis
    # ("lca:<coarse>:<fine>"), stamped by encoders.axis_encoder.stamp_codes.
    # None for anchor and facet heads, and before stamping.
    code: Optional[str] = None

    def __str__(self) -> str:
        return f"addr_{self.head_type}_{self.concept}"


@dataclass
class EncodingResult:
    """Result of encoding through a single encoder head."""
    head_type: str
    addresses: List[SemanticAddress]
    metadata: Dict


class EncoderHead(ABC):
    """Abstract base class for every head implementation.

    An implementation lives in ``<package>/impls/<slot>/<name>/head.py`` with
    exactly one ``EncoderHead`` subclass and a literal ``IMPL_META``
    (``nam_heads.sdk.impl_meta``), and is registered under the ``nam.heads``
    entry-point group. Subclasses implement ``_initialize``, ``encode`` and
    ``encode_query``; ontology implementations also ``classify_nearest`` and
    ``emittable_classes`` (``nam_heads.sdk.testing`` checks the contract).
    """

    def __init__(self, head_type: str, config: Optional[Dict] = None,
                 stores: Optional[Dict[str, Any]] = None):
        self.head_type = head_type
        self.config = config or {}
        # Head stores by name (IMPL_META "stores"): handed in by the host, or
        # opened through host.open_store on first use.
        self._stores: Dict[str, Any] = dict(stores or {})
        self._records_processed = 0
        self._errors = 0
        self._total_encode_ms = 0.0
        # Per-head cache miss policy: if False (default), LMDB miss → return None
        # (head infers). If True, LMDB miss → KV lookup → return None (head infers).
        # Env var: NAM_{HEAD_TYPE}_KV_ON_MISS=1
        env_key = f"NAM_{head_type.upper()}_KV_ON_MISS"
        self.kv_on_cache_miss = self.config.get(
            "kv_on_cache_miss",
            os.getenv(env_key, "0").lower() in ("1", "true", "yes"),
        )
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize encoder-specific components."""
        pass
    
    @abstractmethod
    def encode(self, text: str, context: Optional[Dict] = None) -> EncodingResult:
        """Encode text into semantic addresses."""
        pass
    
    @abstractmethod
    def encode_query(self, query: str, context: Optional[Dict] = None) -> EncodingResult:
        """Encode query into semantic addresses for retrieval."""
        pass
    
    def create_axis_encoder(self):
        """This head's own encoder: value -> address coordinate on its axis.

        Axis heads (attribute, affordance, context) get the host's built-in
        per-axis encoder (``host.set_axis_encoder_factory``); anchor and facet
        heads have none. An implementation that ships another encoder
        overrides this.
        """
        return host.axis_encoder(self.head_type)

    @property
    def axis_encoder(self):
        """The head's encoder, created on first use (None off the axes)."""
        if "_axis_encoder" not in self.__dict__:
            self.__dict__["_axis_encoder"] = self.create_axis_encoder()
        return self.__dict__["_axis_encoder"]

    def store(self, name: str):
        """The handle of one store this implementation declares in
        ``IMPL_META["stores"]`` (nam_heads.sdk.store)."""
        if name not in self._stores:
            from .store import parse_specs
            import sys

            meta = getattr(sys.modules.get(type(self).__module__), "IMPL_META", None) or {}
            specs = {sp.name: sp for sp in parse_specs(meta.get("stores"))}
            if name not in specs:
                raise KeyError(f"{type(self).__name__} declares no store {name!r} "
                               f"(IMPL_META stores: {sorted(specs)})")
            mode = self.config.get("store_mode", "ingest")
            self._stores[name] = host.open_store(specs[name], self.head_type,
                                                 meta.get("name", type(self).__name__), mode)
        return self._stores[name]

    def get_metrics(self) -> Dict:
        """Return observable metrics."""
        avg_ms = (self._total_encode_ms / self._records_processed) if self._records_processed > 0 else 0.0
        return {
            "head_type": self.head_type,
            "records_processed": self._records_processed,
            "errors": self._errors,
            "avg_encode_ms": round(avg_ms, 2),
        }

    def config_summary(self) -> Dict:
        """Override per head to return key config params."""
        return {}
