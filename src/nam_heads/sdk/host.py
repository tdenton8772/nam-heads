"""What a head borrows from the host that runs it.

The heads in this package run inside NAM, which owns the parser, the axis
encoders and the CPU budget. They reach those through the hooks here, never
by importing NAM, so the package installs and tests on its own:

- ``parser()``: the NLP backend whose parse a head reads from
  ``context["spacy_doc"]``. Heads never parse at encode time; training,
  calibration and the codebook's parser fingerprint do. NAM registers its
  rule backend. Without a host, ``parser()`` raises ``HostUnavailable`` and
  the tests that need it skip.
- ``axis_encoder(slot)``: the host's built-in encoder for an axis slot
  (None for anchor and facet slots, and without a host).
- ``ort_session_options(max_threads)``: onnxruntime options sized for this
  process. The host's version reads the container's CPU limit; the fallback
  uses ``os.cpu_count()``.
- ``open_store(spec, slot, impl, mode)``: a head store handle
  (``nam_heads.sdk.store``). The host's backs it with its session bucket and
  node caches; the fallback is an in-memory store, shared by scope within the
  process, writable unless ``mode`` is "query".
"""

import os
from typing import Any, Callable, Optional


class HostUnavailable(RuntimeError):
    """A hook was called that no host has registered."""


_parser_factory: Optional[Callable[[], Any]] = None
_parser: Any = None
_axis_encoder_factory: Optional[Callable[[str], Any]] = None
_ort_options: Optional[Callable[..., Any]] = None


def set_parser_factory(factory: Optional[Callable[[], Any]]) -> None:
    """``factory()`` returns an object with ``parse(text) -> doc``; ``doc.sents``
    yields token spans with ``.ents``. Built once, on first use."""
    global _parser_factory, _parser
    _parser_factory, _parser = factory, None


def has_parser() -> bool:
    return _parser_factory is not None


def parser():
    global _parser
    if _parser_factory is None:
        raise HostUnavailable(
            "no parser registered: run inside NAM, or call "
            "nam_heads.sdk.host.set_parser_factory(...) first")
    if _parser is None:
        _parser = _parser_factory()
    return _parser


def set_axis_encoder_factory(factory: Optional[Callable[[str], Any]]) -> None:
    global _axis_encoder_factory
    _axis_encoder_factory = factory


def axis_encoder(slot: str):
    return _axis_encoder_factory(slot) if _axis_encoder_factory is not None else None


def set_ort_session_options(fn: Optional[Callable[..., Any]]) -> None:
    global _ort_options
    _ort_options = fn


def ort_session_options(max_threads: Optional[int] = None):
    """onnxruntime SessionOptions: intra-op threads capped at ``max_threads``,
    inter-op 1."""
    if _ort_options is not None:
        return _ort_options(max_threads=max_threads)
    import onnxruntime as ort

    n = os.cpu_count() or 1
    if max_threads is not None:
        n = max(1, min(n, int(max_threads)))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n
    opts.inter_op_num_threads = 1
    return opts


_store_factory: Optional[Callable[..., Any]] = None
_memory_stores: dict = {}


def set_store_factory(factory: Optional[Callable[..., Any]]) -> None:
    """``factory(spec, slot, impl, mode) -> HeadStore``."""
    global _store_factory
    _store_factory = factory


def open_store(spec, slot: str, impl: str, mode: str = "ingest"):
    """A handle on one declared store for an implementation of ``slot``."""
    if _store_factory is not None:
        return _store_factory(spec, slot, impl, mode)
    from .store import MemoryStore

    key = (slot, spec.name) if spec.scope == "slot" else (slot, impl, spec.name)
    data = _memory_stores.setdefault(key, {})
    return MemoryStore(spec, writable=(mode != "query"), shared=data)


def reset_memory_stores() -> None:
    """Drop the fallback stores (tests)."""
    _memory_stores.clear()
