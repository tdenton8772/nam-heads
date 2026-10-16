"""The contract checks every implementation must pass, usable from pytest.

``tests/test_conformance.py`` runs them over every installed implementation;
an implementation's own tests can call them on a head built with test config.

    from nam_heads.sdk import testing as T
    head = T.build(T.find("ontology", "gaius_ref"))
    assert T.check_ontology_head(head, T.SAMPLE_TEXTS) == []
"""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import host
from .head import EncoderHead, EncodingResult
from .impl_meta import (InstalledImpl, installed, missing_assets, missing_packages,
                        read_impl_meta)
from .sentence_ontology import attach_sentence_ontologies

SAMPLE_TEXTS = [
    "Albert_Einstein. Albert Einstein was a German-born theoretical physicist. He received the Nobel Prize.",
    "Lundu is a town and the capital of Lundu District in the northwest of Kuching Division of Sarawak, Malaysia.",
    "The Treaty of Versailles was signed in 1919 at the end of the First World War.",
    "Aspirin is a medication used to reduce pain, fever, or inflammation.",
    "",
]
SAMPLE_QUERIES = ["who discovered relativity", "capital of Sarawak", "when was the treaty signed"]


def parser_or_skip():
    """The host parser, or skip the calling test when none is registered."""
    import pytest

    if not host.has_parser():
        pytest.skip("needs a host parser (run inside NAM, or register one with "
                    "nam_heads.sdk.host.set_parser_factory)")
    return host.parser()


def find(slot: str, name: str) -> InstalledImpl:
    for impl in installed():
        if (impl.slot, impl.name) == (slot, name):
            return impl
    raise LookupError(f"{slot}.{name} is not installed (pip install -e . registers this repo's)")


def unavailable(impl: InstalledImpl) -> List[str]:
    """What this process lacks to run ``impl``: packages, assets, or a bad IMPL_META."""
    meta, problems = read_impl_meta(impl.head)
    if meta is None or problems:
        return problems or ["no IMPL_META"]
    return list(missing_packages(meta.get("requires") or ()) + missing_assets(meta.get("assets") or ()))


def load_class(impl: InstalledImpl):
    """The one EncoderHead subclass defined in ``impl``'s head.py."""
    mod = importlib.import_module(impl.module)
    classes = [obj for _, obj in inspect.getmembers(mod, inspect.isclass)
               if issubclass(obj, EncoderHead) and obj is not EncoderHead
               and obj.__module__ == mod.__name__]
    if len(classes) != 1:
        raise AssertionError(f"{impl.module}: expected one EncoderHead subclass, found "
                             f"{[c.__name__ for c in classes]}")
    return classes[0]


def apply_ontology(impl: InstalledImpl) -> Optional[str]:
    """What the host does when the manifest selects ``impl``: make the class
    registry carry the implementation's own classes (IMPL_META "ontology"),
    or only BFO/CCO."""
    from nam_heads.ontology import registry as R

    meta, _ = read_impl_meta(impl.head)
    ont = (meta or {}).get("ontology")
    return R.use_extension(str((impl.head.parent / ont["artifact"]).resolve()) if ont else None)


def build(impl: InstalledImpl, config: Optional[Dict[str, Any]] = None) -> EncoderHead:
    """Build ``impl`` as the host does (``head_type`` = its slot), after
    applying its ontology."""
    if impl.slot == "ontology":
        apply_ontology(impl)
    cls = load_class(impl)
    return cls(head_type=impl.slot, config=config) if config is not None else cls(head_type=impl.slot)


def _check_result(res: Any, where: str, known) -> List[str]:
    out = []
    if not isinstance(res, EncodingResult):
        return [f"{where}: returned {type(res).__name__}, not EncodingResult"]
    if res.head_type != "ontology":
        out.append(f"{where}: head_type {res.head_type!r} != 'ontology'")
    if not isinstance(res.metadata, dict):
        out.append(f"{where}: metadata is not a dict")
    if not res.addresses:
        out.append(f"{where}: no addresses (abstain with bfo:entity instead)")
    for i, a in enumerate(res.addresses):
        if a.head_type != "ontology":
            out.append(f"{where}: address {i} head_type {a.head_type!r}")
        if not known(a.concept):
            out.append(f"{where}: address {i} class {a.concept!r} is not in the registry")
        if not (0.0 <= float(a.confidence) <= 1.0):
            out.append(f"{where}: address {i} confidence {a.confidence} outside [0, 1]")
    so = res.metadata.get("sentence_ontologies") if isinstance(res.metadata, dict) else None
    if so is not None:
        if not isinstance(so, dict):
            out.append(f"{where}: sentence_ontologies is not a dict")
        else:
            for k, v in so.items():
                if not isinstance(k, int) or not known(v):
                    out.append(f"{where}: sentence_ontologies {k!r} -> {v!r}")
    return out


def check_ontology_head(head: EncoderHead, texts: Iterable[str] = SAMPLE_TEXTS,
                        queries: Iterable[str] = SAMPLE_QUERIES,
                        parse=None) -> List[str]:
    """Problems with ``head`` as an ``ontology-v1`` implementation; [] = conforms.

    - ``encode`` / ``encode_query`` return an EncodingResult for the
      ``ontology`` head with at least one address, every class in the
      registry (BFO/CCO plus the implementation's own), confidence in [0, 1];
    - ``sentence_ontologies``, when published, maps sentence index -> class;
    - ``emittable_classes()`` lists only registry classes;
    - ``classify_nearest`` returns a registry class or None;
    - with no parse in ``context`` the head still answers (it may abstain).

    ``parse`` (text -> doc) supplies ``context["spacy_doc"]``; the host parser
    is used when one is registered and ``parse`` is None.
    """
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    known = reg.is_class
    if parse is None and host.has_parser():
        parse = host.parser().parse
    out: List[str] = []
    for fn in ("encode", "encode_query", "classify_nearest", "emittable_classes"):
        if not callable(getattr(head, fn, None)):
            out.append(f"no {fn}()")
    if out:
        return out
    emittable = list(head.emittable_classes())
    if not emittable:
        out.append("emittable_classes() is empty")
    out += [f"emittable class {c!r} is not in the registry" for c in emittable if not known(c)]
    for t in texts:
        out += _check_result(head.encode(t, {}), f"encode({t[:30]!r}, no parse)", known)
        if parse is not None:
            doc = parse(t)
            res = head.encode(t, {"spacy_doc": doc})
            out += _check_result(res, f"encode({t[:30]!r})", known)
            attach_sentence_ontologies(head, doc, res)
            out += _check_result(res, f"encode({t[:30]!r}) + attach", known)
    for q in queries:
        ctx = {"spacy_doc": parse(q)} if parse is not None else {}
        out += _check_result(head.encode_query(q, ctx), f"encode_query({q!r})", known)
        c = head.classify_nearest(q, spacy_doc=ctx.get("spacy_doc"))
        if c is not None and not known(c):
            out.append(f"classify_nearest({q!r}) -> {c!r}, not in the registry")
    return out
