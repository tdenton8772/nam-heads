"""Ontology-keyed table lookup shared by the heads.

Every head keeps tables keyed by BFO/CCO class (``ontology.bindings``
constants: ONTOLOGY_RELEVANCE, FIREWALL_ONTOLOGY_EXEMPTIONS,
ONTOLOGY_FUNCTION_VERBS, EVIDENCE_EXCLUSIONS, ...) and the union of their keys
as ``ONTOLOGY_KEYS``. ``lookup`` is a thin wrapper over
``registry.nearest_key(ontology, keys)``, the same resolution the heads'
``_get_ontology_type`` uses:

* the ontology is a class CURIE (the ontology head emits nothing else);
* it resolves to the deepest of itself and its ancestors that is in ``keys``
  (the head's ONTOLOGY_KEYS; the table's own keys when none are given) --
  never inheriting from bfo:entity;
* ``bfo:entity`` (the head abstaining), ``None``, a non-class string or a
  class with no keyed ancestor give ``default``: no ontology evidence is
  neutral, never a match.
"""

from typing import Any, Iterable, Optional

from . import bindings as B


def _registry():
    try:
        from .registry import get_registry
        return get_registry()
    except Exception:
        return None


def ontology_class(ontology: Optional[str]) -> Optional[str]:
    """The class CURIE ``ontology`` names, or None (no ontology evidence).

    ``bfo:entity`` is the head's abstention, so it is None too."""
    reg = _registry()
    if not ontology or reg is None:
        return None
    o = str(ontology)
    if o == B.UNCLASSIFIED or not reg.is_class(o):
        return None
    return o


def nearest(ontology: Optional[str], keys: Iterable[str]) -> Optional[str]:
    """The key ``ontology`` resolves to among ``keys`` (see module doc)."""
    cls = ontology_class(ontology)
    if cls is None:
        return None
    reg = _registry()
    keys = keys if isinstance(keys, (set, frozenset, dict)) else frozenset(keys)
    hit = reg.nearest_key(cls, keys)
    return None if hit == B.UNCLASSIFIED else hit


def lookup(table: dict, ontology: Optional[str], default: Any = None,
           keys: Optional[Iterable[str]] = None) -> Any:
    """``table``'s entry for ``ontology``, else ``default``.

    ``keys`` is the head's ONTOLOGY_KEYS: resolution stops at the nearest
    class ANY of the head's tables is keyed by, and a table without that key
    gives ``default`` (as ``_get_ontology_type`` + ``table.get`` would)."""
    hit = nearest(ontology, table if keys is None else keys)
    if hit is None:
        return default
    return table.get(hit, default)


# Name used by the attribute head (same function, same arguments).
ontology_lookup = lookup
