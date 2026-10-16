"""The classes the active ontology head can write into an address key.

Readers that must enumerate type partitions (the graph index's nam-bucket
fallback, scripts/rebuild_index.py, the query planner's tiers) ask here instead
of keeping their own list, so swapping the ontology head changes them all.

The head-slot manifest can select another ontology implementation
(NAM's docs/design-head-slots.md). Its classes are enumerated alongside the built-in
codebook's in two ways:

- a process that holds the selected head registers it with ``use_head``;
- a process whose registry carries the selected implementation's ontology
  (``registry.use_extension``, applied by every manifest reader) enumerates
  that ontology's emittable classes without holding the head.

The codebook's always stay in: records placed before a swap keep its classes
in their keys until a restream re-lands them.
"""

from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

from .registry import get_registry

_head_classes: Tuple[str, ...] = ()


def use_head(head) -> None:
    """Also enumerate the classes ``head`` (the selected ontology
    implementation) can emit. None, or a head without emittable_classes,
    leaves only the codebook's."""
    global _head_classes
    fn = getattr(head, "emittable_classes", None)
    _head_classes = tuple(fn()) if callable(fn) else ()
    _emittable.cache_clear()


def emittable_classes() -> List[str]:
    """Trained classes of the built-in head's codebook, every ancestor they
    can back off to, bfo:entity, whatever the registered head emits, and the
    active registry extension's emittable classes."""
    ext = get_registry().extension
    return list(_emittable(ext["name"] if ext else None))


# Callers clear the enumeration with emittable_classes.cache_clear().
emittable_classes.cache_clear = lambda: _emittable.cache_clear()  # type: ignore[attr-defined]


@lru_cache(maxsize=4)
def _emittable(extension: Optional[str]) -> Tuple[str, ...]:
    from nam_heads.impls.ontology.codebook.codebook import Codebook

    reg = get_registry()
    out = set(_head_classes)
    if extension and reg.extension and reg.extension["name"] == extension:
        out |= set(reg.extension["emittable"])
    for c in Codebook.load(check_parser=False).classes:
        out |= reg.ancestors(c, inclusive=True)
    return tuple(sorted(out))


def union(classes: Iterable[str]) -> List[str]:
    """``classes`` plus every enumerable class, sorted."""
    return sorted(set(classes) | set(emittable_classes()))
