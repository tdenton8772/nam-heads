"""Context profiles: what an entity's OWN evidence says about which context
mentions apply to it. A profile never becomes a value.

The record's ontology class does not choose the profile. The ontology head
types every entity of a record with the record's class, so a class-keyed
profile removed whole dimensions from entities the class does not describe
("Apollo 15" lost d:spaceflight under the record class Act; "Croatia" lost
d:politics under Geospatial Region). The bundler instead reads two pieces of
the entity's own evidence (``entity_profile``):

* is the entity a PLACE -- does a geographic mention cover the head token of
  one of its mentions (``bundler._context_own_places``)? A place never takes
  itself, a place inside it, or a sibling of it (``place_conflict``), nor a
  demonym (that describes its people); its own container -- the parent of its
  gazetteer path, which the head emits as a value carrying ``containers`` --
  is its geographic context;
* is the entity the record's TOPIC (its words are the record title's)? A place
  that is not the topic keeps only where it is (``PLACE_ONLY``): the events of
  a sentence that names it are not its context. With no title (a query) the
  topic is unknown and no dimension is removed.

The class-keyed table (``PROFILES``) remains for per-sentence extraction
steering (``sentence_dimensions``, off by default).

Rules common to every profile:
* a mention inside the entity's own tokens is not context for it
  ("Chicago" is not the context of the entity Chicago);
* a value must come from a sentence the entity occurs in (the bundler's scoping);
* per dimension the most specific candidate is kept: a value whose more
  specific version is also a candidate is dropped before the nearest is kept
  ("American" gives way to "Los Angeles").
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import FrozenSet, Iterable, Optional, Sequence, Set, Union

from .extract import DOMAIN, GEOGRAPHIC, INSTITUTIONAL, TEMPORAL

ALL = frozenset({TEMPORAL, GEOGRAPHIC, INSTITUTIONAL, DOMAIN})

from nam_heads.ontology import bindings as B

# Site (BFO 2020) has no constant in ontology/bindings.py (owned by the
# ontology session); the ontology head can emit it as a back-off ancestor of
# Geospatial Region, so the place profile is keyed at both.
SITE = "BFO:0000029"


@dataclass(frozen=True)
class Profile:
    name: str
    dimensions: FrozenSet[str]
    # Demonyms ("American") count as geographic evidence: nationality for a
    # person, origin for a work; for a place they describe its people, not
    # its location, and are not context of it.
    demonyms: bool = True
    # Values kept per dimension, nearest mention first. Each kept value is an
    # xyz row (builder zips the axes), so this bounds the row fan-out.
    per_dimension: int = 1


# Sentence-steering table, keyed by class (an entity's class uses its nearest
# keyed class: itself or an ancestor, registry.nearest_key -- never
# bfo:entity). Used only by ``sentence_dimensions``; bundling never reads it.
_PLACE = Profile("place", frozenset({TEMPORAL, GEOGRAPHIC}), demonyms=False)  # containing region, period
PROFILES = {
    B.PERSON: Profile("person", ALL),                       # birth/death/residence, nationality, institutions, era, field
    B.GROUP_OF_PERSONS: Profile("group", frozenset({TEMPORAL, GEOGRAPHIC, DOMAIN})),
    B.ORGANIZATION: Profile("organization", ALL),           # where based, founding period, parent, sector
    B.PROCESS: Profile("process", frozenset({TEMPORAL, GEOGRAPHIC, INSTITUTIONAL})),  # when, where, who ran it
    B.GEOSPATIAL_REGION: _PLACE,
    SITE: _PLACE,
    B.TEMPORAL_REGION: Profile("time", frozenset({GEOGRAPHIC, INSTITUTIONAL, DOMAIN})),
    B.MATERIAL_ARTIFACT: Profile("artifact", ALL),
    B.INFORMATION_CONTENT_ENTITY: Profile("information", ALL),
    B.QUALITY: Profile("quality", frozenset({DOMAIN, TEMPORAL})),
}
ONTOLOGY_KEYS = frozenset(PROFILES)
DEFAULT = Profile("default", ALL)

# Evidence profiles (bundling).
PLACE = Profile("place", ALL, demonyms=False)                       # a place that is the topic (or topic unknown)
PLACE_ONLY = Profile("place_only", frozenset({GEOGRAPHIC}), demonyms=False)  # a place that is not the topic


def entity_profile(own_places: Iterable[str] = (), topic: Optional[bool] = None) -> Profile:
    """The profile the entity's own evidence gives it: ``own_places`` are the
    geographic values naming the entity itself; ``topic`` is True / False when
    the record title says whether the entity is what the record is about,
    None when there is no title."""
    if not own_places:
        return DEFAULT
    return PLACE_ONLY if topic is False else PLACE


# ---- places -----------------------------------------------------------------
def _seg_kind(seg: str) -> str:
    """Kind of one path segment (data/generate_places.py): a multi-country
    region ("r-") or historical country ("x-") overlaps others; a city ("gn",
    minted "n-"); otherwise an administrative unit (continent, country,
    US/GB subdivision, "s-" region)."""
    if seg.startswith("r-") or seg.startswith("x-"):
        return "overlap"
    if seg.startswith("gn") or seg.startswith("n-"):
        return "city"
    return "unit"


def place_conflict(value: str, own: str) -> bool:
    """True if geographic ``value`` cannot be the place context of an entity
    whose own place is ``own`` (both "g:" values or both bare paths): the same
    place, a place inside it, or a sibling -- the paths part at a level where
    both segments are the same kind of unit (Illinois/Utah, Afghanistan/Iran,
    Cornwall/Essex). A container of it is not a conflict; a region or
    historical country beside it may contain it; a city beside a county may
    lie in it. When the entity is itself a region or historical country, a
    unit or city beside it may lie inside it (Botswana for Southern Africa,
    the path scheme does not nest them): a conflict."""
    v = value[2:] if value.startswith("g:") else value
    o = own[2:] if own.startswith("g:") else own
    if v == o or v.startswith(o + "/"):
        return True
    if o.startswith(v + "/"):
        return False
    pv, po = v.split("/"), o.split("/")
    k = 0
    while k < min(len(pv), len(po)) and pv[k] == po[k]:
        k += 1
    if k >= len(pv) or k >= len(po):
        return False
    a, b = _seg_kind(pv[k]), _seg_kind(po[k])
    if b == "overlap":
        return a != "overlap"
    return a == b and a != "overlap"


def container_of(value: str) -> Optional[str]:
    """The parent of a geographic value's path ("g:EU/GB/GB-ENG/s-cornwall"
    -> "g:EU/GB/GB-ENG"); None for a top-level path."""
    if not value.startswith("g:") or "/" not in value:
        return None
    return value.rsplit("/", 1)[0]


@lru_cache(maxsize=1)
def _registry():
    from nam_heads.ontology.registry import get_registry
    return get_registry()


@lru_cache(maxsize=4096)
def class_of(type_value: Optional[str]) -> Optional[str]:
    """The entity's class CURIE, or None for no ontology evidence (None, a
    non-class string, or bfo:entity -- the ontology head abstaining)."""
    from nam_heads.ontology.lookup import ontology_class
    return ontology_class(type_value)


@lru_cache(maxsize=4096)
def profile_for(type_value: Optional[str]) -> Profile:
    from nam_heads.ontology.lookup import lookup
    return lookup(PROFILES, type_value, DEFAULT, keys=ONTOLOGY_KEYS)


def _profile(profile: Union[Profile, str, None]) -> Profile:
    """A Profile as is; anything else (a class CURIE, None) is not evidence
    about the entity and gives DEFAULT -- a class never removes a dimension."""
    return profile if isinstance(profile, Profile) else DEFAULT


def admit(profile: Union[Profile, str, None], meta: dict, entity_tokens: Iterable[int],
          own_places: Iterable[str] = (), value: Optional[str] = None) -> bool:
    """Does a context value (its address metadata) apply to this entity?

    ``profile``: ``entity_profile(...)`` (a class or None means DEFAULT).
    ``own_places``: the geographic values naming the entity itself;
    ``value``: this value (needed for the place rule)."""
    dim = meta.get("dimension")
    if dim not in ALL:
        return True  # not a value of this extractor
    prof = _profile(profile)
    if dim not in prof.dimensions:
        return False
    spans = meta.get("spans") or []
    if not spans and meta.get("containers"):
        # Only the container of a named place: no mention says it, so it is
        # context of that place alone (bundler adds it there), never of the
        # entities sharing the sentence.
        return False
    own: Set[int] = set(entity_tokens or ())
    if own and spans:
        outside = [sp for sp in spans if not own.intersection(range(sp[0], sp[1]))]
        if not outside:
            return False
        spans = outside
    if dim == GEOGRAPHIC:
        if not prof.demonyms and meta.get("source") == "norp_demonym":
            return False
        if value is not None and any(place_conflict(value, p) for p in own_places or ()):
            return False
    return True


def _nested_out(vals, named_nearest):
    """Drop every candidate whose more specific version is also a candidate
    (same dimension; paths nest by prefix: "g:NA/US" < "g:NA/US/US-CA/gn..",
    "t:C19/D1880" < "t:C19/D1880/Y1885"). ``vals``: (distance, order, value).

    Except a coarser PLACE the text names much nearer the entity -- every
    more specific candidate is over twice as far (``named_nearest``: its
    nearest mention is a place name, not a demonym): that is a different
    place said of the entity ("migrated from North India" beside an earlier
    "Andhra Pradesh"), not a vaguer statement of the same one ("an American
    architect who moved to Los Angeles", "the Heart of Asia ... Afghanistan")."""
    out = []
    for x in vals:
        finer = [o for o in vals if o[2].startswith(x[2] + "/")]
        if not finer or (x[2] in named_nearest and all(o[0] > 2 * x[0] + 2 for o in finer)):
            out.append(x)
    return out


def select(profile: Union[Profile, str, None], entity_tokens: Iterable[int], ctxs: Sequence[str],
           ctx_meta: dict, own_places: Iterable[str] = ()) -> list:
    """Keep, per dimension, the profile's number of values nearest the entity,
    after dropping values a more specific candidate refines.

    Distance is token distance from the value's closest mention to the
    entity's closest token; the container of the entity's own place is at
    distance 0 (it is where the entity is). Ties keep text order. Values not
    from this extractor pass through untouched.
    """
    prof = _profile(profile)
    own = sorted(set(entity_tokens or ()))
    own_places = set(own_places or ())
    by_dim, other = {}, []
    named_nearest = set()
    for order, c in enumerate(ctxs):
        meta = ctx_meta.get(c) or {}
        dim = meta.get("dimension")
        if dim not in ALL:
            other.append(c)
            continue
        dist, named = 10 ** 6, False
        if own_places and own_places.intersection(meta.get("containers") or ()):
            dist = 0
        dem = {tuple(sp) for sp in meta.get("demonym_spans") or ()}
        for sp in meta.get("spans") or []:
            for t in own:
                d = 0 if sp[0] <= t < sp[1] else min(abs(t - sp[0]), abs(t - (sp[1] - 1)))
                if d < dist:
                    dist, named = d, tuple(sp) not in dem
        if dim == GEOGRAPHIC and named:
            named_nearest.add(c)
        by_dim.setdefault(dim, []).append((dist, order, c))
    keep = []
    for dim, vals in by_dim.items():
        vals = _nested_out(vals, named_nearest)
        # The head's max_per_dimension (stamped on each value) overrides the
        # profile's default; 0 means no cap.
        cap = prof.per_dimension
        for _, _, c in vals:
            m = (ctx_meta.get(c) or {}).get("max_per_dimension")
            if m is not None:
                cap = int(m)
                break
        keep.extend(sorted(vals) if cap <= 0 else sorted(vals)[:cap])
    return other + [c for _, _, c in sorted(keep, key=lambda x: x[1])]


def sentence_dimensions(ontology_result) -> Optional[callable]:
    """Per-sentence dimension filter from the ontology stage's output.

    The ontology stage stores each sentence's ontology in
    ``ontology_result.metadata["sentence_ontologies"]`` (sentence index ->
    class CURIE; abstaining sentences are absent) and the record's
    ontology as ``addresses[0].concept``. A sentence with no ontology of its
    own falls back to the record's. Returns None when neither exists (no
    steering: every dimension applies).
    """
    if ontology_result is None:
        return None
    meta = getattr(ontology_result, "metadata", None) or {}
    per_sent = meta.get("sentence_ontologies") or {}
    addrs = getattr(ontology_result, "addresses", None) or []
    record = getattr(addrs[0], "concept", None) if addrs else None
    if record in (None, "__null__") or class_of(record) is None:
        record = None
    if not per_sent and record is None:
        return None
    per_sent = {int(k): v for k, v in per_sent.items()}

    def dims(i: int) -> FrozenSet[str]:
        return profile_for(per_sent.get(i) or record).dimensions
    return dims
