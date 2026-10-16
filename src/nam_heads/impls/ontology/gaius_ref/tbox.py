"""The class side of gaius_ref: gaius's own TBox, loaded and verbalized by gaius's own code.

gaius (github.com/zndx/gaius @ GAIUS_COMMIT) types text against
``external/sdg-corpora/ontology/sdg-ontology.owl`` (``agents/theta/tbox.py:19``),
the HermiT-certified SDG ontology, pinned by the gaius submodule at
SDG_CORPORA_COMMIT.

Loading (tbox.py:189-200) goes through two gaius functions, copied below line
for line:

- ``load_pinned_imports`` (tbox.py:128-149): the import IRI
  https://signals.zndx.org/sdg/imports/cco-module does not resolve, so the
  pinned ``imports/cco-module.ttl`` is written once as RDF/XML to
  ``ontology/.owlready/cco-module`` and that folder goes on owlready2's
  ``onto_path``;
- ``ontology_for_owlready`` (tbox.py:35-82): owlready2 refuses an IRI typed
  both class and property, so an RDF/XML copy of the OWL
  (``.owlready/sdg-ontology.nopun.owl``) drops the ``owl:Class`` typing and
  the ``rdfs:subClassOf`` of every object or datatype property. 13 IRIs lose
  their class typing this way: 918 named classes become 905.

Every named class owlready2 then enumerates (``onto.classes()``,
tbox.py:201), minus owl:Thing / owl:Nothing (tbox.py:21,165-167), becomes one
retrievable class with the text:

- label: the first ``rdfs:label`` owlready2 returns, else the IRI's local name
  (tbox.py:170-174);
- extra: every ``IAO_0000115`` value joined by a space, else the first
  non-empty ``rdfs:comment`` (tbox.py:177-184);
- text: ``f"{label}. {extra}".strip()`` when there is an extra, else the
  label (tbox.py:185). No trailing period is added, nothing is lower-cased.

``extract`` (dev time; needs owlready2 and rdflib) writes
``artifacts/tbox_classes.json``; ``load`` reads it back and refuses one whose
pins differ.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
TBOX_PATH = ARTIFACT_DIR / "tbox_classes.json"
SCHEMA = "nam-gaius-ref-tbox/2"

GAIUS_COMMIT = "06cbf95dbba33d5e865bc76b80bb6d275bc67714"
SDG_CORPORA_COMMIT = "c4138b4d020ef29b5056f6584560b133577ecbe1"
# sdg-ontology.owl and imports/cco-module.ttl are byte-identical at
# sdg-corpora c2079e67 and c4138b4d
OWL_SHA256 = "3bfc4fca671511808d64d684af10ff7dc234fa517561d02c27c0f78b29f8327d"
IMPORT_SHA256 = "9209fcb2c8fbc228598aac8f00e87c83929f4989324161c14c4eab22e0bf5d73"
IMPORT_IRI = "https://signals.zndx.org/sdg/imports/cco-module"

# CURIE prefixes: the registry's own CURIE for any IRI it knows (identity of
# IRI, not a mapping); these prefixes for the rest.
PREFIXES = (
    ("sdg", "https://signals.zndx.org/sdg#"),
    ("xsd", "http://www.w3.org/2001/XMLSchema#"),
    ("cco", "https://www.commoncoreontologies.org/"),
    ("BFO", "http://purl.obolibrary.org/obo/BFO_"),
)

# ---- gaius agents/theta/tbox.py @ GAIUS_COMMIT ------------------------------
_SKIP_IRI_FRAGMENTS = ("Nothing", "Thing")                            # tbox.py:21


def ontology_for_owlready(path: Path) -> Path:                        # tbox.py:35-82
    """RDF/XML owlready2 can load: CCO is local, property/class puns dropped."""
    from rdflib import Graph, Namespace

    src = Path(path)
    cache = src.parent / ".owlready" / f"{src.stem}.nopun.owl"
    cco_xml = src.parent / ".owlready" / "cco-module"
    ttl = src.parent / "imports" / "cco-module.ttl"
    if (
        cache.is_file()
        and cco_xml.is_file()
        and cache.stat().st_mtime >= src.stat().st_mtime
        and (not ttl.is_file() or cco_xml.stat().st_mtime >= ttl.stat().st_mtime)
    ):
        from owlready2 import onto_path

        folder = str(cco_xml.parent)
        if folder not in onto_path:
            onto_path.insert(0, folder)
        return cache
    cache.parent.mkdir(exist_ok=True)
    if ttl.is_file():
        cco = Graph()
        cco.parse(ttl, format="turtle")
        cco.serialize(destination=cco_xml, format="xml")
    graph = Graph()
    graph.parse(src, format="xml")
    OWL = Namespace("http://www.w3.org/2002/07/owl#")
    RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    props = set(graph.subjects(RDF.type, OWL.ObjectProperty)) | set(
        graph.subjects(RDF.type, OWL.DatatypeProperty)
    )
    for prop in props:
        graph.remove((prop, RDF.type, OWL.Class))
        graph.remove((prop, RDFS.subClassOf, None))
    graph.serialize(destination=cache, format="xml")
    from owlready2 import onto_path

    folder = str(cache.parent)
    if folder not in onto_path:
        onto_path.insert(0, folder)
    return cache


def load_pinned_imports(certified_owl: Path) -> None:                 # tbox.py:128-149
    """Put CCO on owlready2's onto_path so the import IRI is not fetched.
    (gaius reads its module constant CERTIFIED_OWL; here it is a parameter.)"""
    from owlready2 import onto_path
    from rdflib import Graph

    ttl = certified_owl.parent / "imports" / "cco-module.ttl"
    if not ttl.is_file():
        return
    cache = ttl.parent / ".owlready"
    cache.mkdir(exist_ok=True)
    xml = cache / "cco-module"
    if not xml.is_file() or xml.stat().st_mtime < ttl.stat().st_mtime:
        graph = Graph()
        graph.parse(ttl, format="turtle")
        graph.serialize(destination=xml, format="xml")
    folder = str(cache)
    if folder not in onto_path:
        onto_path.insert(0, folder)


def _local(iri: str) -> str:                                          # tbox.py:159-162
    if "#" in iri:
        return iri.rsplit("#", 1)[1]
    return iri.rstrip("/").rsplit("/", 1)[-1]


def _skip(iri: str) -> bool:                                          # tbox.py:165-167
    loc = _local(iri)
    return loc in _SKIP_IRI_FRAGMENTS or iri.endswith("#Thing") or iri.endswith("#Nothing")


def _label_of(cls: object) -> str:                                    # tbox.py:170-174
    labs = list(getattr(cls, "label", []) or [])
    if labs:
        return str(labs[0])
    return _local(str(getattr(cls, "iri", "")))


def _retrieval_text(cls: object, label: str) -> str:                  # tbox.py:177-185
    comments = [str(c) for c in (getattr(cls, "comment", []) or []) if str(c).strip()]
    defs = []
    for name in ("IAO_0000115",):
        vals = getattr(cls, name, None)
        if vals:
            defs.extend(str(v) for v in vals if str(v).strip())
    extra = " ".join(defs or comments[:1])
    return f"{label}. {extra}".strip() if extra else label
# -----------------------------------------------------------------------------


class TBoxError(RuntimeError):
    """The TBox artifact is missing or not the pinned one."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def punned_iris(owl_path: Path) -> List[str]:
    """IRIs the OWL types both owl:Class and an object or datatype property:
    the ones ``ontology_for_owlready`` stops being classes."""
    from rdflib import Graph
    from rdflib.namespace import OWL, RDF

    g = Graph().parse(owl_path, format="xml")
    props = set(g.subjects(RDF.type, OWL.ObjectProperty)) | set(g.subjects(RDF.type, OWL.DatatypeProperty))
    return sorted(str(s) for s in props if (s, RDF.type, OWL.Class) in g)


def extract(owl_path: Path) -> dict:
    """gaius's ``tbox_relations`` class records for the pinned SDG OWL, loaded
    by gaius's loaders (tbox.py:198-201). Writes gaius's ``.owlready`` caches
    beside the OWL, as gaius does."""
    import owlready2

    owl_path = Path(owl_path)
    load_pinned_imports(owl_path)
    ready = ontology_for_owlready(owl_path)
    onto = owlready2.World().get_ontology(ready.resolve().as_uri()).load()
    imported = [o.base_iri for o in onto.imported_ontologies]
    if not any(b.rstrip("#/") == IMPORT_IRI for b in imported):
        raise SystemExit(f"the CCO import did not load: {imported}")
    classes = [c for c in onto.classes() if getattr(c, "iri", None) and not _skip(c.iri)]  # tbox.py:201
    rows = []
    for cls in classes:
        label = _label_of(cls)
        rows.append({"iri": cls.iri, "label": label, "text": _retrieval_text(cls, label),
                     "definition": _definition(cls),
                     "parents": _named_parents(cls),
                     "told_parents": _told_subsumers(cls),
                     "ancestors": sorted(_named_ancestors(cls, owlready2.Thing))})
    # gaius's own enumeration order is not stable: rdflib writes the
    # .nopun.owl cache in hash-seed order, and 6 fresh builds gave 6 orders
    # of the same 905 (iri, label, text) rows (2026-10-07). Order only breaks
    # exact score ties; the artifact keeps one fixed order, by IRI.
    rows.sort(key=lambda r: r["iri"])
    # the told parents of every other named class on those classes' parent
    # chains (the CCO module's), so a parent outside the TBox can be climbed
    known = {r["iri"] for r in rows}
    world_parents: Dict[str, List[str]] = {}
    todo = [p for r in rows for p in r["told_parents"] if p not in known]
    while todo:
        iri = todo.pop()
        if iri in world_parents or iri in known:
            continue
        ent = onto.world[iri]
        world_parents[iri] = _told_subsumers(ent) if ent is not None else []
        todo.extend(world_parents[iri])
    return {
        "schema": SCHEMA,
        "source": {
            "gaius_commit": GAIUS_COMMIT,
            "sdg_corpora_commit": SDG_CORPORA_COMMIT,
            "owl": "external/sdg-corpora/ontology/sdg-ontology.owl",
            "owl_sha256": file_sha256(owl_path),
            "import": IMPORT_IRI,
            "import_file": "ontology/imports/cco-module.ttl",
            "import_sha256": file_sha256(owl_path.parent / "imports" / "cco-module.ttl"),
            "loader": "gaius tbox.py load_pinned_imports + ontology_for_owlready",
            "owlready2": getattr(owlready2, "VERSION", ""),
            "punned_class_typing_dropped": punned_iris(owl_path),
        },
        "classes": rows,
        "world_parents": dict(sorted(world_parents.items())),
    }


def _definition(cls) -> str:
    """The ``extra`` of tbox.py:177-184 alone (the registry's definition field)."""
    label = _label_of(cls)
    text = _retrieval_text(cls, label)
    return text[len(label) + 2:] if text != label else ""


def _named_parents(cls) -> List[str]:
    """Named superclass IRIs (``cls.is_a``), as tbox.py:213-215 reads them."""
    out = []
    for parent in getattr(cls, "is_a", []) or []:
        iri = getattr(parent, "iri", None)
        if iri and iri != cls.iri and not _skip(iri) and iri not in out:
            out.append(iri)
    return out


_OWL_NS = "http://www.w3.org/2002/07/owl#"


def _told_subsumers(cls) -> List[str]:
    """Named classes the TBox states ``cls`` is subsumed by, without a
    reasoner: named ``rdfs:subClassOf`` superclasses, named
    ``owl:equivalentClass`` classes, and the named conjuncts of an
    intersection in either (A = B and ... entails A subClassOf B).
    Restrictions are not subsumers. The registry's hierarchy, not gaius's
    (tbox.py reads ``is_a`` only)."""
    out: List[str] = []

    def add(c):
        iri = getattr(c, "iri", None)
        if iri and iri != cls.iri and not _skip(iri) and not iri.startswith(_OWL_NS) and iri not in out:
            out.append(iri)

    for expr in list(getattr(cls, "is_a", []) or []) + list(getattr(cls, "equivalent_to", []) or []):
        if getattr(expr, "iri", None):
            add(expr)
        for part in getattr(expr, "Classes", None) or []:   # owlready2 And(...)
            if getattr(part, "iri", None) and type(expr).__name__ == "And":
                add(part)
    return out


def _named_ancestors(cls, thing) -> set:
    """``cls.ancestors()`` minus itself, owl:Thing and skipped IRIs (tbox.py:205-209)."""
    return {a.iri for a in cls.ancestors()
            if a is not cls and a is not thing and getattr(a, "iri", None) and not _skip(a.iri)}


@lru_cache(maxsize=1)
def load(path: str = str(TBOX_PATH)) -> Dict:
    p = Path(path)
    if not p.is_file():
        raise TBoxError(f"gaius TBox artifact missing: {p} (python -m nam_heads.impls.ontology."
                        "gaius_ref.tbox --owl ...)")
    doc = json.loads(p.read_text())
    if doc.get("schema") != SCHEMA:
        raise TBoxError(f"TBox artifact schema {doc.get('schema')!r} != {SCHEMA!r}")
    src = doc.get("source") or {}
    if (src.get("gaius_commit"), src.get("sdg_corpora_commit"), src.get("owl_sha256"),
            src.get("import_sha256")) != (GAIUS_COMMIT, SDG_CORPORA_COMMIT, OWL_SHA256, IMPORT_SHA256):
        raise TBoxError("TBox artifact was extracted from another gaius / sdg-corpora pin")
    rows = doc.get("classes") or []
    iris = [r["iri"] for r in rows]
    if not rows or len(set(iris)) != len(iris) or any(not (r.get("text") or "").strip() for r in rows):
        raise TBoxError("TBox artifact has no classes, a duplicate IRI or an empty class text")
    return doc


EXTENSION_PATH = ARTIFACT_DIR / "sdg_ontology.json"
EXTENSION_NAME = "sdg"


def build_extension(doc: Dict, base) -> Dict:
    """The registry extension (``nam_heads/ontology/registry.py``, schema
    ``nam-ontology-extension/1``) for gaius's TBox, from the extracted rows.

    - classes: the TBox classes the base registry does not have (by IRI),
      keyed by CURIE (PREFIXES), with gaius's label, the definition part of
      gaius's text, and parents: each row's told subsumers mapped to CURIEs.
      A told subsumer outside both the base and the TBox is climbed through
      ``world_parents`` to the nearest known classes.
    - emittable: every class gaius can return (all 905), as CURIEs. A TBox
      class the base registry has keeps the base's CURIE, label and parents.
    """
    base_iris = {base.iri(c): c for c in base.classes()}
    rows = doc["classes"]
    cur = {r["iri"]: curie(r["iri"], base_iris) for r in rows}
    if len(set(cur.values())) != len(cur):
        raise SystemExit("two TBox IRIs share a CURIE")
    world = doc.get("world_parents") or {}

    def known(iri, seen=()):
        if iri in cur:
            return [cur[iri]]
        if iri in base_iris:
            return [base_iris[iri]]
        out = []
        for p in world.get(iri, []):
            if p not in seen:
                out += [k for k in known(p, seen + (iri,)) if k not in out]
        return out

    ext: Dict[str, Dict] = {}
    climbed = 0
    for r in rows:
        if r["iri"] in base_iris:
            continue
        parents: List[str] = []
        for p in r["told_parents"]:
            ks = known(p)
            climbed += p not in cur and p not in base_iris
            parents += [k for k in ks if k not in parents and k != cur[r["iri"]]]
        ext[cur[r["iri"]]] = {"iri": r["iri"], "label": r["label"],
                              "definition": r.get("definition", ""), "parents": parents}
    # break cycles (named equivalences) deterministically: drop the edge
    # that closes a cycle in a depth-first walk in CURIE order
    dropped: List[List[str]] = []
    state: Dict[str, int] = {}

    def visit(c):
        state[c] = 1
        for p in list(ext[c]["parents"]):
            if p not in ext:
                continue
            if state.get(p) == 1:
                ext[c]["parents"].remove(p)
                dropped.append([c, p])
            elif p not in state:
                visit(p)
        state[c] = 2

    for c in sorted(ext):
        if c not in state:
            visit(c)
    base_edges = sum(1 for r in rows if r["iri"] in base_iris
                     for p in r["told_parents"] if p in base_iris
                     and base_iris[p] not in base.parents(base_iris[r["iri"]]))
    return {
        "schema": "nam-ontology-extension/1",
        "name": EXTENSION_NAME,
        "description": ("gaius's certified TBox, the SDG ontology (sdg-corpora "
                        f"{SDG_CORPORA_COMMIT[:8]}), as gaius's tbox.py reads it"),
        "base_sources": base.sources,
        "source": doc["source"],
        "prefixes": {p: ns for p, ns in PREFIXES},
        "classes": dict(sorted(ext.items())),
        "emittable": sorted(cur.values()),
        "notes": {
            "parents": "told subsumers: named rdfs:subClassOf and owl:equivalentClass classes and the "
                       "named conjuncts of an intersection in either",
            "parents_climbed_through_the_import": climbed,
            "cycle_edges_dropped": dropped,
            "sdg_edges_on_base_classes_not_applied": base_edges,
        },
    }


def curie(iri: str, registry_iris: Dict[str, str]) -> str:
    """The CURIE for ``iri``: the registry's own when it knows the IRI,
    else a prefix from PREFIXES; the IRI itself when no prefix fits."""
    if iri in registry_iris:
        return registry_iris[iri]
    for prefix, ns in PREFIXES:
        if iri.startswith(ns) and len(iri) > len(ns):
            return f"{prefix}:{iri[len(ns):]}"
    return iri


def registry_iris() -> Dict[str, str]:
    """IRI -> CURIE for every class of nam's built-in (BFO/CCO) registry."""
    from nam_heads.ontology.registry import base_registry

    reg = base_registry()
    return {reg.iri(c): c for c in reg.classes()}


@lru_cache(maxsize=1)
def load_extension(path: str = str(EXTENSION_PATH)) -> Dict:
    """The registry extension artifact, checked against this TBox pin."""
    p = Path(path)
    if not p.is_file():
        raise TBoxError(f"ontology artifact missing: {p}")
    doc = json.loads(p.read_text())
    src = doc.get("source") or {}
    if (doc.get("schema") != "nam-ontology-extension/1" or doc.get("name") != EXTENSION_NAME
            or src.get("gaius_commit") != GAIUS_COMMIT or src.get("owl_sha256") != OWL_SHA256):
        raise TBoxError(f"{p}: not the {EXTENSION_NAME} extension of this TBox pin")
    return doc


def iri_to_curie() -> Dict[str, str]:
    """Every TBox IRI -> the CURIE the extension (and so the registry) uses."""
    ext = load_extension()
    base_iri_of = {c: i for i, c in registry_iris().items()}
    out = {v["iri"]: c for c, v in ext["classes"].items()}
    for c in ext["emittable"]:
        if c not in ext["classes"]:
            if c not in base_iri_of:
                raise TBoxError(f"{c}: emittable but neither BFO/CCO nor in the extension")
            out[base_iri_of[c]] = c
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract gaius's TBox class texts and ontology (dev time).")
    ap.add_argument("--owl", required=True,
                    help="sdg-corpora@%s ontology/sdg-ontology.owl (its imports/ beside it)" % SDG_CORPORA_COMMIT[:8])
    ap.add_argument("--out", default=str(TBOX_PATH))
    a = ap.parse_args(argv)
    owl = Path(a.owl)
    for path, want in ((owl, OWL_SHA256), (owl.parent / "imports" / "cco-module.ttl", IMPORT_SHA256)):
        if file_sha256(path) != want:
            raise SystemExit(f"{path}: not the pinned file (sha256 {want})")
    doc = extract(owl)
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    rows = doc["classes"]
    print(f"wrote {a.out}: {len(rows)} classes, "
          f"{sum(1 for r in rows if r['text'] == r['label'])} label-only, "
          f"{sum(1 for r in rows if r['label'] == _local(r['iri']))} labelled by IRI local name, "
          f"{len(doc['source']['punned_class_typing_dropped'])} punned IRIs not classes")
    from nam_heads.ontology.registry import base_registry

    ext = build_extension(doc, base_registry())
    EXTENSION_PATH.write_text(json.dumps(ext, indent=1, ensure_ascii=False) + "\n")
    base_registry().extended(ext)          # refuses an extension the registry cannot load
    print(f"wrote {EXTENSION_PATH}: {len(ext['classes'])} classes beyond BFO/CCO, "
          f"{len(ext['emittable'])} emittable, notes {json.dumps(ext['notes'])[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
