"""Generate classes.json from the pinned BFO 2020 and CCO v2.2 releases.

    python -m nam_heads.ontology.generate [--bfo PATH] [--cco PATH]

Downloads the two pinned files (or reads local copies), verifies their sha256,
and writes every labelled class of the merged TBox, with labels, definitions,
parents and the disjointness axioms the files assert. The merged TBox is
consistent under HermiT (`certify` below).
Requires rdflib (a dev-time dependency; the runtime reads only the JSON).
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from .bindings import BFO_IRI, CCO_IRI, named_classes

SOURCES = {
    "bfo": {
        "url": "https://raw.githubusercontent.com/BFO-ontology/BFO-2020/master/21838-2/owl/bfo-core.ttl",
        "sha256": "a7f0385eb859e707fa81bad532ff252427dc3490e3896a929332016fc53d9ece",
        "version": "http://purl.obolibrary.org/obo/bfo/2020/bfo-core.ttl",
    },
    "cco": {
        "url": "https://raw.githubusercontent.com/CommonCoreOntology/CommonCoreOntologies/v2.2/src/cco-merged/CommonCoreOntologiesMerged.ttl",
        "sha256": "a9453382b25b40781c181d6ba44981f228abb76d90d548846713adba9baadb42",
        "version": "https://www.commoncoreontologies.org/2026-08-13/CommonCoreOntologiesMerged",
    },
}
OUT = Path(__file__).parent / "classes.json"
SKOS_DEFINITION = "http://www.w3.org/2004/02/skos/core#definition"


def curie(iri: str) -> str:
    if iri.startswith(BFO_IRI):
        return "BFO:" + iri[len(BFO_IRI):]
    if iri.startswith(CCO_IRI):
        return "cco:" + iri[len(CCO_IRI):]
    return ""


def _load(name: str, path: str) -> bytes:
    src = SOURCES[name]
    data = Path(path).read_bytes() if path else urllib.request.urlopen(src["url"], timeout=120).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != src["sha256"]:
        raise SystemExit(f"{name}: sha256 {digest} != pinned {src['sha256']}")
    return data


def build(bfo_bytes: bytes, cco_bytes: bytes) -> dict:
    import rdflib
    from rdflib.collection import Collection
    from rdflib.namespace import OWL, RDF, RDFS

    g = rdflib.Graph()
    g.parse(data=bfo_bytes, format="turtle")
    g.parse(data=cco_bytes, format="turtle")

    def label(node):
        for o in g.objects(node, RDFS.label):
            if getattr(o, "language", None) in (None, "en"):
                return str(o)
        return None

    def definition(node):
        for o in g.objects(node, rdflib.URIRef(SKOS_DEFINITION)):
            if getattr(o, "language", None) in (None, "en"):
                return str(o)
        return ""

    def parents(node):
        return sorted({curie(str(p)) for p in g.objects(node, RDFS.subClassOf)
                       if isinstance(p, rdflib.URIRef) and curie(str(p))})

    def node_of(c):
        prefix, local = c.split(":", 1)
        return rdflib.URIRef((BFO_IRI if prefix == "BFO" else CCO_IRI) + local)

    # Every labelled BFO/CCO class in the certified TBox (the codebook can emit
    # any of them), plus anything a binding names.
    classes = {}
    todo = sorted(named_classes() | {
        curie(str(c)) for c in g.subjects(RDF.type, OWL.Class)
        if isinstance(c, rdflib.URIRef) and curie(str(c)) and label(c) is not None})
    while todo:
        c = todo.pop()
        if c in classes:
            continue
        node = node_of(c)
        lab = label(node)
        if lab is None:
            raise SystemExit(f"{c}: not a labelled class in the pinned files")
        ps = parents(node)
        classes[c] = {"iri": str(node), "label": lab, "definition": definition(node), "parents": ps}
        todo.extend(p for p in ps if p not in classes)

    disjoint = []
    for a, b in g.subject_objects(OWL.disjointWith):
        pair = sorted({curie(str(a)), curie(str(b))})
        if len(pair) == 2 and all(pair):
            disjoint.append(pair)
    for n in g.subjects(RDF.type, OWL.AllDisjointClasses):
        for lst in g.objects(n, OWL.members):
            members = sorted(curie(str(x)) for x in Collection(g, lst))
            if all(members):
                disjoint.append(members)
    # BFO's axioms appear in both files; keep each once.
    disjoint = sorted({tuple(d) for d in disjoint})
    disjoint = [list(d) for d in disjoint]

    return {
        "sources": {k: {"url": v["url"], "sha256": v["sha256"], "version": v["version"]}
                    for k, v in SOURCES.items()},
        "classes": dict(sorted(classes.items())),
        "disjoint": disjoint,
    }


def certify(bfo_bytes: bytes, cco_bytes: bytes) -> dict:
    """Run HermiT over the merged TBox (owlready2 bundles it; needs Java)."""
    import os
    import tempfile

    import rdflib
    from owlready2 import World, sync_reasoner

    g = rdflib.Graph()
    g.parse(data=bfo_bytes, format="turtle")
    g.parse(data=cco_bytes, format="turtle")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "tbox.owl")
        g.serialize(path, format="xml")
        world = World()
        onto = world.get_ontology("file://" + path).load()
        with onto:
            sync_reasoner(world, infer_property_values=False, debug=0)
        unsat = sorted(curie(c.iri) or c.iri for c in world.inconsistent_classes())
    return {"reasoner": "HermiT (owlready2)", "consistent": True, "unsatisfiable": unsat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bfo", default="", help="local copy of bfo-core.ttl")
    ap.add_argument("--cco", default="", help="local copy of CommonCoreOntologiesMerged.ttl")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--certify", action="store_true",
                    help="run HermiT and record the result (needs owlready2 + Java)")
    a = ap.parse_args(argv)
    bfo, cco = _load("bfo", a.bfo), _load("cco", a.cco)
    doc = build(bfo, cco)
    if a.certify:
        doc["certification"] = certify(bfo, cco)
        if doc["certification"]["unsatisfiable"]:
            raise SystemExit(f"unsatisfiable classes: {doc['certification']['unsatisfiable']}")
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {a.out}: {len(doc['classes'])} classes, {len(doc['disjoint'])} disjointness axioms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
