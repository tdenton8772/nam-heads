"""TBox verbalization: each BFO/CCO class as text.

The gaius method (github.com/zndx/gaius, ``agents/theta/tbox.py``) types text
by retrieval against the certified TBox: every named class becomes a short
text (its label and definition) and is embedded once. This module produces
those texts from the same pinned BFO 2020 + CCO v2.2 releases that
``nam_heads/ontology/classes.json`` is generated from.

- ``extract`` (dev time, needs rdflib) reads the pinned Turtle files and
  writes ``artifacts/tbox_annotations.json``: for every class of the registry,
  the annotations a verbalization can use (label, alternative labels and
  acronyms, definitions, examples, scope notes). It records the TBox pins, and
  ``load_annotations`` refuses a file whose pins differ from the registry's.
- ``verbalize`` turns one class into its text, in a named STYLE. The style is
  pinned in the calibration artifact: a different style is a different head.

Nothing here is learned from data; the texts come from the TBox alone.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List

from nam_heads.ontology.registry import get_registry

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ANNOTATIONS_PATH = ARTIFACT_DIR / "tbox_annotations.json"
SCHEMA = "nam-gaius-tbox-annotations/1"

SKOS = "http://www.w3.org/2004/02/skos/core#"
CCO_ACRONYM = "https://www.commoncoreontologies.org/ont00001753"

# gaius: "label. definition" (tbox.py _retrieval_text).
# tbox: label and alternative labels, definition, the parents it specialises
# and the TBox's own examples -- everything the TBox says about the class in
# words, nothing else.
STYLES = ("gaius", "tbox")

_ELUCIDATION = re.compile(r"^\s*\(elucidation\)\s*", re.I)


class AnnotationsError(RuntimeError):
    """The annotations artifact is missing or does not match the registry's TBox."""


def _clean(s: str) -> str:
    s = _ELUCIDATION.sub("", s or "")
    return re.sub(r"\s+", " ", s).strip()


def _sentence(s: str) -> str:
    s = _clean(s)
    return s if not s or s[-1] in ".!?" else s + "."


def extract(bfo_bytes: bytes, cco_bytes: bytes) -> dict:
    """Annotations of every registry class from the pinned Turtle files."""
    import rdflib
    from rdflib.namespace import RDFS

    reg = get_registry()
    g = rdflib.Graph()
    g.parse(data=bfo_bytes, format="turtle")
    g.parse(data=cco_bytes, format="turtle")

    def literals(node, pred) -> List[str]:
        out = []
        for o in g.objects(node, rdflib.URIRef(pred)):
            if isinstance(o, rdflib.Literal) and getattr(o, "language", None) in (None, "en"):
                v = _clean(str(o))
                if v and v not in out:
                    out.append(v)
        return sorted(out)

    classes: Dict[str, dict] = {}
    for c in sorted(reg.classes()):
        node = rdflib.URIRef(reg.iri(c))
        label = reg.label(c)
        alts = [a for a in literals(node, SKOS + "altLabel") + literals(node, SKOS + "prefLabel")
                + literals(node, CCO_ACRONYM) if a.lower() != label.lower()]
        defs = literals(node, SKOS + "definition")
        primary = _clean(reg.definition(c))
        if primary in defs:   # the registry's definition first, as the registry reads it
            defs.remove(primary)
        classes[c] = {
            "label": label,
            "alt_labels": sorted(set(alts)),
            "definitions": ([primary] if primary else []) + defs,
            "examples": literals(node, SKOS + "example"),
            "scope_notes": literals(node, SKOS + "scopeNote"),
        }
        if not g.value(node, RDFS.label):
            raise SystemExit(f"{c}: not labelled in the pinned files")
    return {"schema": SCHEMA, "tbox": reg.sources, "classes": classes}


@lru_cache(maxsize=1)
def load_annotations(path: str = str(ANNOTATIONS_PATH)) -> Dict[str, dict]:
    p = Path(path)
    if not p.is_file():
        raise AnnotationsError(f"TBox annotations missing: {p} (run tools/verbalize_tbox.py)")
    doc = json.loads(p.read_text())
    if doc.get("schema") != SCHEMA:
        raise AnnotationsError(f"annotations schema {doc.get('schema')!r} != {SCHEMA!r}")
    if doc.get("tbox") != get_registry().sources:
        raise AnnotationsError("annotations were extracted from a different TBox than the registry's")
    return doc["classes"]


def verbalize(c: str, style: str = "tbox", annotations: Dict[str, dict] = None) -> str:
    """The text that stands for class ``c``."""
    if style not in STYLES:
        raise ValueError(f"unknown verbalization style {style!r}; known: {STYLES}")
    reg = get_registry()
    ann = (annotations or load_annotations()).get(c)
    if ann is None:
        raise KeyError(f"{c}: no annotations")
    label = ann["label"]
    definition = ann["definitions"][0] if ann["definitions"] else ""
    if style == "gaius":
        return f"{label}. {_sentence(definition)}".strip() if definition else label
    head = label
    if ann["alt_labels"]:
        head += " (" + ", ".join(ann["alt_labels"]) + ")"
    parts = [head + "."]
    if definition:
        parts.append(_sentence(definition))
    parents = [reg.label(p) for p in reg.parents(c)]
    if parents:
        parts.append(f"A kind of {', '.join(parents)}.")
    if ann["examples"]:
        parts.append("Examples: " + _sentence("; ".join(ann["examples"])))
    return " ".join(parts)


def verbalizations(classes: Iterable[str], style: str) -> Dict[str, str]:
    ann = load_annotations()
    return {c: verbalize(c, style, ann) for c in classes}


def digest(texts: Dict[str, str]) -> str:
    """sha256 of a class -> text map (the class side's identity)."""
    return hashlib.sha256(json.dumps(texts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(argv=None) -> int:
    import argparse

    from nam_heads.ontology.generate import _load

    ap = argparse.ArgumentParser(description="Extract TBox annotations for the gaius head.")
    ap.add_argument("--bfo", default="", help="local copy of the pinned bfo-core.ttl")
    ap.add_argument("--cco", default="", help="local copy of the pinned CommonCoreOntologiesMerged.ttl")
    ap.add_argument("--out", default=str(ANNOTATIONS_PATH))
    a = ap.parse_args(argv)
    doc = extract(_load("bfo", a.bfo), _load("cco", a.cco))
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    n = doc["classes"]
    print(f"wrote {a.out}: {len(n)} classes, "
          f"{sum(1 for v in n.values() if v['examples'])} with examples, "
          f"{sum(1 for v in n.values() if v['alt_labels'])} with alternative labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
