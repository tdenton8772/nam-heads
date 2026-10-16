"""Training set for the CCO ontology codebook.

Each wiki record's subject gets a CCO class from Wikidata: its "instance of"
values are walked up "subclass of" to the nearest anchor in
``cco_anchors.ANCHORS``. The record's text is rebuilt exactly as production
sees it (the loader's title, prepended by the DCP adapter's rule), parsed with
the production NLP backend, and the sentences ABOUT the subject (the first
real sentence, and later sentences whose subject is a pronoun or the title) are
labelled with that class. Each row stores the sentence's cues (not a dense
vector), so the set stays small and vectorising happens at training time.

    python -m nam_heads.impls.ontology.codebook.training.cco_dataset \
        --corpus $NAM_DATA/wikipedia_abstracts.jsonl \
        --types $NAM_DATA/ontology/wikidata_p31.jsonl \
        --classes $NAM_DATA/ontology/wikidata_classes.jsonl \
        --output $NAM_DATA/ontology/cco_sentences.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..features import is_pointer, span_cues, title_kind
from .cco_anchors import ANCHOR_ROWS, ANCHORS, EXCLUDE

MAX_SENTENCES = 4
_SUBJ_PRON = {"he", "she", "it", "they"}


def loader_title(url: str) -> str:
    """The `__title__` the wiki loader writes (NAM: scripts/wiki_load_kv.py:_parse_record)."""
    name = url.split("/wiki/")[-1] if "/wiki/" in url else url
    name = urllib.parse.unquote(name)
    name = re.sub(r'["\'\\/:<>|*?]', '', name)
    name = name.replace("::", "_")
    return re.sub(r'[\s_]+', '_', name)


def production_text(url: str, text: str) -> str:
    """Text as the DCP adapter publishes it (adapter.py title prepend rule)."""
    title = loader_title(url)
    if title and not text.lower().startswith(title.lower()[:20]):
        return f"{title}. {text}"
    return text


class AnchorResolver:
    """Walk Wikidata P31 -> P279* to the nearest anchor class."""

    def __init__(self, classes: Dict[str, dict], max_depth: int = 8):
        self.parents = {q: r.get("p279", []) for q, r in classes.items()}
        self.max_depth = max_depth
        self._cache: Dict[str, Optional[Tuple[str, int]]] = {}

    def nearest(self, qid: str) -> Optional[Tuple[str, int]]:
        """(anchor qid, distance) by breadth-first search; ties go to ANCHORS order."""
        if qid in self._cache:
            return self._cache[qid]
        order = {a: i for i, a in enumerate(ANCHORS)}
        seen, frontier, depth, hit = {qid}, [qid], 0, None
        while frontier and depth <= self.max_depth and hit is None:
            found = [q for q in frontier if q in ANCHORS or q in EXCLUDE]
            if found:
                hit = (min(found, key=lambda q: order.get(q, -1)), depth)
                break
            nxt = []
            for q in frontier:
                for p in self.parents.get(q, []):
                    if p not in seen:
                        seen.add(p)
                        nxt.append(p)
            frontier, depth = nxt, depth + 1
        self._cache[qid] = hit
        return hit

    def label(self, p31: List[str]) -> Optional[str]:
        """CCO CURIE for a record, None if unlabelled or excluded."""
        best = None
        for q in p31:
            r = self.nearest(q)
            if r and (best is None or r[1] < best[1]):
                best = r
        if best is None or best[0] in EXCLUDE:
            return None
        return ANCHORS[best[0]]


def check_anchor_labels(classes: Dict[str, dict]) -> List[tuple]:
    """Anchors whose stated Wikidata label differs from the fetched one."""
    return [(q, want, classes[q].get("label")) for q, want, _ in ANCHOR_ROWS
            if q in classes and (classes[q].get("label") or "") != want]


def subject_sentences(doc, title: str) -> List[int]:
    """Indices of sentences about the record's subject."""
    words = {w.lower() for w in re.split(r"[_\s]+", title) if len(w) > 2}
    out = []
    for si, sent in enumerate(doc.sents):
        if title_kind(list(sent)):   # list-style title: what the record is
            out.append(si)
            continue
        if is_pointer(list(sent)):   # "Title." fragment, "See also ...", list/note leads
            continue
        toks = [t for t in sent if not getattr(t, "is_punct", False)]
        if not out:
            out.append(si)           # first real sentence: the definition
            continue
        subj = [t for t in toks if (t.dep_ or "").lower() in ("nsubj", "nsubjpass")]
        if any((t.lemma_ or "").lower() in _SUBJ_PRON or (t.text or "").lower() in words
               for t in subj):
            out.append(si)
        if len(out) >= MAX_SENTENCES:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--types", required=True)
    ap.add_argument("--classes", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)

    from nam_heads.sdk.host import parser as host_parser

    # Type evidence per title: "instance of"; for an item modelled only as a
    # kind of something, its "subclass of". Later rows for a title win.
    p31 = {}
    for line in Path(a.types).expanduser().open():
        r = json.loads(line)
        p31[r["title"]] = r.get("p31") or r.get("p279") or []
    classes = {}
    for line in Path(a.classes).expanduser().open():
        r = json.loads(line)
        classes[r["qid"]] = r
    wrong = check_anchor_labels(classes)
    if wrong:
        raise SystemExit(f"anchor labels differ from Wikidata (wrong QID?): {wrong}")
    resolver = AnchorResolver(classes)
    nlp = host_parser()

    stats = Counter()
    with Path(a.output).expanduser().open("w") as out:
        for n, line in enumerate(Path(a.corpus).expanduser().open()):
            if a.limit and n >= a.limit:
                break
            rec = json.loads(line)
            url, text = rec.get("url", ""), rec.get("text", "")
            title = urllib.parse.unquote(url.rsplit("/wiki/", 1)[-1]).replace("_", " ")
            if title not in p31:
                stats["no_types_row"] += 1
                continue
            label = resolver.label(p31[title])
            if label is None:
                stats["unlabelled"] += 1
                continue
            doc = nlp.parse(production_text(url, text))
            sents = list(doc.sents)
            for si in subject_sentences(doc, loader_title(url)):
                sp = sents[si]
                c = span_cues(list(sp), sp.ents)
                if not c:
                    continue
                out.write(json.dumps({"record": n, "title": title, "sent": si,
                                      "label": label, "cues": c}) + "\n")
                stats["sentences"] += 1
            stats["records"] += 1
            stats["label:" + label] += 1
    print(json.dumps(stats, indent=1, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
