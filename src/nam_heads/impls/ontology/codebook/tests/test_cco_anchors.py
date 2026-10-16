"""Anchor table for the codebook's training labels."""

import json
import os
from pathlib import Path

import pytest

from nam_heads.impls.ontology.codebook.training.cco_anchors import ANCHOR_ROWS, ANCHORS, EXCLUDE
from nam_heads.impls.ontology.codebook.training.cco_dataset import AnchorResolver, check_anchor_labels
from nam_heads.ontology.registry import get_registry

CLASSES = Path(os.path.expanduser(os.environ.get("NAM_DATA", "~/Development/nam-data"))) / "ontology" / "wikidata_classes.jsonl"


def test_targets_are_certified_classes():
    reg = get_registry()
    assert [r for r in ANCHOR_ROWS if not reg.is_class(r[2])] == []


def test_no_duplicate_or_excluded_anchor():
    qids = [r[0] for r in ANCHOR_ROWS]
    assert len(qids) == len(set(qids))
    assert not set(qids) & EXCLUDE


@pytest.mark.skipif(not CLASSES.is_file(), reason="Wikidata class file not fetched")
def test_anchor_labels_match_wikidata():
    classes = {}
    for line in CLASSES.open():
        r = json.loads(line)
        classes[r["qid"]] = r
    assert check_anchor_labels(classes) == []


def _resolver(edges):
    return AnchorResolver({q: {"qid": q, "p279": ps} for q, ps in edges.items()})


def test_nearest_anchor_wins_over_generic():
    # Q_city_kind -> city (Q515) -> ... -> organization? nearer anchor wins
    r = _resolver({"Qx": ["Q515"], "Q515": ["Q486972"], "Q486972": ["Q43229"]})
    assert r.label(["Qx"]) == ANCHORS["Q515"]


def test_tie_goes_to_earlier_anchor():
    r = _resolver({"Qx": ["Q43229", "Q4830453"]})      # organization vs business, both depth 1
    assert r.label(["Qx"]) == ANCHORS["Q4830453"]       # business is listed first


def test_excluded_records_are_unlabelled():
    r = _resolver({})
    assert r.label(["Q4167410"]) is None
    assert r.label([]) is None
    assert r.label(["Qunknown"]) is None
