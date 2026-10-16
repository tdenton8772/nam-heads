"""The labelled records both ontology heads are measured on.

The codebook head's held-out protocol (``heads/ontology/training``): each wiki
record's subject is labelled from Wikidata P31 walked to ``cco_anchors.py``,
the label is rolled up to the codebook's classes, the record goes to
train/val/test by ``split_of``, and its text is rebuilt as the DCP adapter
publishes it (``production_text``). Reusing them keeps the two heads on the
same records, labels and text. This is harness code; the gaius head itself
imports nothing from the codebook head's training.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterator, List

DATA = os.environ.get("NAM_DATA", "~/Development/nam-data")  # NAM's training corpora
DEFAULT_DATASET = os.path.join(DATA, "ontology", "cco_sentences_v3.jsonl")
DEFAULT_CORPUS = os.path.join(DATA, "wikipedia_abstracts.jsonl")


def labelled_records(split: str, dataset: str = DEFAULT_DATASET, corpus: str = DEFAULT_CORPUS,
                     codebook_meta: str = None) -> List[Dict]:
    """[{record, gold, text}] of ``split`` ("val" or "test"), in corpus order."""
    from nam_heads.impls.ontology.codebook.codebook import ARTIFACT_DIR
    from nam_heads.impls.ontology.codebook.training.cco_dataset import production_text
    from nam_heads.impls.ontology.codebook.training.cco_train import split_of

    meta = json.loads(open(codebook_meta or os.path.join(ARTIFACT_DIR, "cco_codebook.json")).read())
    rollups, classes = meta["rollups"], set(meta["classes"])
    gold: Dict[int, str] = {}
    for line in open(os.path.expanduser(dataset)):
        r = json.loads(line)
        if r["record"] in gold or split_of(r["record"]) != split:
            continue
        y = rollups.get(r["label"], r["label"])
        if y in classes:
            gold[r["record"]] = y
    out = []
    for i, line in enumerate(open(os.path.expanduser(corpus))):
        if i in gold:
            rec = json.loads(line)
            out.append({"record": i, "gold": gold[i], "text": production_text(rec["url"], rec["text"])})
    return out


def sample(records: List[Dict], limit: int, seed: int = 0) -> List[Dict]:
    if not limit or limit >= len(records):
        return records
    import numpy as np

    idx = sorted(np.random.default_rng(seed).choice(len(records), size=limit, replace=False))
    return [records[i] for i in idx]


def chunks(xs: List, n: int) -> Iterator[List]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]
