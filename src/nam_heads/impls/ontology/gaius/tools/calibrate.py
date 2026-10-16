"""Calibrate the gaius head on labelled validation records.

    NAM_NLP_BACKEND=rule NAM_GAIUS_MODEL_DIR=docker/assets/colbert-zero-onnx \\
    python -m nam_heads.impls.ontology.gaius.tools.calibrate [--procs 5]

Writes ``artifacts/calibration.json``: the retrievable classes, the
verbalization style and its digest, the model pin (source weights + probe
scores), the softmax temperature and the acceptance threshold, with the
validation numbers they were chosen on. Nothing about the classes is learned:
the temperature (NLL on validation) and the threshold are the only fitted
numbers.

Retrievable classes: every class the active ontology class set holds
(``nam_heads/ontology/active.py``) except bfo:entity and the BFO top categories
(continuant, occurrent and the four branch classes), which a record reaches
only by back-off. Retrieving them directly let 'continuant' win 7% of
validation records (offline 2026-10-07).

Acceptance: ``--accept`` (default 0 -- commit to the top class). Every
threshold tried on validation traded exact classes for back-offs to upper BFO
categories (material entity, continuant) that do not narrow a query; the
curve is recorded in the artifact so the trade is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

STYLE = "gaius"
TOP_CATEGORIES = ("BFO:0000001", "BFO:0000002", "BFO:0000003")
PROBES = (
    ("a German-born theoretical physicist", "cco:ont00001262"),
    ("a species of moth", "cco:ont00000551"),
    ("a 2004 book", "cco:ont00002039"),
)

_W = {}


def candidate_classes():
    from nam_heads.ontology.active import emittable_classes
    from nam_heads.ontology.registry import BRANCHES

    drop = set(TOP_CATEGORIES) | set(BRANCHES)
    return [c for c in emittable_classes() if c not in drop]


def _init(model_dir, candidates):
    from nam_heads.impls.ontology.gaius.encoder import LateInteractionEncoder
    from nam_heads.impls.ontology.gaius.scoring import ClassBank
    from nam_heads.impls.ontology.gaius.verbalize import verbalizations
    from nam_heads.sdk.host import parser as host_parser

    enc = LateInteractionEncoder(model_dir, max_threads=1)
    texts = verbalizations(candidates, STYLE)
    model_id = f"{enc.meta.get('model')}:{enc.meta.get('onnx_sha256')}"
    _W.update(enc=enc, nlp=host_parser(),
              bank=ClassBank.build(enc, texts, candidates, model_id,
                                   os.environ.get("NAM_GAIUS_CACHE_DIR")))


def _score(rec):
    from nam_heads.impls.ontology.gaius.text import record_text

    q = record_text(_W["nlp"].parse(rec["text"])).query_text()
    if not q:
        return rec["record"], rec["gold"], None
    return rec["record"], rec["gold"], _W["bank"].scores(_W["enc"].encode_query(q))


def fit_temperature(S: np.ndarray, y: np.ndarray):
    from nam_heads.impls.ontology.gaius.scoring import probabilities

    best = None
    for T in np.geomspace(5e-4, 0.2, 80):
        P = np.array([probabilities(s, T) for s in S])
        nll = float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
        if best is None or nll < best[0]:
            best = (nll, float(T))
    return best


def curve(S, gold, candidates, T, thresholds):
    from nam_heads.impls.ontology.gaius.scoring import Resolver, probabilities
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    P = [probabilities(s, T) for s in S]
    out = []
    for a in thresholds:
        r = Resolver(candidates, a)
        ex = co = ab = 0
        for p, g in zip(P, gold):
            pr = r.resolve(p)
            if pr.abstained:
                ab += 1
            elif pr.cls == g:
                ex += 1
                co += 1
            elif reg.is_a(g, pr.cls):
                co += 1
        n = len(gold)
        out.append({"accept": round(float(a), 3), "exact": round(ex / n, 4),
                    "exact_or_coarse": round(co / n, 4), "abstained": round(ab / n, 4),
                    "accepted_precision": round(co / max(n - ab, 1), 4)})
    return out


def main(argv=None) -> int:
    from nam_heads.impls.ontology.gaius.encoder import LateInteractionEncoder, resolve_model_dir
    from nam_heads.impls.ontology.gaius.head import CALIBRATION_PATH, CALIBRATION_SCHEMA, class_set_id
    from nam_heads.impls.ontology.gaius.verbalize import digest, verbalizations
    from nam_heads.impls.ontology.gaius.tools.evalset import (
        DEFAULT_CORPUS, DEFAULT_DATASET, labelled_records, sample)
    from nam_heads.ontology.registry import get_registry

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="validation records (0 = all)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--accept", type=float, default=0.0)
    ap.add_argument("--out", default=str(CALIBRATION_PATH))
    a = ap.parse_args(argv)
    if os.environ.get("NAM_NLP_BACKEND") != "rule":
        raise SystemExit("set NAM_NLP_BACKEND=rule (the production parser)")
    t0 = time.time()
    reg = get_registry()
    model_dir = str(resolve_model_dir(a.model_dir))
    candidates = candidate_classes()
    texts = verbalizations(candidates, STYLE)
    recs = sample(labelled_records("val", a.dataset, a.corpus), a.limit)
    recs = [r for r in recs if r["gold"] in candidates]
    with Pool(a.procs, initializer=_init, initargs=(model_dir, candidates)) as pool:
        res = pool.map(_score, recs, chunksize=16)
    res = [r for r in res if r[2] is not None]
    S = np.array([r[2] for r in res])
    gold = [r[1] for r in res]
    y = np.array([candidates.index(g) for g in gold])
    nll, T = fit_temperature(S, y)
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cv = curve(S, gold, candidates, T, thresholds)
    chosen = curve(S, gold, candidates, T, [a.accept])[0]

    enc = LateInteractionEncoder(model_dir, max_threads=1)
    probes = []
    for text, cls in PROBES:
        q, d = enc.encode_query(text), enc.encode_document(texts[cls])
        probes.append({"text": text, "class": cls, "class_text": texts[cls],
                       "score": round(float((q @ d.T).max(axis=1).mean()), 6)})
    emittable = {"BFO:0000001"}
    for c in candidates:
        emittable |= reg.ancestors(c, inclusive=True)
    cal = {
        "schema": CALIBRATION_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tbox": reg.sources,
        "style": STYLE,
        "candidates": candidates,
        "class_set_id": class_set_id(emittable),
        "verbalizations_sha256": digest(texts),
        "model": {"name": enc.meta.get("model"), "precision": enc.meta.get("precision"),
                  "source": enc.meta.get("source"), "onnx_sha256": enc.meta.get("onnx_sha256")},
        "query": {"text": "lead defining phrase, else title + lead", "max_length": enc.query_length,
                  "aggregation": "MaxSim mean over all query tokens (gaius tbox_maxsim)"},
        "probes": probes,
        "temperature": T,
        "accept": a.accept,
        "validation": {
            "records": len(res),
            "dataset": {"path": a.dataset,
                        "sha256": hashlib.sha256(open(os.path.expanduser(a.dataset), "rb").read()).hexdigest()},
            "nll": round(nll, 4),
            "chosen": chosen,
            "curve": cv,
        },
    }
    with open(a.out, "w") as f:
        f.write(json.dumps(cal, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(res), "T": T, "nll": round(nll, 4), "chosen": chosen,
                      "seconds": round(time.time() - t0, 1)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
