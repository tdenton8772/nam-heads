"""Measure the gaius head against the codebook head on the same held-out records.

    NAM_NLP_BACKEND=rule NAM_GAIUS_MODEL_DIR=docker/assets/colbert-zero-onnx \\
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    python -m nam_heads.impls.ontology.gaius.tools.evaluate \\
        [--limit 0] [--procs 5] [--latency 300] [--out report.json]

Both heads see the same records (the codebook's test split, ``evalset``), the
same production text and the same rule-backend parse, and are scored the same
way: exact class; exact or correct-but-coarse (the prediction subsumes the
gold class); abstained (bfo:entity); precision when committed. Also: the
share typed Person, the predicted-class spread, the top confusions, and how
often the two heads agree.

Latency is measured in this process, one thread (ORT intra-op 1, BLAS 1),
on ``--latency`` records: parse once, then each head's full ``encode``.
Memory is the peak RSS of a fresh process that builds one head and encodes 50
records.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context

import numpy as np

_W = {}


def _init(model_dir):
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    from nam_heads.impls.ontology.gaius.head import GaiusOntologyEncoder
    from nam_heads.sdk.host import parser as host_parser

    _W["nlp"] = host_parser()
    _W["codebook"] = OntologyEncoder(head_type="ontology")
    _W["gaius"] = GaiusOntologyEncoder(head_type="ontology",
                                       config={"model_dir": model_dir, "max_threads": 1})


def _both(rec):
    doc = _W["nlp"].parse(rec["text"])
    a = _W["codebook"].encode(rec["text"], {"spacy_doc": doc})
    b = _W["gaius"].encode(rec["text"], {"spacy_doc": doc})
    return {"record": rec["record"], "gold": rec["gold"],
            "codebook": a.addresses[0].concept, "gaius": b.addresses[0].concept,
            "gaius_conf": b.addresses[0].confidence,
            "gaius_sentences": len(b.metadata.get("sentence_ontologies", {})),
            "codebook_sentences": len(a.metadata.get("sentence_ontologies", {}))}


def score(preds, gold):
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    n = len(gold)
    ex = co = ab = 0
    conf = Counter()
    for p, g in zip(preds, gold):
        if p == "BFO:0000001":
            ab += 1
        elif p == g:
            ex += 1
            co += 1
        elif reg.is_a(g, p):
            co += 1
        else:
            conf[(reg.label(g), reg.label(p))] += 1
    dist = Counter(preds)
    return {
        "n": n,
        "exact": round(ex / n, 4),
        "exact_or_coarse": round(co / n, 4),
        "abstained": round(ab / n, 4),
        "accepted_precision": round(co / max(n - ab, 1), 4),
        "person_share": round(dist.get("cco:ont00001262", 0) / n, 4),
        "distinct_classes": len(dist),
        "predicted_top": [[reg.label(c), round(k / n, 4)] for c, k in dist.most_common(10)],
        "top_confusions": [[g, p, k] for (g, p), k in conf.most_common(12)],
    }


def agreement(a, b):
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    n = len(a)
    same = sum(x == y for x, y in zip(a, b))
    compat = sum(x == y or reg.is_a(x, y) or reg.is_a(y, x) for x, y in zip(a, b))
    return {"same_class": round(same / n, 4), "same_or_subsumes": round(compat / n, 4)}


def _peak_rss(which, model_dir, texts):
    """Peak RSS (MB) of a process holding one head after 50 encodes."""
    import resource

    from nam_heads.sdk.host import parser as host_parser

    nlp = host_parser()
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if which == "codebook":
        from nam_heads.impls.ontology.codebook.head import OntologyEncoder
        head = OntologyEncoder(head_type="ontology")
    else:
        from nam_heads.impls.ontology.gaius.head import GaiusOntologyEncoder
        head = GaiusOntologyEncoder(head_type="ontology", config={"model_dir": model_dir, "max_threads": 1})
    for t in texts:
        head.encode(t, {"spacy_doc": nlp.parse(t)})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1 << 20 if sys.platform == "darwin" else 1 << 10   # bytes on macOS, KB on Linux
    return {"peak_rss_mb": round(peak / div, 1), "head_mb": round((peak - base) / div, 1)}


def latency(records, model_dir):
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    from nam_heads.impls.ontology.gaius.head import GaiusOntologyEncoder
    from nam_heads.sdk.host import parser as host_parser

    nlp = host_parser()
    heads = {
        "codebook": OntologyEncoder(head_type="ontology"),
        "gaius": GaiusOntologyEncoder(head_type="ontology", config={"model_dir": model_dir, "max_threads": 1}),
        "gaius_record_only": GaiusOntologyEncoder(
            head_type="ontology", config={"model_dir": model_dir, "max_threads": 1, "max_sentences": 1,
                                          "text_cache_size": 0}),
    }
    heads["gaius"]._cache.size = 0          # time the model, not the cache
    docs = [(r["text"], nlp.parse(r["text"])) for r in records]
    for h in heads.values():
        for text, doc in docs[:5]:          # warm-up
            h.encode(text, {"spacy_doc": doc})
    # interleaved per record, so machine load hits every head alike
    ts = {name: [] for name in heads}
    for text, doc in docs:
        for name, h in heads.items():
            t0 = time.perf_counter()
            h.encode(text, {"spacy_doc": doc})
            ts[name].append((time.perf_counter() - t0) * 1000)
    out = {}
    for name, v in ts.items():
        v = np.array(v)
        out[name] = {"mean_ms": round(float(v.mean()), 2), "p50_ms": round(float(np.percentile(v, 50)), 2),
                     "p95_ms": round(float(np.percentile(v, 95)), 2), "n": len(v)}
    out["load_avg_1m"] = round(os.getloadavg()[0], 1)
    return out


def main(argv=None) -> int:
    from nam_heads.impls.ontology.gaius.encoder import resolve_model_dir
    from nam_heads.impls.ontology.gaius.tools.evalset import (
        DEFAULT_CORPUS, DEFAULT_DATASET, labelled_records, sample)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--limit", type=int, default=0, help="test records (0 = all)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--latency", type=int, default=300, help="records timed single-threaded")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if os.environ.get("NAM_NLP_BACKEND") != "rule":
        raise SystemExit("set NAM_NLP_BACKEND=rule (the production parser)")
    model_dir = str(resolve_model_dir(a.model_dir))
    t0 = time.time()
    recs = sample(labelled_records("test", a.dataset, a.corpus), a.limit)
    ctx = get_context("spawn")
    with ctx.Pool(a.procs, initializer=_init, initargs=(model_dir,)) as pool:
        rows = pool.map(_both, recs, chunksize=16)
    gold = [r["gold"] for r in rows]
    report = {
        "records": len(rows),
        "codebook": score([r["codebook"] for r in rows], gold),
        "gaius": score([r["gaius"] for r in rows], gold),
        "agreement": agreement([r["codebook"] for r in rows], [r["gaius"] for r in rows]),
        "sentences_classified_mean": {
            "codebook": round(float(np.mean([r["codebook_sentences"] for r in rows])), 2),
            "gaius": round(float(np.mean([r["gaius_sentences"] for r in rows])), 2)},
        "gaius_confidence_mean": round(float(np.mean([r["gaius_conf"] for r in rows])), 4),
    }
    lat = sample(recs, a.latency, seed=1)
    report["latency_single_thread"] = latency(lat, model_dir)
    texts = [r["text"] for r in lat[:50]]
    for which in ("codebook", "gaius"):
        with ctx.Pool(1) as p:
            report.setdefault("memory", {})[which] = p.apply(_peak_rss, (which, model_dir, texts))
    report["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(report, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
