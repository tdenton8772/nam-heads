"""Coverage of gaius_ref on the held-out records the other ontology heads are measured on.

    NAM_NLP_BACKEND=rule NAM_GAIUS_REF_MODEL_DIR=<export dir> OMP_NUM_THREADS=1 \\
    python -m nam_heads.impls.ontology.gaius_ref.tools.evaluate \\
        [--limit 0] [--procs 4] [--latency 300] [--allow-substitute] [--out report.json] \\
        [--predictions preds.jsonl]

The records are the gaius head's harness (``impls/ontology/gaius/tools``):
the codebook's test split (``evalset.labelled_records``), with the same
production text. gaius_ref reads the record text itself and needs no parse.

gaius_ref's classes are the SDG ontology's, so accuracy against the BFO/CCO
gold labels does not measure it. The report is coverage:

- the share of records with a class (not the empty-text abstain);
- the share the bundler keys by that class, with the SDG ontology active in
  the registry (as the manifest makes it when gaius_ref is selected);
- the class distribution: distinct classes, the top 20 with counts, the
  share of the top 10, SDG against BFO/CCO classes;
- single-thread ms/record and the RSS of a process holding the head.

The BFO/CCO ``score`` is included for reference only. ``--predictions``
writes one line per record (record, title, class, label, score) for spot
checks. ``--allow-substitute`` runs an export that is not
lightonai/ColBERT-Zero; the report names the weights it ran.
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


def _config(model_dir, allow_substitute, cache_dir):
    return {"model_dir": model_dir, "max_threads": 1, "require_colbert_zero": not allow_substitute,
            "cache_dir": cache_dir}


def _activate():
    """The registry as a manifest selecting gaius_ref leaves it."""
    from nam_heads.impls.ontology.gaius_ref import tbox as T
    from nam_heads.ontology.registry import use_extension

    use_extension(str(T.EXTENSION_PATH))


def _init(model_dir, allow_substitute, cache_dir):
    from nam_heads.impls.ontology.gaius_ref.head import GaiusRefOntologyEncoder

    _activate()
    _W["head"] = GaiusRefOntologyEncoder(head_type="ontology",
                                         config=_config(model_dir, allow_substitute, cache_dir))


def _one(rec):
    from nam_heads.ontology.registry import get_registry

    r = _W["head"].encode(rec["text"])
    top = r.addresses[0]
    return {"record": rec["record"], "gold": rec["gold"], "pred": top.concept,
            "top": [c for c, _iri, _s in r.metadata.get("top_k") or []], "score": top.confidence,
            "abstained": bool(r.metadata.get("abstained")),
            "sdg": bool(r.metadata.get("outside_bfo_cco")),
            # NAM's bundler keys a record by its class only when the registry
            # knows it (addressing/bundler.py _record_class), else bfo:entity
            "keyed": get_registry().is_class(top.concept),
            "text": rec["text"][:300]}


def _peak_rss(model_dir, allow_substitute, cache_dir, texts):
    import resource

    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    from nam_heads.impls.ontology.gaius_ref.head import GaiusRefOntologyEncoder

    head = GaiusRefOntologyEncoder(head_type="ontology", config=_config(model_dir, allow_substitute, cache_dir))
    for t in texts:
        head.encode(t)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    div = 1 << 20 if sys.platform == "darwin" else 1 << 10
    return {"peak_rss_mb": round(peak / div, 1), "head_mb": round((peak - base) / div, 1)}


def _latency(records, model_dir, allow_substitute, cache_dir):
    from nam_heads.impls.ontology.gaius_ref.head import GaiusRefOntologyEncoder

    cfg = {**_config(model_dir, allow_substitute, cache_dir), "text_cache_size": 0}
    head = GaiusRefOntologyEncoder(head_type="ontology", config=cfg)
    for r in records[:5]:
        head.encode(r["text"])
    ts = []
    for r in records:
        t0 = time.perf_counter()
        head.encode(r["text"])
        ts.append((time.perf_counter() - t0) * 1000)
    v = np.array(ts)
    return {"mean_ms": round(float(v.mean()), 2), "p50_ms": round(float(np.percentile(v, 50)), 2),
            "p95_ms": round(float(np.percentile(v, 95)), 2), "n": len(v),
            "load_avg_1m": round(os.getloadavg()[0], 1)}


def coverage(rows):
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    n = len(rows)
    preds = [r["pred"] for r in rows if not r["abstained"]]
    dist = Counter(preds)
    top10 = sum(k for _, k in dist.most_common(10))
    return {
        "records": n,
        "with_a_class": round(len(preds) / n, 4),
        "keyed_by_that_class": round(sum(r["keyed"] for r in rows) / n, 4),
        "distinct_classes": len(dist),
        "top10_share": round(top10 / max(len(preds), 1), 4),
        "sdg_class_share": round(sum(r["sdg"] for r in rows if not r["abstained"]) / max(len(preds), 1), 4),
        "top20": [[c, reg.label(c), k] for c, k in dist.most_common(20)],
        "top_score_mean": round(float(np.mean([r["score"] for r in rows])), 4),
    }


def main(argv=None) -> int:
    from nam_heads.impls.ontology.gaius.tools.evalset import (
        DEFAULT_CORPUS, DEFAULT_DATASET, labelled_records, sample)
    from nam_heads.impls.ontology.gaius.tools.evaluate import score
    from nam_heads.impls.ontology.gaius_ref.head import GaiusRefOntologyEncoder, model_dir_of

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--cache-dir", default=os.environ.get("NAM_GAIUS_CACHE_DIR"))
    ap.add_argument("--limit", type=int, default=0, help="test records (0 = all)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--latency", type=int, default=300, help="records timed single-threaded")
    ap.add_argument("--allow-substitute", action="store_true",
                    help="run an export that is not lightonai/ColBERT-Zero (named in the report)")
    ap.add_argument("--out", default="")
    ap.add_argument("--predictions", default="", help="per-record JSONL for spot checks")
    a = ap.parse_args(argv)
    model_dir = str(model_dir_of(a.model_dir))
    t0 = time.time()
    _activate()
    recs = sample(labelled_records("test", a.dataset, a.corpus), a.limit)
    # build the class bank once, in this process, so the workers read it from the cache
    head = GaiusRefOntologyEncoder(head_type="ontology", config=_config(model_dir, a.allow_substitute, a.cache_dir))
    ctx = get_context("spawn")
    with ctx.Pool(a.procs, initializer=_init, initargs=(model_dir, a.allow_substitute, a.cache_dir)) as pool:
        rows = pool.map(_one, recs, chunksize=32)
    out = {"head": head.config_summary(), "coverage": coverage(rows),
           "against_bfo_cco_gold_reference_only": {
               k: v for k, v in score([r["pred"] for r in rows], [r["gold"] for r in rows]).items()
               if k in ("exact", "exact_or_coarse", "abstained")}}
    lat = sample(recs, a.latency, seed=1)
    out["latency_single_thread"] = _latency(lat, model_dir, a.allow_substitute, a.cache_dir)
    with ctx.Pool(1) as p:
        out["memory"] = p.apply(_peak_rss, (model_dir, a.allow_substitute, a.cache_dir,
                                            [r["text"] for r in lat[:50]]))
    out["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=1, ensure_ascii=False)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    if a.predictions:
        from nam_heads.ontology.registry import get_registry
        reg = get_registry()
        with open(a.predictions, "w") as f:
            for r in rows:
                f.write(json.dumps({"record": r["record"], "text": r["text"], "class": r["pred"],
                                    "label": reg.label(r["pred"]), "score": round(r["score"], 4),
                                    "top8": r["top"]}, ensure_ascii=False) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
