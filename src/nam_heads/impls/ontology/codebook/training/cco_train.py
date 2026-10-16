"""Train, calibrate and evaluate the CCO ontology codebook.

    python -m nam_heads.impls.ontology.codebook.training.cco_train \
        --dataset $NAM_DATA/ontology/cco_sentences.jsonl \
        --out nam/nam_heads/impls/ontology/codebook/artifacts

Split: by record, deterministic (sha1 of the record id): 80% train, 10%
validation (temperature and acceptance threshold), 10% test (reported only).

Classes: a class needs ``--min-support`` training records. A class below it is
rolled up to its nearest registry ancestor that has (or accumulates) the
support; the roll-ups are written to the artifact. Nothing is dropped silently.

Centroids: sentence vectors are IDF-weighted (weights from the training split,
shipped in the artifact and applied at inference) and renormalised; per class,
spherical k-means (k grows with the class's size, capped); then LVQ refines
all centroids discriminatively (own-class centroid pulled in, nearest
other-class centroid pushed out, for rows misclassified or near the boundary).

Calibration on validation:
- temperature: minimises the negative log-likelihood of the true class;
- acceptance threshold (on validation records' first sentences, which is
  what decides a record's class): the lowest probability at which accepted
  top-1 predictions reach ``--precision`` (default 0.85);
- margin ratio: below the threshold, the smallest top/runner-up ratio at which
  the extra accepted predictions still reach ``--precision``.

Reported on test (sentence and record level): top-1 accuracy over all rows,
accuracy counting an ancestor of the true class as correct-but-coarse,
abstention rate, per-class precision/recall, confusion pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import sparse

from nam_heads.ontology.registry import get_registry

from .. import features
from ..codebook import UNCLASSIFIED, Codebook, parser_fingerprint


def split_of(record: int) -> str:
    h = int(hashlib.sha1(str(record).encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def to_matrix(rows: List[dict]) -> sparse.csr_matrix:
    data, ind, ptr = [], [], [0]
    for r in rows:
        acc: Dict[int, float] = defaultdict(float)
        for key, w in r["cues"]:
            idx, sign = features._bucket(key)
            acc[idx] += sign * w
        norm = float(np.sqrt(sum(v * v for v in acc.values()))) or 1.0
        for idx, v in acc.items():
            ind.append(idx)
            data.append(v / norm)
        ptr.append(len(ind))
    return sparse.csr_matrix((np.array(data, np.float32), np.array(ind, np.int32),
                              np.array(ptr, np.int64)), shape=(len(rows), features.DIM))


def rollup(counts: Counter, min_support: int) -> Dict[str, str]:
    """Map each label to a class with enough support: itself or an ancestor.

    Repeatedly moves the deepest class short of support into its direct
    parent (the deepest ancestor, ties by CURIE) until every remaining class
    has the support or is the root, which is not trained.
    """
    reg = get_registry()
    target = {c: c for c in counts}
    support = Counter(counts)
    while True:
        short = [c for c in support if 0 < support[c] < min_support and c != UNCLASSIFIED]
        if not short:
            return target
        c = max(short, key=lambda c: (reg.depth(c), c))
        parents = sorted((p for p in reg.ancestors(c) if p != UNCLASSIFIED),
                         key=lambda p: (-reg.depth(p), p))
        nv = parents[0] if parents else UNCLASSIFIED
        support[nv] += support[c]
        support[c] = 0
        for k, v in target.items():
            if v == c:
                target[k] = nv


def idf_weights(X: sparse.csr_matrix) -> np.ndarray:
    """Smoothed inverse document frequency per bucket (>= 1)."""
    df = np.asarray((X != 0).sum(axis=0)).ravel()
    return (np.log((1.0 + X.shape[0]) / (1.0 + df)) + 1.0).astype(np.float32)


def weighted(X: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    """Rows weighted by idf and renormalised -- exactly Codebook.weight()."""
    X = X.multiply(idf).tocsr()
    n = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    n[n == 0] = 1.0
    return (sparse.diags(1.0 / n) @ X).tocsr()


def lvq(C: np.ndarray, cc: np.ndarray, X: sparse.csr_matrix, y: np.ndarray,
        epochs: int = 10, lr: float = 0.05, margin: float = 0.1,
        batch: int = 512, seed: int = 0) -> np.ndarray:
    """Discriminative refinement of the centroids (mini-batch LVQ2.1-style).

    For each row, the best centroid of its own class is pulled towards it and
    the best centroid of any other class pushed away, whenever the row is
    misclassified or within ``margin`` of the boundary. Centroids stay unit.
    """
    rng = np.random.default_rng(seed)
    C = C.copy()
    for _ in range(epochs):
        order = rng.permutation(X.shape[0])
        for b in range(0, len(order), batch):
            ib = order[b:b + batch]
            Xb, yb = X[ib], y[ib]
            S = np.asarray(Xb @ C.T)
            same = cc[None, :] == yb[:, None]
            Sp = np.where(same, S, -9.0)
            Sn = np.where(~same, S, -9.0)
            jp, jn = Sp.argmax(axis=1), Sn.argmax(axis=1)
            r = np.arange(len(ib))
            act = (Sn[r, jn] - Sp[r, jp]) > -margin
            if not act.any():
                continue
            Xa, n = Xb[act], int(act.sum())
            P = sparse.csr_matrix((np.ones(n), (jp[act], np.arange(n))), shape=(len(C), n))
            N = sparse.csr_matrix((np.ones(n), (jn[act], np.arange(n))), shape=(len(C), n))
            C += lr * (P @ Xa).toarray() - lr * (N @ Xa).toarray()
            C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
        lr *= 0.75
    return C.astype(np.float32)


def spherical_kmeans(X: sparse.csr_matrix, k: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    C = X[rng.choice(n, size=min(k, n), replace=False)].toarray()
    for _ in range(iters):
        sims = X @ C.T                                   # (n, k) dense
        assign = np.asarray(sims.argmax(axis=1)).ravel()
        newC = np.zeros_like(C)
        for j in range(C.shape[0]):
            m = assign == j
            if m.any():
                newC[j] = np.asarray(X[m].sum(axis=0)).ravel()
            else:
                newC[j] = C[j]
        newC /= np.maximum(np.linalg.norm(newC, axis=1, keepdims=True), 1e-9)
        if np.allclose(newC, C, atol=1e-5):
            break
        C = newC
    return C.astype(np.float32)


def scores_matrix(cb: Codebook, X: sparse.csr_matrix) -> np.ndarray:
    sims = np.asarray((X @ cb.centroids.T))              # (n, K)
    out = np.full((X.shape[0], len(cb.classes)), -1.0, dtype=np.float32)
    for k, ci in enumerate(cb.centroid_class):
        np.maximum(out[:, ci], sims[:, k], out=out[:, ci])
    return out


def softmax(S: np.ndarray, T: float) -> np.ndarray:
    Z = S / T
    Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def evaluate(cb: Codebook, P: np.ndarray, gold: List[str]) -> dict:
    reg = get_registry()
    n = len(gold)
    exact = coarse = abst = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion = Counter()
    for i in range(n):
        pred = cb.resolve(P[i])
        g = gold[i]
        if pred.abstained:
            abst += 1
            fn[g] += 1
            continue
        if pred.cls == g:
            exact += 1
            coarse += 1
            tp[g] += 1
        elif reg.is_a(g, pred.cls):
            coarse += 1          # correct but coarse
            fn[g] += 1
        else:
            fp[pred.cls] += 1
            fn[g] += 1
            confusion[(g, pred.cls)] += 1
    per_class = {}
    for c in sorted(set(gold) | set(fp)):
        p = tp[c] / max(tp[c] + fp[c], 1)
        r = tp[c] / max(tp[c] + fn[c], 1)
        per_class[c] = {"label": reg.label(c), "precision": round(p, 3), "recall": round(r, 3),
                        "support": int(sum(1 for g in gold if g == c))}
    return {
        "n": n,
        "exact": round(exact / max(n, 1), 4),
        "exact_or_coarse": round(coarse / max(n, 1), 4),
        "abstained": round(abst / max(n, 1), 4),
        "accepted_precision": round(coarse / max(n - abst, 1), 4),
        "per_class": per_class,
        "top_confusions": [[reg.label(a), reg.label(b), c] for (a, b), c in confusion.most_common(15)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-support", type=int, default=150)
    ap.add_argument("--precision", type=float, default=0.85)
    ap.add_argument("--max-k", type=int, default=8)
    ap.add_argument("--per-centroid", type=int, default=400, help="sentences per centroid")
    ap.add_argument("--lvq-epochs", type=int, default=10)
    a = ap.parse_args(argv)
    t0 = time.time()
    reg = get_registry()

    rows = [json.loads(l) for l in Path(a.dataset).expanduser().open()]
    for r in rows:
        r["split"] = split_of(r["record"])
    train = [r for r in rows if r["split"] == "train"]
    rec_labels = {r["record"]: r["label"] for r in train}
    target = rollup(Counter(rec_labels.values()), a.min_support)
    for r in rows:
        r["y"] = target.get(r["label"], r["label"])
    # a label first seen outside train rolls up through the registry too
    classes = sorted({r["y"] for r in train if r["y"] != UNCLASSIFIED})
    cls_index = {c: i for i, c in enumerate(classes)}
    print(f"{len(rows)} sentences, {len(classes)} classes after roll-up", file=sys.stderr)

    tr = [r for r in train if r["y"] in cls_index]
    Xtr_raw = to_matrix(tr)
    idf = idf_weights(Xtr_raw)
    Xtr = weighted(Xtr_raw, idf)
    ytr = np.array([cls_index[r["y"]] for r in tr])
    cents, cc = [], []
    for ci in range(len(classes)):
        Xc = Xtr[ytr == ci]
        k = int(np.clip(Xc.shape[0] // a.per_centroid, 1, a.max_k))
        C = spherical_kmeans(Xc, k)
        cents.append(C)
        cc += [ci] * C.shape[0]
    C = np.vstack(cents)
    cc = np.array(cc)
    C = lvq(C, cc, Xtr, ytr, epochs=a.lvq_epochs)
    cb = Codebook(C, cc, classes, temperature=1.0, accept=0.5, meta={}, idf=idf)

    def mapped(rs):
        keep = [r for r in rs if r["y"] in cls_index]
        return keep, weighted(to_matrix(keep), idf), [r["y"] for r in keep]

    val, Xv, gv = mapped([r for r in rows if r["split"] == "val"])
    test, Xt, gt = mapped([r for r in rows if r["split"] == "test"])
    Sv = scores_matrix(cb, Xv)
    yv = np.array([cls_index[g] for g in gv])
    best = None
    for T in np.geomspace(0.005, 1.0, 40):
        Pv = softmax(Sv, T)
        nll = -np.mean(np.log(Pv[np.arange(len(yv)), yv] + 1e-12))
        if best is None or nll < best[0]:
            best = (nll, float(T))
    cb.temperature = best[1]
    # Threshold and margin decide the RECORD's class, which comes from its
    # first real sentence (Codebook.predict_spans): calibrate on those.
    first_of = {}
    for i, r in enumerate(val):
        if r["record"] not in first_of or r["sent"] < val[first_of[r["record"]]]["sent"]:
            first_of[r["record"]] = i
    vf = np.array(sorted(first_of.values()))
    Pv = softmax(Sv[vf], cb.temperature)
    yv = yv[vf]
    top = Pv.max(axis=1)
    correct = Pv.argmax(axis=1) == yv
    accept = 0.99
    for thr in np.linspace(0.05, 0.99, 95):
        m = top >= thr
        if m.sum() and correct[m].mean() >= a.precision:
            accept = float(thr)
            break
    cb.accept = accept
    # margin rule: below the threshold, accept the top class when it beats the
    # runner-up by a ratio; the smallest ratio whose margin-accepted
    # predictions still reach the precision target on validation
    srt = np.sort(Pv, axis=1)
    ratio = srt[:, -1] / np.maximum(srt[:, -2], 1e-9)
    below = top < accept
    margin = 0.0
    for r in (2.0, 3.0, 4.0, 6.0, 8.0):
        m = below & (ratio >= r)
        if m.sum() >= 20 and correct[m].mean() >= a.precision:
            margin = r
            break
    cb.margin = margin

    Pt = softmax(scores_matrix(cb, Xt), cb.temperature)
    sent_eval = evaluate(cb, Pt, gt)
    # record level: the record's first real sentence, as Codebook.predict_spans
    by_rec = defaultdict(list)
    for i, r in enumerate(test):
        by_rec[r["record"]].append(i)
    firsts = [min(ix, key=lambda i: test[i]["sent"]) for ix in by_rec.values()]
    Pr = Pt[firsts]
    gr = [test[i]["y"] for i in firsts]
    rec_eval = evaluate(cb, Pr, gr)

    cb.meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parser": parser_fingerprint(),
        "dataset": {"path": str(a.dataset), "sentences": len(rows),
                    "sha256": hashlib.sha256(Path(a.dataset).expanduser().read_bytes()).hexdigest()},
        "training": {"min_support": a.min_support, "precision_target": a.precision,
                     "max_k": a.max_k, "per_centroid": a.per_centroid,
                     "lvq_epochs": a.lvq_epochs, "idf": "smoothed, training split",
                     "centroids": int(cb.centroids.shape[0])},
        "rollups": {c: t for c, t in sorted(target.items()) if c != t},
        "class_labels": {c: reg.label(c) for c in classes},
        "eval": {"sentence": sent_eval, "record": rec_eval,
                 "validation_nll": round(best[0], 4)},
    }
    cb.save(Path(a.out))
    summary = {k: v for k, v in sent_eval.items() if k not in ("per_class",)}
    rsum = {k: v for k, v in rec_eval.items() if k not in ("per_class",)}
    print(json.dumps({"classes": len(classes), "T": cb.temperature, "accept": cb.accept,
                      "margin": cb.margin,
                      "sentence": summary, "record": rsum,
                      "seconds": round(time.time() - t0, 1)}, indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
