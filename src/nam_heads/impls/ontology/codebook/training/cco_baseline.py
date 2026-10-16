"""Score the retired sketch ontology head on the codebook's test records.

Same records, same production text, same parser, same record-level metrics
as ``cco_train``. Three steps, because the sketch head only exists in an older
checkout:

    # 1. this tree: the test records' production text
    python nam/nam_heads/impls/ontology/codebook/training/cco_baseline.py export \\
        --dataset $NAM_DATA/ontology/cco_sentences.jsonl \\
        --corpus $NAM_DATA/wikipedia_abstracts.jsonl \\
        --codebook nam/nam_heads/impls/ontology/codebook/artifacts --out /tmp/test_texts.jsonl
    # 2. a checkout with the sketch head (e.g. git worktree add --detach /tmp/legacy 4c76a230)
    NAM_SKIP_HEAD_NLP=1 PYTHONPATH=/tmp/legacy/nam python \\
        nam/nam_heads/impls/ontology/codebook/training/cco_baseline.py score \\
        --in /tmp/test_texts.jsonl --out /tmp/sketch_preds.jsonl
    # 3. this tree: metrics
    python nam/nam_heads/impls/ontology/codebook/training/cco_baseline.py compare \\
        --in /tmp/sketch_preds.jsonl

The sketch head's category maps to a class through SKETCH_CLASS; an ancestor of
the true class counts as correct-but-coarse; bfo:entity (concept/OTHER) as
abstained. Only the standard library is imported at module level, so step 2
runs against the old tree.
"""

import argparse
import json
import os
import sys
from collections import Counter

# The sketch head's runtime categories -> the class each is bound to.
SKETCH_CLASS = {
    "person": "cco:ont00001262", "group": "cco:ont00000914",
    "organization": "cco:ont00001180", "object": "cco:ont00000995",
    "location": "cco:ont00000472", "event": "BFO:0000015", "action": "cco:ont00000005",
    "concept": "BFO:0000001",
}


def export(a):
    from nam_heads.impls.ontology.codebook.training.cco_dataset import production_text
    from nam_heads.impls.ontology.codebook.training.cco_train import split_of

    meta = json.load(open(os.path.join(a.codebook, "cco_codebook.json")))
    rollups, classes = meta["rollups"], set(meta["classes"])
    gold = {}
    for line in open(os.path.expanduser(a.dataset)):
        r = json.loads(line)
        if split_of(r["record"]) == "test":
            y = rollups.get(r["label"], r["label"])
            if y in classes:
                gold[r["record"]] = y
    with open(a.out, "w") as out:
        for i, line in enumerate(open(os.path.expanduser(a.corpus))):
            if i in gold:
                rec = json.loads(line)
                out.write(json.dumps({"record": i, "gold": gold[i],
                                      "text": production_text(rec["url"], rec["text"])}) + "\n")


def score(a):
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    from nam_heads.sdk.host import parser as host_parser

    enc = OntologyEncoder(head_type="ontology", config={"require_spacy": False})
    nlp = host_parser()
    with open(a.out, "w") as out:
        for line in open(a.inp):
            r = json.loads(line)
            res = enc.encode(r["text"], {"spacy_doc": nlp.parse(r["text"])})
            out.write(json.dumps({"record": r["record"], "gold": r["gold"],
                                  "category": res.addresses[0].concept}) + "\n")


def compare(a):
    from nam_heads.ontology.registry import get_registry

    reg = get_registry()
    n = exact = coarse = abst = 0
    dist = Counter()
    for line in open(a.inp):
        r = json.loads(line)
        cls = SKETCH_CLASS.get(r["category"], "BFO:0000001")
        dist[reg.label(cls)] += 1
        n += 1
        if cls == "BFO:0000001":
            abst += 1
        elif cls == r["gold"]:
            exact += 1
            coarse += 1
        elif reg.is_a(r["gold"], cls):
            coarse += 1
    print(json.dumps({"n": n, "exact": round(exact / n, 4), "exact_or_coarse": round(coarse / n, 4),
                      "abstained": round(abst / n, 4),
                      "accepted_precision": round(coarse / max(n - abst, 1), 4),
                      "predicted": dist.most_common(10)}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--dataset", required=True)
    e.add_argument("--corpus", required=True)
    e.add_argument("--codebook", required=True)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args(argv)
    {"export": export, "score": score, "compare": compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
