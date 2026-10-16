"""Ontology head diagnostics: entropy, alerts, rolling readout, metrics."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nam_heads.sdk import host

# These tests run the codebook against real parses and its parser fingerprint,
# so they need the host's parser (NAM_SRC=<nam checkout>/nam, see conftest.py).
# tests/test_conformance.py covers the head without one.
pytestmark = pytest.mark.skipif(not host.has_parser(), reason="needs a host parser")

from nam_heads.impls.ontology.codebook import diagnostics as D
from nam_heads.impls.ontology.codebook.codebook import UNCLASSIFIED, Prediction
from nam_heads.impls.ontology.codebook.head import OntologyEncoder
from nam_heads.sdk.testing import parser_or_skip

PROBE = Path(__file__).resolve().parents[1] / "artifacts" / "parser_probe.json"
ENV_KEYS = ("NAM_ONTOLOGY_DIAGNOSTICS", "NAM_ONTOLOGY_UNKNOWN_THRESHOLD",
            "NAM_ONTOLOGY_COLLAPSE_ENTROPY", "NAM_ONTOLOGY_MIN_ACTIVE",
            "NAM_ONTOLOGY_ACTIVE_FLOOR", "NAM_ONTOLOGY_DIAG_WINDOW")
THR = dict(unknown_threshold=0.25, collapse_entropy_threshold=0.8,
           min_active_for_collapse=5, active_floor=0.02)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def nlp():
    return parser_or_skip()


@pytest.fixture(scope="module")
def parsed(nlp):
    texts = json.loads(PROBE.read_text())["texts"][:200]
    return [(t, nlp.parse(t)) for t in texts]


# ---- normalized_entropy --------------------------------------------------
@pytest.mark.parametrize("probs", [None, [], [1.0], [0.0, 0.0, 0.0]])
def test_entropy_degenerate_inputs_are_zero(probs):
    assert D.normalized_entropy(probs) == 0.0


def test_entropy_uniform_is_one_and_one_hot_is_zero():
    assert D.normalized_entropy([0.25] * 4) == pytest.approx(1.0)
    assert D.normalized_entropy([0.0, 1.0, 0.0, 0.0]) == 0.0
    assert D.normalized_entropy([3, 3, 3]) == pytest.approx(1.0)   # renormalized


def test_entropy_is_normalized_by_k_not_by_nonzero_count():
    # a 50/50 split among 46 classes is 1 bit of log2(46), not 1.0
    p = [0.0] * 46
    p[0] = p[1] = 0.5
    assert D.normalized_entropy(p) == pytest.approx(1 / math.log2(46))
    assert D.normalized_entropy([0.5, 0.5], n_classes=46) == pytest.approx(1 / math.log2(46))


def test_numpy_and_list_paths_agree():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.full(46, 0.3)).astype(np.float32)
        p[rng.random(46) < 0.3] = 0.0
        p /= p.sum()
        a = D.compute_diagnostics(p, abstained=False, **THR)
        b = D.compute_diagnostics(p.tolist(), abstained=False, **THR)
        assert a["entropy"] == pytest.approx(b["entropy"], abs=1e-4)
        assert a["active_count"] == b["active_count"]


# ---- alerts --------------------------------------------------------------
def test_no_distribution_is_unclassified():
    d = D.compute_diagnostics(None, abstained=True, **THR)
    assert d == {"entropy": 0.0, "max_prob": 0.0, "active_count": 0,
                 "alerts": [D.UNCLASSIFIED_ALERT]}


def test_unclassified_threshold_boundary():
    def at(top):
        p = np.full(10, (1 - top) / 9)
        p[0] = top
        return D.compute_diagnostics(p, abstained=False, **THR)["alerts"]
    assert D.UNCLASSIFIED_ALERT in at(0.24)
    assert D.UNCLASSIFIED_ALERT not in at(0.26)


def test_abstain_class_is_unclassified_even_when_confident():
    p = np.array([0.9, 0.1])
    assert D.compute_diagnostics(p, abstained=True, **THR)["alerts"] == [D.UNCLASSIFIED_ALERT]
    assert D.compute_diagnostics(p, abstained=False, **THR)["alerts"] == []


def test_overactivated_needs_entropy_and_enough_active_classes():
    flat = np.full(10, 0.1)                  # entropy 1.0, 10 active
    assert D.OVERACTIVATED_ALERT in D.compute_diagnostics(flat, abstained=False, **THR)["alerts"]
    few = np.zeros(10)
    few[:4] = 0.25                           # entropy log2(4)/log2(10)=0.60
    assert D.OVERACTIVATED_ALERT not in D.compute_diagnostics(few, abstained=False, **THR)["alerts"]
    # high entropy but only 4 classes above the floor
    d = D.compute_diagnostics(few, abstained=False, **dict(THR, collapse_entropy_threshold=0.5))
    assert d["active_count"] == 4 and D.OVERACTIVATED_ALERT not in d["alerts"]
    d = D.compute_diagnostics(few, abstained=False,
                              **dict(THR, collapse_entropy_threshold=0.5, min_active_for_collapse=4))
    assert D.OVERACTIVATED_ALERT in d["alerts"]


# ---- rolling window ------------------------------------------------------
def _diag(h, mp=0.9, alerts=()):
    return {"entropy": h, "max_prob": mp, "active_count": 1, "alerts": list(alerts)}


def test_window_is_bounded_and_counts_are_lifetime():
    w = D.DiagnosticsWindow(50, 10, {})
    for i in range(500):
        w.record(_diag(0.05, alerts=[D.UNCLASSIFIED_ALERT] if i % 2 else []), f"C{i % 3}")
    assert len(w) == 50
    s = w.snapshot()
    assert s["count"] == 500
    assert s["alerts_total"][D.UNCLASSIFIED_ALERT] == 250
    win = s["window"]
    assert win["count"] == 50 and win["size"] == 50
    assert sum(win["entropy_hist"]) == 50 and win["entropy_hist"][0] == 50
    assert win["unclassified_rate"] == pytest.approx(0.5)
    assert sum(win["class_counts"].values()) == 50
    json.dumps(s)


def test_window_class_spread():
    w = D.DiagnosticsWindow(100, 4, {})
    for _ in range(10):
        w.record(_diag(0.0), "A")
    s = w.snapshot()["window"]
    assert s["class_entropy"] == 0.0 and s["top1_share"] == 1.0 and s["top1_class"] == "A"
    for c in "BCD":
        for _ in range(10):
            w.record(_diag(1.0, alerts=[D.OVERACTIVATED_ALERT]), c)
    s = w.snapshot()["window"]
    assert s["class_entropy"] == pytest.approx(1.0)
    assert s["top1_share"] == pytest.approx(0.25)
    assert s["entropy_hist"][9] == 30           # entropy 1.0 lands in the last bin
    assert s["overactivated_rate"] == pytest.approx(0.75)


def test_merge_snapshots():
    a, b = D.DiagnosticsWindow(10, 4, {}), D.DiagnosticsWindow(10, 4, {})
    for _ in range(4):
        a.record(_diag(0.2), "A")
    for _ in range(12):
        b.record(_diag(0.6, alerts=[D.UNCLASSIFIED_ALERT]), "B")
    m = D.merge_snapshots([a.snapshot(), b.snapshot(), None, {}])
    assert m["workers"] == 2 and m["count"] == 16
    w = m["window"]
    assert w["count"] == 14                     # b's window holds 10 of its 12
    assert w["entropy_mean"] == pytest.approx((0.2 * 4 + 0.6 * 10) / 14, abs=1e-4)
    assert w["class_counts"] == {"A": 4, "B": 10}
    assert m["alerts_total"][D.UNCLASSIFIED_ALERT] == 12
    assert D.merge_snapshots([None]) is None


# ---- the head ------------------------------------------------------------
def test_encode_and_encode_query_carry_diagnostics(nlp):
    enc = OntologyEncoder(head_type="ontology")
    text = "Albert Einstein was a German-born theoretical physicist."
    for res in (enc.encode(text, {"spacy_doc": nlp.parse(text)}),
                enc.encode_query(text, {"spacy_doc": nlp.parse(text)})):
        d = res.metadata["diagnostics"]
        assert set(d) == {"entropy", "max_prob", "active_count", "alerts"}
        assert 0.0 <= d["entropy"] <= 1.0 and 0.0 < d["max_prob"] <= 1.0
        json.dumps(res.metadata)


def test_head_abstain_is_unclassified():
    enc = OntologyEncoder(head_type="ontology")
    res = enc.encode("Paris is the capital of France.", {})      # no parse
    assert res.addresses[0].concept == UNCLASSIFIED
    assert res.metadata["diagnostics"]["alerts"] == [D.UNCLASSIFIED_ALERT]
    # an abstained prediction that still had a distribution
    meta = {}
    enc._diagnose(Prediction(UNCLASSIFIED, 0.9, [], abstained=True,
                             probs=np.array([0.9, 0.1], np.float32)), meta)
    assert D.UNCLASSIFIED_ALERT in meta["diagnostics"]["alerts"]


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("NAM_ONTOLOGY_UNKNOWN_THRESHOLD", "0.5")
    monkeypatch.setenv("NAM_ONTOLOGY_COLLAPSE_ENTROPY", "0.3")
    monkeypatch.setenv("NAM_ONTOLOGY_MIN_ACTIVE", "2")
    monkeypatch.setenv("NAM_ONTOLOGY_ACTIVE_FLOOR", "0.1")
    monkeypatch.setenv("NAM_ONTOLOGY_DIAG_WINDOW", "7")
    enc = OntologyEncoder(head_type="ontology")
    assert (enc.unknown_threshold, enc.collapse_entropy_threshold,
            enc.min_active_for_collapse, enc.active_floor) == (0.5, 0.3, 2, 0.1)
    assert enc.diagnostics.size == 7
    # explicit head config wins over the environment
    enc = OntologyEncoder(head_type="ontology", config={"unknown_threshold": 0.1})
    assert enc.unknown_threshold == 0.1


@pytest.mark.parametrize("how", ["env", "config"])
def test_disabled(monkeypatch, nlp, how):
    if how == "env":
        monkeypatch.setenv("NAM_ONTOLOGY_DIAGNOSTICS", "0")
        enc = OntologyEncoder(head_type="ontology")
    else:
        enc = OntologyEncoder(head_type="ontology", config={"enable_diagnostics": False})
    text = "The Beatles were an English rock band formed in Liverpool in 1960."
    res = enc.encode(text, {"spacy_doc": nlp.parse(text)})
    assert "diagnostics" not in res.metadata
    m = enc.get_metrics()
    assert "diagnostics" not in m and "ontology_entropy_mean" not in m
    assert enc.config_summary()["diagnostics"] is False


def _strip(res):
    meta = {k: v for k, v in res.metadata.items() if k != "diagnostics"}
    return json.dumps({"addresses": [(a.head_type, a.concept, a.confidence, a.metadata)
                                     for a in res.addresses], "meta": meta},
                      sort_keys=True, default=str)


def test_classification_identical_with_diagnostics_on_and_off(parsed):
    on = OntologyEncoder(head_type="ontology", config={"enable_diagnostics": True})
    off = OntologyEncoder(head_type="ontology", config={"enable_diagnostics": False})
    for text, doc in parsed:
        for fn in ("encode", "encode_query"):
            a = getattr(on, fn)(text, {"spacy_doc": doc})
            b = getattr(off, fn)(text, {"spacy_doc": doc})
            assert _strip(a) == _strip(b), text
            assert "diagnostics" in a.metadata and "diagnostics" not in b.metadata


def test_metrics_publish_the_aggregates(parsed):
    enc = OntologyEncoder(head_type="ontology", config={"diagnostics_window": 150})
    emitted = [enc.encode(t, {"spacy_doc": d}).addresses[0].concept for t, d in parsed]
    m = enc.get_metrics()
    json.dumps(m)
    s = m["diagnostics"]
    assert s["count"] == len(parsed)
    w = s["window"]
    assert w["count"] == 150 and sum(w["entropy_hist"]) == 150
    assert len(w["entropy_hist"]) == 10
    assert sum(w["class_counts"].values()) == 150
    tail = emitted[-150:]
    assert w["class_counts"] == {c: tail.count(c) for c in set(tail)}
    assert w["top1_share"] == pytest.approx(max(tail.count(c) for c in set(tail)) / 150, abs=1e-4)
    assert 0.0 <= w["class_entropy"] <= 1.0
    assert w["n_emittable"] == len(enc.emittable_classes())
    assert set(s["alerts_total"]) == {D.UNCLASSIFIED_ALERT, D.OVERACTIVATED_ALERT}
    for k in ("ontology_entropy_mean", "ontology_unclassified_rate",
              "ontology_overactivated_rate", "ontology_class_entropy", "ontology_top1_share"):
        assert isinstance(m[k], float)
    assert m["ontology_entropy_mean"] == w["entropy_mean"]


# ---- head status from aggregate alert rates ------------------------------

from nam_heads.impls.ontology.codebook.diagnostics import rate_status  # noqa: E402

RATES = {"rates": {"unclassified": {"amber": 0.25, "red": 0.5},
                   "overactivated": {"amber": 0.25, "red": 0.5}}}


@pytest.mark.parametrize("un,ov,want", [
    (0.0, 0.0, "green"), (0.24, 0.10, "green"), (0.25, 0.0, "amber"),
    (0.10, 0.49, "amber"), (0.50, 0.0, "red"), (0.30, 0.60, "red"),
])
def test_rate_status_levels(un, ov, want):
    h = rate_status({"unclassified_rate": un, "overactivated_rate": ov}, RATES)
    assert h["status"] == want
    assert h["rates"]["unclassified"]["rate"] == un
    assert h["rates"]["overactivated"]["amber"] == 0.25


def test_rate_status_without_thresholds_is_green():
    h = rate_status({"unclassified_rate": 0.9}, {})
    assert h["status"] == "green" and h["rates"]["unclassified"]["red"] is None


def test_rate_thresholds_from_env(monkeypatch):
    monkeypatch.setenv("NAM_ONTOLOGY_RATE_UNCLASSIFIED_AMBER", "0.10")
    monkeypatch.setenv("NAM_ONTOLOGY_RATE_OVERACTIVATED_RED", "0.90")
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    h = OntologyEncoder(head_type="ontology")
    t = h.diagnostics.thresholds["rates"]
    assert t["unclassified"]["amber"] == 0.10 and t["overactivated"]["red"] == 0.90
    assert t["unclassified"]["red"] == 0.50          # default kept
    snap = h.get_metrics()["diagnostics"]
    assert snap["health"]["status"] in ("green", "amber", "red")


def test_merged_snapshot_recomputes_health():
    from nam_heads.impls.ontology.codebook.diagnostics import DiagnosticsWindow, merge_snapshots
    th = {"unknown": 0.25, **RATES}
    a, b = DiagnosticsWindow(10, 5, th), DiagnosticsWindow(10, 5, th)
    for _ in range(6):
        a.record({"entropy": 0.1, "max_prob": 0.9, "alerts": ["ontology_unclassified"]}, "c1")
    for _ in range(4):
        b.record({"entropy": 0.1, "max_prob": 0.9, "alerts": []}, "c1")
    assert a.snapshot()["health"]["status"] == "red"       # 6/6
    assert b.snapshot()["health"]["status"] == "green"     # 0/4
    m = merge_snapshots([a.snapshot(), b.snapshot()])
    assert m["window"]["unclassified_rate"] == 0.6
    assert m["health"]["status"] == "red"
    assert m["health"]["rates"]["unclassified"]["red"] == 0.5


def test_window_is_last_n_or_last_seconds_whichever_fewer():
    from nam_heads.impls.ontology.codebook.diagnostics import DiagnosticsWindow
    now = [1000.0]
    w = DiagnosticsWindow(5, 3, {**RATES}, seconds=60, clock=lambda: now[0])
    for i in range(4):                                   # 4 old unclassified encodes
        w.record({"entropy": 0.1, "max_prob": 0.9, "alerts": ["ontology_unclassified"]}, "c")
    now[0] += 120                                        # they age out of the 60 s window
    for i in range(2):
        w.record({"entropy": 0.1, "max_prob": 0.9, "alerts": []}, "c")
    snap = w.snapshot()
    assert snap["window"]["count"] == 2                  # time bound wins (5 would keep all)
    assert snap["window"]["unclassified_rate"] == 0.0
    assert snap["window"]["seconds"] == 60
    assert snap["health"]["status"] == "green"
    assert snap["count"] == 6                            # lifetime counter unaffected
    for i in range(10):
        w.record({"entropy": 0.1, "max_prob": 0.9, "alerts": []}, "c")
    assert w.snapshot()["window"]["count"] == 5          # count bound wins


def test_window_defaults_are_short(monkeypatch):
    monkeypatch.delenv("NAM_ONTOLOGY_DIAG_WINDOW", raising=False)
    monkeypatch.delenv("NAM_ONTOLOGY_DIAG_WINDOW_SECONDS", raising=False)
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    h = OntologyEncoder(head_type="ontology")
    assert h.diagnostics.size == 1000 and h.diagnostics.seconds == 300.0
