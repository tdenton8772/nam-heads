"""Ontology head diagnostics: per-encode health signals and a live readout.

Per encode (``compute_diagnostics``), from the calibrated softmax distribution
over the codebook's K trained classes that decided the record's class:

- ``entropy``: normalized Shannon entropy of that distribution, in [0, 1].
  Normalized by log2(K), the number of trained classes, not by the number of
  classes with p > 0: the softmax temperature is small enough that most
  probabilities underflow to exactly 0, so the count of non-zero classes swings
  from record to record and would make a 50/50 split between two classes read
  as 1.0. Against log2(K) a one-hot distribution is 0.0 and a uniform one 1.0.
- ``max_prob``: the top class's probability.
- ``active_count``: classes with probability >= ``active_floor``.
- ``alerts``:
  ``ontology_unclassified`` -- the head emitted bfo:entity (BFO:0000001, the
  abstain class) or ``max_prob < unknown_threshold``;
  ``ontology_overactivated`` -- ``entropy > collapse_entropy_threshold`` with
  ``active_count >= min_active_for_collapse``: many classes equally plausible.

Diagnostics only read the distribution; they never change the emitted class.

``DiagnosticsWindow`` keeps the last N encodes (a bounded deque) plus lifetime
counters and turns them into the readout the head's ``get_metrics()`` publishes
(heartbeat doc -> nam-query ``/v1/admin/system-status`` and
``/v1/admin/encoder-heads``). ``merge_snapshots`` combines the readouts of
several workers into one.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from typing import Dict, Iterable, List, Optional, Sequence

UNCLASSIFIED_ALERT = "ontology_unclassified"
OVERACTIVATED_ALERT = "ontology_overactivated"
ALERTS = (UNCLASSIFIED_ALERT, OVERACTIVATED_ALERT)
HIST_BINS = 10
NORMALIZATION = "log2(K)"


def normalized_entropy(probs: Optional[Sequence[float]], n_classes: Optional[int] = None) -> float:
    """Shannon entropy of ``probs`` (renormalized to sum 1) over log2(n_classes).

    ``n_classes`` defaults to ``len(probs)``. Empty, single-class, all-zero or
    one-hot inputs are 0.0; a uniform distribution over ``n_classes`` is 1.0.
    """
    if probs is None:
        return 0.0
    n = int(n_classes) if n_classes is not None else len(probs)
    if n <= 1:
        return 0.0
    total = float(sum(probs))
    if total <= 0.0:
        return 0.0
    h = 0.0
    for p in probs:
        if p > 0.0:
            q = p / total
            h -= q * math.log2(q)
    return min(1.0, max(0.0, h / math.log2(n)))


def _entropy_np(probs) -> float:
    """normalized_entropy for a softmax ndarray (sums to 1), over log2(len)."""
    import numpy as np
    k = probs.shape[0]
    if k <= 1:
        return 0.0
    q = probs[probs > 0]
    h = float(-(q * np.log2(q)).sum())
    return min(1.0, max(0.0, h / math.log2(k)))


def compute_diagnostics(probs, *, abstained: bool, unknown_threshold: float,
                        collapse_entropy_threshold: float, min_active_for_collapse: int,
                        active_floor: float) -> Dict:
    """Diagnostics for one prediction. ``probs`` is the calibrated softmax over
    the K trained classes (ndarray or sequence), or None when the record had no
    cues (no parse / no sentence with cues)."""
    if probs is None or len(probs) == 0:
        return {"entropy": 0.0, "max_prob": 0.0, "active_count": 0,
                "alerts": [UNCLASSIFIED_ALERT]}
    if hasattr(probs, "dtype"):
        entropy = _entropy_np(probs)
        max_prob = float(probs.max())
        active = int((probs >= active_floor).sum())
    else:
        entropy = normalized_entropy(probs)
        max_prob = float(max(probs))
        active = sum(1 for p in probs if p >= active_floor)
    alerts: List[str] = []
    if abstained or max_prob < unknown_threshold:
        alerts.append(UNCLASSIFIED_ALERT)
    if entropy > collapse_entropy_threshold and active >= min_active_for_collapse:
        alerts.append(OVERACTIVATED_ALERT)
    return {"entropy": round(entropy, 4), "max_prob": round(max_prob, 4),
            "active_count": active, "alerts": alerts}


def _hist_bin(h: float) -> int:
    return min(HIST_BINS - 1, max(0, int(h * HIST_BINS)))


class DiagnosticsWindow:
    """Bounded rolling window of per-encode diagnostics plus lifetime counters.

    ``record`` is the hot path: one deque append and two counter bumps under a
    lock. Everything else is computed in ``snapshot`` (heartbeat cadence).
    """

    def __init__(self, size: int, n_emittable: int, thresholds: Dict,
                 seconds: float = 0.0, clock=time.monotonic):
        self.size = max(1, int(size))
        # Also drop entries older than this many seconds at snapshot time
        # (0 = count bound only): the window is the last `size` encodes or the
        # last `seconds`, whichever holds fewer.
        self.seconds = max(0.0, float(seconds))
        self._clock = clock
        self.n_emittable = max(1, int(n_emittable))
        self.thresholds = dict(thresholds)
        self._win: deque = deque(maxlen=self.size)  # (t, entropy, max_prob, unclassified, overactivated, cls)
        self._count = 0
        self._alerts_total = Counter()
        self._lock = threading.Lock()

    def record(self, diag: Dict, cls: str) -> None:
        alerts = diag["alerts"]
        un = UNCLASSIFIED_ALERT in alerts
        ov = OVERACTIVATED_ALERT in alerts
        with self._lock:
            self._win.append((self._clock(), diag["entropy"], diag["max_prob"], un, ov, cls))
            self._count += 1
            if un:
                self._alerts_total[UNCLASSIFIED_ALERT] += 1
            if ov:
                self._alerts_total[OVERACTIVATED_ALERT] += 1

    def __len__(self) -> int:
        return len(self._win)

    def snapshot(self) -> Dict:
        with self._lock:
            win = list(self._win)
            count = self._count
            totals = {a: int(self._alerts_total[a]) for a in ALERTS}
        if self.seconds > 0:
            cutoff = self._clock() - self.seconds
            win = [e for e in win if e[0] >= cutoff]
        n = len(win)
        hist = [0] * HIST_BINS
        ent_sum = mp_sum = 0.0
        un = ov = 0
        classes: Counter = Counter()
        for _t, h, mp, u, o, cls in win:
            hist[_hist_bin(h)] += 1
            ent_sum += h
            mp_sum += mp
            un += u
            ov += o
            classes[cls] += 1
        snap = {
            "enabled": True,
            "normalization": NORMALIZATION,
            "thresholds": dict(self.thresholds),
            "count": count,
            "alerts_total": totals,
            "window": _window_stats(n, self.size, ent_sum, mp_sum, hist, un, ov,
                                    classes, self.n_emittable),
        }
        snap["window"]["seconds"] = self.seconds
        snap["health"] = rate_status(snap["window"], self.thresholds)
        return snap


_LEVELS = ("green", "amber", "red")


def rate_status(window: Dict, thresholds: Dict) -> Dict:
    """Head status from the window's aggregate alert rates.

    ``thresholds["rates"]`` = {"unclassified": {"amber", "red"},
    "overactivated": {"amber", "red"}}. Returns {"status": worst level,
    "rates": {name: {"rate", "amber", "red", "status"}}}; a rate at or above
    a threshold takes that level. No thresholds -> green with rates shown.
    """
    rates_cfg = (thresholds or {}).get("rates") or {}
    out, worst = {}, 0
    for name, key in (("unclassified", "unclassified_rate"), ("overactivated", "overactivated_rate")):
        rate = float((window or {}).get(key, 0.0) or 0.0)
        t = rates_cfg.get(name) or {}
        amber, red = t.get("amber"), t.get("red")
        lvl = 0
        if red is not None and rate >= float(red):
            lvl = 2
        elif amber is not None and rate >= float(amber):
            lvl = 1
        worst = max(worst, lvl)
        out[name] = {"rate": round(rate, 4), "amber": amber, "red": red, "status": _LEVELS[lvl]}
    return {"status": _LEVELS[worst], "rates": out}


def _window_stats(n: int, size: int, ent_sum: float, mp_sum: float, hist: List[int],
                  un: int, ov: int, classes: Counter, n_emittable: int) -> Dict:
    top_cls, top_n = (classes.most_common(1)[0] if classes else (None, 0))
    return {
        "size": size,
        "count": n,
        "entropy_mean": round(ent_sum / n, 4) if n else 0.0,
        "max_prob_mean": round(mp_sum / n, 4) if n else 0.0,
        "entropy_hist": hist,
        "unclassified_rate": round(un / n, 4) if n else 0.0,
        "overactivated_rate": round(ov / n, 4) if n else 0.0,
        "alert_counts": {UNCLASSIFIED_ALERT: un, OVERACTIVATED_ALERT: ov},
        "class_counts": dict(classes),
        "distinct_classes": len(classes),
        # emitted-class spread over the classes the head can emit (log2 of that
        # count): 0.0 = every record got the same class
        "class_entropy": round(normalized_entropy(list(classes.values()), n_emittable), 4),
        "n_emittable": n_emittable,
        "top1_class": top_cls,
        "top1_share": round(top_n / n, 4) if n else 0.0,
    }


def merge_snapshots(snaps: Iterable[Optional[Dict]]) -> Optional[Dict]:
    """Combine ``DiagnosticsWindow.snapshot()`` readouts of several workers.

    Window means and rates are weighted by each worker's window count;
    histograms, class counts and alert counts are summed. Returns None when no
    snapshot carries diagnostics.
    """
    snaps = [s for s in snaps if isinstance(s, dict) and isinstance(s.get("window"), dict)]
    if not snaps:
        return None
    n = ent = mp = 0.0
    hist = [0] * HIST_BINS
    un = ov = size = count = 0
    totals: Counter = Counter()
    classes: Counter = Counter()
    n_emittable = 1
    for s in snaps:
        w = s["window"]
        c = int(w.get("count", 0) or 0)
        n += c
        ent += float(w.get("entropy_mean", 0.0) or 0.0) * c
        mp += float(w.get("max_prob_mean", 0.0) or 0.0) * c
        for i, v in enumerate((w.get("entropy_hist") or [])[:HIST_BINS]):
            hist[i] += int(v)
        ac = w.get("alert_counts") or {}
        un += int(ac.get(UNCLASSIFIED_ALERT, 0) or 0)
        ov += int(ac.get(OVERACTIVATED_ALERT, 0) or 0)
        classes.update({k: int(v) for k, v in (w.get("class_counts") or {}).items()})
        size += int(w.get("size", 0) or 0)
        n_emittable = max(n_emittable, int(w.get("n_emittable", 1) or 1))
        count += int(s.get("count", 0) or 0)
        totals.update({k: int(v) for k, v in (s.get("alerts_total") or {}).items()})
    n = int(n)
    merged = {
        "enabled": True,
        "normalization": snaps[0].get("normalization", NORMALIZATION),
        "thresholds": snaps[0].get("thresholds", {}),
        "workers": len(snaps),
        "count": count,
        "alerts_total": {a: int(totals[a]) for a in ALERTS},
        "window": _window_stats(n, size, ent, mp, hist, un, ov, classes, n_emittable),
    }
    merged["window"]["seconds"] = max(float((s["window"].get("seconds") or 0.0)) for s in snaps)
    merged["health"] = rate_status(merged["window"], merged["thresholds"])
    return merged
