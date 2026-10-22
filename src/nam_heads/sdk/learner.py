"""Head learning: what a head does with a value it cannot identify.

A head has two parts:

- **live**: the head itself. At ingest, when a record holds a value the head
  cannot identify and ``self.parking`` is True, the head returns
  ``parked_result(...)`` for that record in place of its addresses. The host
  takes the record out of the pipeline, queues it under each parked value, and
  sends it through again once the learner part has decided those values.
  ``parking`` is set by the host, never by the head: only when the
  implementation declares a learner part, learning is on for its slot, and the
  host can hand the record off durably. With ``parking`` False the head leaves
  the value out, as it would without learning.

- **learner**: one ``HeadLearner`` subclass in ``learner.py`` next to
  ``head.py``, declared in IMPL_META:

      IMPL_META = {..., "learner": {
          "module": "learner",              # learner.py in the package (the default)
          "version": "0.1.0",
          "gpu": {"image": "you/your-learner", "model_id": "your-learner-v1",
                  "port": 50051, "disk_gb": 40,
                  "gpu_types": ["NVIDIA GeForce RTX 4090"]}}}

  Without ``"gpu"`` the model runs in the host's process (``decide_local``).

Learning is optional: an implementation that declares no learner part is never
asked to park, and runs exactly as it did before.

What the host does with a batch of one slot's parked values (in NAM: up to
1,000 records, or the oldest waiting an hour; docs/design-head-learning.md):

1. ``pending(values)``: which values still need the model, and the decisions
   the store already holds. A value an earlier batch decided is not sent again.
2. ``build_item(item_id, value, occurrence, record, decided)`` on the value's
   records in turn, until one yields an item. ``occurrence`` is the parked
   entry the head returned (minus ``value``); ``record`` is the host's view of
   the record, ``{"key", "title", "text"}``. None: this record cannot show the
   value; a value with no item at all gets ``leave_out(value, reason)``.
3. The items go to the model: the GPU server ``gpu`` names, over HTTP
   (``GET /health`` answers ``{"ok": true}`` once the model is loaded;
   ``POST /decide {"items": [...]}`` answers ``{"answers": [...]}``, one per
   item, in order), or ``decide_local(items)`` with no ``gpu``.
4. ``decision(value, item, answer, decided)`` turns each answer into the
   value's decision: a JSON object with a ``"kind"``. ``answer`` may be None or
   malformed; the decision is then ``leave_out``, never an exception.
5. ``store_writes(decisions)``: what the host writes to the head's store,
   every write landing before any record goes through again.

The rule both parts keep: **a value with a decision never parks again**,
``leave_out`` included (its records go through without it). Otherwise a
record would loop between the head and the learner.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .head import EncodingResult

PARKED = "parked"
LEAVE_OUT = "leave_out"
MODULE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
GPU_KEYS = ("image", "model_id", "port", "disk_gb", "gpu_types", "env")


class LearnerSpecError(ValueError):
    """IMPL_META "learner" is malformed."""


@dataclass(frozen=True)
class LearnerSpec:
    """IMPL_META "learner": where the learner part is and where its model runs."""
    module: str = "learner"
    version: str = ""
    gpu: Optional[Dict[str, Any]] = field(default=None, hash=False, compare=True)

    def path(self, head: Path) -> Path:
        """The learner module's file, next to ``head`` (a head.py)."""
        return head.parent / f"{self.module}.py"


def parse_learner(raw: Any) -> Optional[LearnerSpec]:
    """IMPL_META["learner"] -> LearnerSpec, None when absent. Raises LearnerSpecError."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise LearnerSpecError("IMPL_META learner must be an object")
    unknown = set(raw) - {"module", "version", "gpu"}
    if unknown:
        raise LearnerSpecError(f"IMPL_META learner: unknown keys {sorted(unknown)}")
    module = raw.get("module", "learner")
    if not isinstance(module, str) or not MODULE_RE.match(module):
        raise LearnerSpecError(f"IMPL_META learner module must match {MODULE_RE.pattern}")
    version = raw.get("version", "")
    if not isinstance(version, str):
        raise LearnerSpecError("IMPL_META learner version must be a string")
    gpu = raw.get("gpu")
    if gpu is not None:
        gpu = dict(_check_gpu(gpu, "IMPL_META learner gpu"))
    return LearnerSpec(module=module, version=version, gpu=gpu)


def _check_gpu(gpu: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(gpu, dict):
        raise LearnerSpecError(f"{where} must be an object")
    unknown = set(gpu) - set(GPU_KEYS)
    if unknown:
        raise LearnerSpecError(f"{where}: unknown keys {sorted(unknown)} (known {list(GPU_KEYS)})")
    for k in ("image", "model_id"):
        if not isinstance(gpu.get(k), str) or not gpu[k]:
            raise LearnerSpecError(f"{where}: {k!r} must be a non-empty string")
    for k in ("port", "disk_gb"):
        if k in gpu and (not isinstance(gpu[k], int) or isinstance(gpu[k], bool) or gpu[k] <= 0):
            raise LearnerSpecError(f"{where}: {k!r} must be a positive integer")
    if "gpu_types" in gpu and (not isinstance(gpu["gpu_types"], (list, tuple))
                               or not all(isinstance(t, str) and t for t in gpu["gpu_types"])):
        raise LearnerSpecError(f"{where}: 'gpu_types' must be a list of GPU type names")
    if "env" in gpu and (not isinstance(gpu["env"], dict)
                         or not all(isinstance(k, str) and isinstance(v, str) for k, v in gpu["env"].items())):
        raise LearnerSpecError(f"{where}: 'env' must map strings to strings")
    return gpu


# ---- the live part's side ---------------------------------------------------------------------

def parked_result(head_type: str, parked: Sequence[Dict[str, Any]], **metadata: Any) -> EncodingResult:
    """The result of a record the head parks: no addresses, and ``metadata["parked"]`` listing each
    value it cannot identify with whatever the learner needs to find it again (JSON-serialisable;
    keep it small -- never the record body, which the host keeps)."""
    entries = []
    for p in parked:
        if not isinstance(p, dict) or not isinstance(p.get("value"), str) or not p["value"]:
            raise ValueError(f"a parked entry needs a non-empty 'value': {p!r}")
        json.dumps(p)                     # raises on anything the host cannot carry
        entries.append(dict(p))
    if not entries:
        raise ValueError("parked_result needs at least one parked value")
    return EncodingResult(head_type, [], dict(metadata, **{PARKED: entries}))


def parked_values(result: Any) -> Optional[List[Dict[str, Any]]]:
    """The parked entries of a head's result (an EncodingResult or its dict form), or None."""
    md = getattr(result, "metadata", None)
    if md is None and isinstance(result, dict):
        md = result.get("metadata")
    return (md or {}).get(PARKED) or None


# ---- the learner part ---------------------------------------------------------------------------

class HeadLearner(ABC):
    """The learner part of a head implementation (module docstring).

    The host builds it as ``cls(slot, store, config)``: ``store`` is a
    ``nam_heads.sdk.store.HeadStore`` on the head's state -- the same state
    the live part reads its decisions from -- and ``config`` the host's dict
    (``config["gpu"]`` overrides IMPL_META's gpu entries, e.g. the image tag a
    deployment runs).

    A subclass implements ``decision_key``, ``build_item`` and ``decision``,
    and ``decide_local`` when it declares no gpu.
    """

    def __init__(self, slot: str, store, config: Optional[Dict[str, Any]] = None):
        self.slot = slot
        self.store = store
        self.config = dict(config or {})
        self._initialize()

    def _initialize(self) -> None:
        """Load what the learner part needs (tables, artifacts). Never a GPU model."""

    # -- the store -------------------------------------------------------------------------------
    @abstractmethod
    def decision_key(self, value: str) -> str:
        """The store key a value's decision lives under (the key the live part reads)."""

    def pending(self, values: Sequence[str]) -> Tuple[List[str], Dict[str, dict]]:
        """(values that need the model, {value: decision} the store already holds). One batched read."""
        keys = {v: self.decision_key(v) for v in values}
        got = self.store.get_docs(list(keys.values())) if keys else {}
        decided = {v: got[k] for v, k in keys.items() if isinstance(got.get(k), dict)}
        return [v for v in values if v not in decided], decided

    def store_writes(self, decisions: Mapping[str, dict]) -> List[Tuple[str, bytes]]:
        """(key, value) pairs for the host to write: one JSON doc per decided value."""
        return [(self.decision_key(v), json.dumps(d, sort_keys=True, separators=(",", ":")).encode())
                for v, d in decisions.items()]

    # -- items and answers -----------------------------------------------------------------------
    @abstractmethod
    def build_item(self, item_id: str, value: str, occurrence: Dict[str, Any], record: Dict[str, Any],
                   decided: Mapping[str, dict]) -> Optional[Dict[str, Any]]:
        """The model's input for ``value`` as it occurs in ``record``, carrying ``"item": item_id``;
        None when this record cannot show it."""

    @abstractmethod
    def decision(self, value: str, item: Dict[str, Any], answer: Any,
                 decided: Mapping[str, dict]) -> Dict[str, Any]:
        """The value's decision from the model's ``answer`` to ``item``: a JSON object with a "kind".
        Never raises on a missing or malformed answer: return ``leave_out``."""

    def leave_out(self, value: str, reason: str) -> Dict[str, Any]:
        """The decision for a value the learner cannot decide: its records go through without it."""
        return {"kind": LEAVE_OUT, "value": value, "reason": reason, "decided_at": round(time.time(), 1)}

    # -- where the model runs --------------------------------------------------------------------
    def gpu_workload(self) -> Optional[Dict[str, Any]]:
        """The GPU server this learner's model runs in (IMPL_META learner gpu, then config["gpu"]),
        or None: the host calls ``decide_local``."""
        base = dict((_declared_learner(type(self)) or LearnerSpec()).gpu or {})
        base.update(self.config.get("gpu") or {})
        if not base:
            return None
        return dict(_check_gpu(base, f"{type(self).__name__} gpu"))

    def decide_local(self, items: Sequence[Dict[str, Any]]) -> List[Any]:
        """Answers for ``items`` computed in the host's process (a learner with no gpu)."""
        raise NotImplementedError(f"{type(self).__name__} declares no gpu and implements no decide_local")


def _declared_learner(cls) -> Optional[LearnerSpec]:
    """IMPL_META["learner"] of the implementation ``cls`` belongs to (its package's head.py)."""
    import sys
    pkg = cls.__module__.rpartition(".")[0]
    head = sys.modules.get(f"{pkg}.head") if pkg else None
    meta = getattr(head, "IMPL_META", None)
    if meta is None and pkg:
        spec = importlib.util.find_spec(f"{pkg}.head")
        if spec is not None and spec.origin:
            from .impl_meta import read_module_literal
            meta = read_module_literal(Path(spec.origin), ("IMPL_META",)).get("IMPL_META")
    try:
        return parse_learner((meta or {}).get("learner"))
    except LearnerSpecError:
        return None


def load_learner_class(package: str, spec: LearnerSpec):
    """Import ``<package>.<spec.module>`` and return its one HeadLearner subclass."""
    mod = importlib.import_module(f"{package}.{spec.module}")
    classes = [obj for _, obj in inspect.getmembers(mod, inspect.isclass)
               if issubclass(obj, HeadLearner) and obj is not HeadLearner and obj.__module__ == mod.__name__]
    if len(classes) != 1:
        raise RuntimeError(f"{mod.__name__}: expected one HeadLearner subclass, found "
                           f"{[c.__name__ for c in classes]}")
    return classes[0]
