"""batch_cas_merge_set: same stored result as per-key cas_merge_set, in a
constant number of KV round trips.

Production regression (2026-10-04): the affordance flush wrote each
counter doc with its own GET + CAS SET, ~30 serial session round trips per
record, and the head went from ~7 ms to 70-90 ms per record.
"""
import json
import random

from nam_heads.impls.affordance.default.storage.merge import (
    batch_cas_merge_set, cas_merge_set, encode)
from nam_heads.impls._testing.fake_kv import FakeKV


class CountingKV(FakeKV):
    """FakeKV that counts per-key and batch calls."""

    def __init__(self):
        super().__init__()
        self.gets = self.sets = self.adds = self.batch_gets = 0

    def get(self, key):
        self.gets += 1
        return super().get(key)

    def set(self, key, value, cas=0, expiration=0):
        self.sets += 1
        return super().set(key, value, cas=cas, expiration=expiration)

    def add(self, key, value, expiration=0):
        self.adds += 1
        return super().add(key, value, expiration=expiration)

    def pipeline_get(self, keys):
        self.batch_gets += 1
        return super().pipeline_get(keys)


def _doc(witness, n, ont="cco:ont00001262", kind="verb"):
    return encode({"kind": kind, "lemma": "found", "_g": {"count": {witness: n}},
                   "_go": {ont: {witness: n}}})


def _stored(kv):
    return {k: json.loads(v) for k, v in kv._data.items()}


def test_matches_per_key_path_over_random_histories():
    rng = random.Random(7)
    keys = [f"vcat::p:v{i}" for i in range(25)]
    for _ in range(20):
        a, b = FakeKV(), FakeKV()
        for _ in range(4):
            items = [(rng.choice(keys), _doc(rng.choice(["pod-0", "pod-1", "pod-2"]),
                                             rng.randint(1, 50)))
                     for _ in range(rng.randint(1, 30))]
            batch_cas_merge_set(a, items)
            for k, v in items:
                cas_merge_set(b, k, v)
        assert _stored(a) == _stored(b)


def test_constant_round_trips_without_conflicts():
    kv = CountingKV()
    items = [(f"vcat::p:v{i}", _doc("pod-0", i + 1)) for i in range(40)]
    out = batch_cas_merge_set(kv, items)
    assert all(out[k] is not None for k, _ in items)
    assert kv.batch_gets == 1
    assert kv.gets == 0  # no per-key reads
    # Second flush with higher counts: still one batch_get, no per-key reads.
    kv.batch_gets = kv.gets = 0
    items2 = [(f"vcat::p:v{i}", _doc("pod-0", i + 100)) for i in range(40)]
    batch_cas_merge_set(kv, items2)
    assert kv.batch_gets == 1 and kv.gets == 0
    assert json.loads(kv._data["vcat::p:v3"])["_g"]["count"]["pod-0"] == 103


def test_unchanged_docs_are_not_rewritten():
    # A new key is stored as given (no totals yet), exactly like cas_merge_set;
    # the second write adds the recomputed totals; from then on an unchanged
    # doc must not be rewritten.
    kv = CountingKV()
    items = [("vcat::p:a", _doc("pod-0", 5)), ("vcat::p:b", _doc("pod-0", 6))]
    batch_cas_merge_set(kv, items)
    batch_cas_merge_set(kv, items)
    cas_before = dict(kv._cas)
    batch_cas_merge_set(kv, items)
    assert kv._cas == cas_before  # nothing stored the second time


def test_lost_race_falls_back_and_keeps_both_writers():
    kv = FakeKV()
    batch_cas_merge_set(kv, [("vcat::p:x", _doc("pod-0", 3))])

    class RacingKV(FakeKV):
        """Another pod writes between our batch_get and pipeline_set_cas."""

    racing = RacingKV()
    racing._data, racing._cas, racing._cas_counter = kv._data, kv._cas, kv._cas_counter
    real_batch_get = racing.pipeline_get

    def batch_get_then_race(keys):
        snap = real_batch_get(keys)
        cas_merge_set(racing, "vcat::p:x", _doc("pod-1", 9))  # the other pod
        return snap

    racing.pipeline_get = batch_get_then_race
    batch_cas_merge_set(racing, [("vcat::p:x", _doc("pod-0", 4))])
    counts = json.loads(racing._data["vcat::p:x"])["_g"]["count"]
    assert counts == {"pod-0": 4, "pod-1": 9}


def test_duplicate_keys_in_one_batch_fold():
    kv = FakeKV()
    batch_cas_merge_set(kv, [("vcat::p:x", _doc("pod-0", 2)), ("vcat::p:x", _doc("pod-1", 5)),
                             ("vcat::p:x", _doc("pod-0", 7))])
    assert json.loads(kv._data["vcat::p:x"])["_g"]["count"] == {"pod-0": 7, "pod-1": 5}


def test_kv_without_batch_calls_uses_per_key_path():
    class PlainKV:
        def __init__(self):
            self.inner = FakeKV()

        def get(self, k):
            return self.inner.get(k)

        def set(self, k, v, cas=0, expiration=0):
            return self.inner.set(k, v, cas=cas)

        def add(self, k, v, expiration=0):
            return self.inner.add(k, v)

    kv = PlainKV()
    out = batch_cas_merge_set(kv, [("vcat::p:x", _doc("pod-0", 2))])
    assert out["vcat::p:x"] is not None
    assert json.loads(kv.inner._data["vcat::p:x"])["_g"]["count"] == {"pod-0": 2}
