"""Class bank MaxSim, calibrated probabilities, resolution, caches."""

import numpy as np
import pytest

from nam_heads.impls.ontology.gaius.scoring import (
    UNCLASSIFIED, ClassBank, Resolver, TextScoreCache, probabilities)
from nam_heads.ontology import bindings as B

from .conftest import FakeEncoder


def _unit(rng, n):
    v = rng.standard_normal((n, 128)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_scores_are_maxsim_mean_over_query_tokens():
    rng = np.random.default_rng(0)
    docs = [_unit(rng, 3), _unit(rng, 5), _unit(rng, 1)]
    offs = np.concatenate([[0], np.cumsum([d.shape[0] for d in docs])])
    bank = ClassBank(["a", "b", "c"], np.concatenate(docs), offs, "x")
    q = _unit(rng, 4)
    want = [float((q @ d.T).max(axis=1).mean()) for d in docs]
    assert np.allclose(bank.scores(q), want, atol=1e-6)
    assert np.all(bank.scores(np.zeros((0, 128), np.float32)) == 0)


def test_bank_rejects_bad_offsets():
    v = np.ones((3, 128), np.float32)
    with pytest.raises(ValueError):
        ClassBank(["a", "b"], v, np.array([0, 3]), "x")
    with pytest.raises(ValueError):
        ClassBank(["a", "b"], v, np.array([0, 0, 3]), "x")     # a class with no tokens


def test_probabilities_are_a_distribution_sharpened_by_temperature():
    s = np.array([0.90, 0.89, 0.80], np.float32)
    p_cold, p_warm = probabilities(s, 0.004), probabilities(s, 0.05)
    assert abs(p_cold.sum() - 1) < 1e-9 and abs(p_warm.sum() - 1) < 1e-9
    assert p_cold.argmax() == 0 and p_cold[0] > p_warm[0]


def test_resolver_accepts_the_top_class():
    r = Resolver([B.PERSON, B.ORGANIZATION], accept=0.5)
    p = r.resolve(np.array([0.8, 0.2]))
    assert (p.cls, p.backed_off, p.abstained) == (B.PERSON, False, False)
    assert p.top[0] == (B.PERSON, 0.8)


def test_resolver_backs_off_to_the_ancestor_holding_the_mass():
    city, town, village = "cco:ont00000887", "cco:ont00000071", "cco:ont00000718"
    r = Resolver([city, town, village, B.PERSON], accept=0.6)
    p = r.resolve(np.array([0.35, 0.3, 0.05, 0.3]))
    assert p.backed_off and p.cls == "cco:ont00000460"      # Local Administrative Region
    assert abs(p.confidence - 0.7) < 1e-9


def test_resolver_abstains_to_bfo_entity():
    r = Resolver([B.PERSON, B.ORGANIZATION, B.ACT], accept=0.99)
    p = r.resolve(np.array([0.4, 0.3, 0.3]))
    assert p.abstained and p.cls == UNCLASSIFIED == B.UNCLASSIFIED
    assert r.resolve(None).abstained


def test_resolver_breaks_ties_by_class_order():
    r = Resolver([B.ORGANIZATION, B.PERSON], accept=0.0)
    assert r.resolve(np.array([0.5, 0.5])).cls == B.ORGANIZATION


def test_text_cache_is_bounded_lru():
    c = TextScoreCache(2)
    c.put("a", np.zeros(1)); c.put("b", np.ones(1))
    assert c.get("a") is not None                # a is now most recent
    c.put("c", np.ones(1))
    assert c.get("b") is None and c.get("a") is not None and c.get("c") is not None
    off = TextScoreCache(0)
    off.put("a", np.zeros(1))
    assert off.get("a") is None


def test_bank_disk_cache_round_trip_and_bad_file(tmp_path):
    enc = FakeEncoder()
    texts = {"x:1": "Person. An animal.", "x:2": "City. A region."}
    a = ClassBank.build(enc, texts, ["x:1", "x:2"], "fake", str(tmp_path))
    files = list(tmp_path.glob("gaius_classbank_*.npz"))
    assert len(files) == 1
    b = ClassBank.build(enc, texts, ["x:1", "x:2"], "fake", str(tmp_path))
    assert np.array_equal(a.vecs, b.vecs) and a.digest == b.digest
    files[0].write_bytes(b"not an npz")
    c = ClassBank.build(enc, texts, ["x:1", "x:2"], "fake", str(tmp_path))   # rebuilt, not trusted
    assert np.array_equal(a.vecs, c.vecs)
    other = ClassBank.build(enc, {**texts, "x:2": "City. Changed."}, ["x:1", "x:2"], "fake", str(tmp_path))
    assert other.digest != a.digest
