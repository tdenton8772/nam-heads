"""The affordance head (and its learner part) works copied anywhere and imported under another
package name, as the plugin factory loads it (docs/design-head-plugins.md section 5), and declares
a static IMPL_META."""

import importlib

import pytest

from nam_heads.sdk.impl_meta import learner_problems
from nam_heads.sdk.learner import load_learner_class, parse_learner

from nam_heads.impls._testing.relocation import (
    addresses, assert_located_inside, head_dir, check_impl_meta, head_class, package_name,
    relocated_head,
)

SLOT = "affordance"
CONFIG = {"category_store_backend": "memory"}
SENTENCES = (
    "The company acquired a startup and launched a new product.",
    "Lincoln signed the proclamation and delivered a speech at Gettysburg.",
    "Researchers measured the temperature of the ocean and published their results.",
)
QUERY = "Who signed the proclamation?"

# The learner entry as it was before the plugin fields were added: it must not change.
LEARNER = {
    "module": "learner",
    "version": "r5",
    "gpu": {"image": "tdenton8772/nam-head-learner:v0.0.1",
            "model_id": "head-learner-affordance-r5",
            "port": 50051, "disk_gb": 40,
            "gpu_types": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090",
                          "NVIDIA L4", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX A6000"]},
}


@pytest.fixture(scope="module")
def nlp():
    from nam_heads.impls.nlp.rule import RuleNLPService
    return RuleNLPService()


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    monkeypatch.delenv("NAM_AFFORDANCE_SENSE_SEED", raising=False)
    yield from relocated_head(SLOT, tmp_path, monkeypatch)


def _run(cls, nlp):
    head = cls(head_type=SLOT, config=dict(CONFIG))
    out = [addresses(head.encode(t, {"spacy_doc": nlp.parse(t)})) for t in SENTENCES]
    out.append(addresses(head.encode_query(QUERY, {"spacy_doc": nlp.parse(QUERY)})))
    return out


def test_relocated_head_encodes_like_the_in_tree_head(relocated, nlp, tmp_path):
    from nam_heads.impls.affordance.default.head import AffordanceEncoder

    cls = head_class(relocated)
    assert cls is not AffordanceEncoder and cls.__module__ == f"{package_name(SLOT)}.head"
    got = _run(cls, nlp)
    assert got == _run(AffordanceEncoder, nlp)
    assert all(got[:len(SENTENCES)]), "every sentence names an action"
    # data/*.json load from the copy.
    assert_located_inside(package_name(SLOT), tmp_path / "pkgs" / package_name(SLOT), SLOT)


def test_relocated_predicates_find_the_rule_lemma_exceptions(relocated):
    """The rule parser's lemma exceptions are the host's (nam_heads.impls.nlp.rule), found by
    package, not by climbing out of the head's directory."""
    from nam_heads.impls.affordance.default import predicates as here

    there = importlib.import_module(f"{package_name(SLOT)}.predicates")
    assert there is not here
    exc = there._rule_exceptions()
    assert exc and exc == here._rule_exceptions()


def test_relocated_learner_part_loads_from_the_copy(relocated, tmp_path):
    pkg = package_name(SLOT)
    spec = parse_learner(relocated.IMPL_META["learner"])
    cls = load_learner_class(pkg, spec)
    assert cls.__module__ == f"{pkg}.learner"
    learner = cls(SLOT, None, None)
    # its own modules are the copy's, and its sense table loads from the copy's data/
    assert type(learner.table).__module__ == f"{pkg}.senses"
    senses = importlib.import_module(f"{pkg}.senses")
    assert senses.seed_path().resolve().is_relative_to((tmp_path / "pkgs" / pkg).resolve())
    assert learner.model == LEARNER["gpu"]["model_id"]

    from nam_heads.impls.affordance.default.learner import AffordanceLearner
    here = AffordanceLearner(SLOT, None, None)
    assert learner.decision_key("run") == here.decision_key("run")
    assert learner.table.seed_cell("acquire") == here.table.seed_cell("acquire") is not None
    assert_located_inside(pkg, tmp_path / "pkgs" / pkg, SLOT)


def test_impl_meta_reads_statically():
    meta = check_impl_meta(SLOT, "affordance-senses-v1")
    assert meta["resources"] == {"cpu": 1.0, "memory_mb": 1024, "workers": 1}
    assert meta["learner"] == LEARNER
    assert learner_problems(meta, head_dir(SLOT) / "head.py") == []
