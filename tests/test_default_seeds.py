"""The default heads' seeds (export ``seed``): what the host writes to each head's store in the session bucket
where the key is absent. The key formats are the stores' own (their storage/couchbase_kv_store.py)."""
import json

import pytest


def _seed(slot):
    try:
        mod = __import__(f"nam_heads.impls.{slot}.default.exports", fromlist=["seed"])
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "src":
            pytest.skip(f"needs the NAM host ({e.name}); set NAM_SRC")
        raise
    return mod.seed()


def test_attribute_seed_is_category_entries():
    entries = _seed("attribute")
    keys = {k for k, _ in entries}
    assert len(keys) > 100 and "acat::large" in keys
    assert any(k.count("::") == 2 for k in keys)                  # ontology-scoped acat::<lemma>::<ontology>
    for k, v in entries:
        assert k.split("::")[0] == "acat" and len(k.split("::")) in (2, 3)
        doc = json.loads(v)
        assert doc["confidence"] == 1.0 and doc["category"]


def test_affordance_seed_is_category_entries():
    entries = _seed("affordance")
    assert len(entries) > 1000
    for k, v in entries:
        assert k.split("::")[0] == "vcat" and len(k.split("::")) in (2, 3)
        assert json.loads(v)["category"]


def test_context_seed_has_every_dimension_and_the_extractor_keys():
    entries = _seed("context")
    keys = [k for k, _ in entries]
    assert all(k.startswith("ctx::") for k in keys)
    for suffix in ("::geographic", "::temporal", "::domain", "::institutional"):
        assert any(k.endswith(suffix) for k in keys), suffix
    assert all(k.startswith("ctx::domain::") for k in keys if "::domain::" in k)
    assert all(k.startswith("ctx::inst::") for k in keys if "::inst::" in k)
    assert any(k.startswith("ctx::cx:") for k in keys)             # context_axis/seed.py
    assert all("canonical_value" in json.loads(v) for _, v in entries[:500])
