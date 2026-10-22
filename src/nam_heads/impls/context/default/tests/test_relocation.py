"""The context head works copied anywhere and imported under another package name, as the plugin
factory loads it (docs/design-head-plugins.md section 5), and declares a static IMPL_META.

Both extractions are checked: ``particular`` (the default, delegating to the host's
nam_heads.impls.context.default.context_axis) and ``legacy`` (the head's own NER-driven extraction, definitions and
normalizers -- the code the copy actually carries)."""

import pytest

from nam_heads.impls._testing.relocation import (
    addresses, assert_located_inside, check_impl_meta, head_class, package_name, relocated_head,
)

SLOT = "context"
SENTENCES = (
    "Lincoln was born in Kentucky in 1809. He worked for the Illinois Central Railroad.",
    "During the Civil War, the Union Army fought in Virginia in 1863.",
    "Harvard University researchers published a medical study in Boston in 2015.",
)
QUERY = "What happened in Virginia in 1863?"
CONFIGS = {
    "particular": {"context_store_backend": "memory"},
    "legacy": {"context_store_backend": "memory", "context_store_seed": True, "head": "legacy"},
}


@pytest.fixture(scope="module")
def nlp():
    from nam_heads.impls.nlp.rule import RuleNLPService
    return RuleNLPService()


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    monkeypatch.delenv("NAM_CONTEXT_HEAD", raising=False)
    yield from relocated_head(SLOT, tmp_path, monkeypatch)


def _run(cls, nlp, mode):
    head = cls(head_type=SLOT, config=dict(CONFIGS[mode]))
    assert head.head_mode == mode
    out = [addresses(head.encode(t, {"spacy_doc": nlp.parse(t)})) for t in SENTENCES]
    out.append(addresses(head.encode_query(QUERY, {"spacy_doc": nlp.parse(QUERY)})))
    return out


@pytest.mark.parametrize("mode", sorted(CONFIGS))
def test_relocated_head_encodes_like_the_in_tree_head(relocated, nlp, tmp_path, mode):
    from nam_heads.impls.context.default.head import ContextEncoder

    cls = head_class(relocated)
    assert cls is not ContextEncoder and cls.__module__ == f"{package_name(SLOT)}.head"
    got = _run(cls, nlp, mode)
    assert got == _run(ContextEncoder, nlp, mode)
    assert any(got[:len(SENTENCES)]), "the sentences carry context"
    assert_located_inside(package_name(SLOT), tmp_path / "pkgs" / package_name(SLOT), SLOT)


def test_impl_meta_reads_statically():
    meta = check_impl_meta(SLOT, "context-values-v1")
    assert meta["resources"] == {"cpu": 1.0, "memory_mb": 1024, "workers": 1}
