"""The attribute head works copied anywhere and imported under another package name, as the plugin
factory loads it (docs/design-head-plugins.md section 5), and declares a static IMPL_META."""

import pytest

from nam_heads.impls._testing.relocation import (
    addresses, assert_located_inside, check_impl_meta, head_class, package_name, relocated_head,
)

SLOT = "attribute"
CONFIG = {"category_store_backend": "memory"}
SENTENCES = (
    "The red ball is heavy.",
    "Abraham Lincoln was a tall lawyer from Kentucky.",
    "The ancient city was famous for its beautiful gardens and wealthy merchants.",
)
QUERY = "Which famous lawyer was tall?"


@pytest.fixture(scope="module")
def nlp():
    from nam_heads.impls.nlp.rule import RuleNLPService
    return RuleNLPService()


@pytest.fixture
def relocated(tmp_path, monkeypatch):
    yield from relocated_head(SLOT, tmp_path, monkeypatch)


def _run(cls, nlp):
    head = cls(head_type=SLOT, config=dict(CONFIG))
    out = [addresses(head.encode(t, {"spacy_doc": nlp.parse(t)})) for t in SENTENCES]
    out.append(addresses(head.encode_query(QUERY, {"spacy_doc": nlp.parse(QUERY)})))
    return out


def test_relocated_head_encodes_like_the_in_tree_head(relocated, nlp, tmp_path):
    from nam_heads.impls.attribute.default.head import AttributeEncoder

    cls = head_class(relocated)
    assert cls is not AttributeEncoder and cls.__module__ == f"{package_name(SLOT)}.head"
    got = _run(cls, nlp)
    assert got == _run(AttributeEncoder, nlp)
    assert all(got[:len(SENTENCES)]), "every sentence asserts an attribute"
    # person_nouns.txt / abstract_nouns.txt load from the copy.
    assert_located_inside(package_name(SLOT), tmp_path / "pkgs" / package_name(SLOT), SLOT)


def test_impl_meta_reads_statically():
    meta = check_impl_meta(SLOT, "attribute-values-v1")
    assert meta["resources"] == {"cpu": 1.0, "memory_mb": 1024, "workers": 2}
