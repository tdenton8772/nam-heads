"""Every installed implementation against the contract (nam_heads.sdk.testing)."""

import pytest

from nam_heads.ontology import registry as R
from nam_heads.sdk import host
from nam_heads.sdk import testing as T
from nam_heads.sdk.impl_meta import installed, read_impl_meta

IMPLS = installed()
IDS = [f"{i.slot}.{i.name}" for i in IMPLS]


def test_this_repo_registers_its_heads():
    assert {"ontology.codebook", "ontology.abstain", "ontology.gaius",
            "ontology.gaius_ref"} <= set(IDS)


@pytest.mark.parametrize("impl", IMPLS, ids=IDS)
def test_impl_meta_is_well_formed(impl):
    meta, problems = read_impl_meta(impl.head)
    assert problems == []
    assert (meta["slot"], meta["name"]) == (impl.slot, impl.name)


@pytest.mark.parametrize("impl", IMPLS, ids=IDS)
def test_one_encoder_head_subclass(impl):
    if T.unavailable(impl):
        pytest.skip(f"unavailable here: {T.unavailable(impl)}")
    T.load_class(impl)


@pytest.mark.parametrize("impl", [i for i in IMPLS if i.slot == "ontology"],
                         ids=[x for x in IDS if x.startswith("ontology.")])
def test_ontology_contract(impl):
    if T.unavailable(impl):
        pytest.skip(f"unavailable here: {T.unavailable(impl)}")
    # Without a host parser the codebook cannot check its fingerprint.
    config = None
    if impl.name == "codebook" and not host.has_parser():
        config = {"check_parser": False}
    try:
        head = T.build(impl, config)
        assert T.check_ontology_head(head) == []
    finally:
        R.use_extension(None)
