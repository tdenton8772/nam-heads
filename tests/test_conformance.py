"""Every installed implementation against the contract (nam_heads.sdk.testing)."""

import pytest

from nam_heads.ontology import registry as R
from nam_heads.sdk import host
from nam_heads.sdk import testing as T
from nam_heads.sdk.impl_meta import installed, read_impl_meta

IMPLS = installed()
IDS = [f"{i.slot}.{i.name}" for i in IMPLS]


def test_this_repo_registers_its_heads():
    assert {"ontology.codebook", "ontology.abstain", "ontology.gaius", "ontology.gaius_ref", "nlp.rule",
            "entity.default", "attribute.default", "affordance.default", "context.default"} <= set(IDS)


def _load_or_skip(impl):
    """Load the implementation's class; skip when it imports the NAM host (the default heads use NAM's src
    modules) and NAM is not on the path (set NAM_SRC)."""
    try:
        return T.load_class(impl)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "src":
            pytest.skip(f"needs the NAM host ({e.name}); set NAM_SRC")
        raise


@pytest.mark.parametrize("impl", IMPLS, ids=IDS)
def test_impl_meta_is_well_formed(impl):
    meta, problems = read_impl_meta(impl.head)
    assert problems == []
    assert (meta["slot"], meta["name"]) == (impl.slot, impl.name)


HEADS = [i for i in IMPLS if i.slot != "nlp"]
PARSERS = [i for i in IMPLS if i.slot == "nlp"]


@pytest.mark.parametrize("impl", HEADS, ids=[f"{i.slot}.{i.name}" for i in HEADS])
def test_one_encoder_head_subclass(impl):
    if T.unavailable(impl):
        pytest.skip(f"unavailable here: {T.unavailable(impl)}")
    _load_or_skip(impl)


@pytest.mark.parametrize("impl", PARSERS, ids=[f"{i.slot}.{i.name}" for i in PARSERS])
def test_one_parser_subclass(impl):
    """An NLP plugin (slot nlp, contract parser-v1) defines exactly one Parser subclass in head.py."""
    import importlib
    import inspect
    from nam_heads.sdk.parser import Parser
    try:
        mod = importlib.import_module(impl.module)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "src":
            pytest.skip(f"needs the NAM host ({e.name}); set NAM_SRC")
        raise
    found = [c for _, c in inspect.getmembers(mod, inspect.isclass)
             if issubclass(c, Parser) and c is not Parser and c.__module__ == mod.__name__]
    assert len(found) == 1, found


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
