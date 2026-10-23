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


@pytest.mark.parametrize("impl", IMPLS, ids=IDS)
def test_declared_exports_load(impl):
    """Every export an implementation declares (IMPL_META "provides") imports and is of its kind."""
    if T.unavailable(impl):
        pytest.skip(f"unavailable here: {T.unavailable(impl)}")
    problems = T.check_provides(impl)
    if any("No module named 'src" in p for p in problems):
        pytest.skip(f"needs the NAM host; set NAM_SRC ({problems[0]})")
    assert problems == []


def test_the_defaults_provide_what_nam_reads():
    """The exports NAM's host stages read from the default implementations are declared."""
    want = {"nlp.rule": {"pos_dictionary", "countries", "cities"},
            "ontology.codebook": {"merge_diagnostics"},
            "entity.default": {"normalize_mention", "comparator", "open_store"},
            "attribute.default": {"entity_admits", "evidence_exclusions_enabled", "title_spans", "seed",
                                  "axis_encoder"},
            "affordance.default": {"verb_synonyms", "verb_categories", "verb_bases", "resolve_base",
                                   "phrasal_verbs", "link_profile", "senses_enabled", "seed", "axis_encoder"},
            "context.default": {"entity_profile", "admit_context", "select_contexts", "seed", "axis_encoder"}}
    for impl in IMPLS:
        key = f"{impl.slot}.{impl.name}"
        if key in want:
            meta, _ = read_impl_meta(impl.head)
            assert set(meta.get("provides") or {}) == want[key], key


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
