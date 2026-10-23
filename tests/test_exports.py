"""IMPL_META "provides": what an implementation gives host stages besides the head (nam_heads.sdk.exports)."""
import textwrap

import pytest

from nam_heads.sdk import exports as X
from nam_heads.sdk.impl_meta import read_impl_meta


def test_absent_provides_is_nothing():
    assert X.parse_provides(None, "affordance-v1") == {}


def test_a_declaration_parses_to_module_and_attribute():
    refs = X.parse_provides({"verb_synonyms": "definitions:VERB_SYNONYMS",
                             "seed": "sub.pkg:seed"}, "affordance-v1")
    assert refs["verb_synonyms"] == X.ExportRef("verb_synonyms", "definitions", "VERB_SYNONYMS")
    assert refs["seed"].module == "sub.pkg" and refs["seed"].ref == "sub.pkg:seed"


@pytest.mark.parametrize("raw, contract, why", [
    (["seed"], "affordance-v1", "must be an object"),
    ({"verb_synonym": "definitions:X"}, "affordance-v1", "not an export of affordance-v1"),
    ({"verb_synonyms": "definitions:X"}, "entity-v1", "not an export of entity-v1"),
    ({"anything": "m:x"}, "facet-v1", "known: none"),
    ({"seed": "definitions"}, "attribute-v1", "is not 'module:attribute'"),
    ({"seed": "../up:seed"}, "attribute-v1", "is not 'module:attribute'"),
    ({"seed": 3}, "attribute-v1", "is not 'module:attribute'"),
])
def test_a_malformed_declaration_is_refused(raw, contract, why):
    with pytest.raises(X.ProvidesError, match=why):
        X.parse_provides(raw, contract)


def _impl(tmp_path, provides, files):
    """A context.demo implementation on disk: head.py with IMPL_META plus ``files``."""
    pkg = tmp_path / "context" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "head.py").write_text(textwrap.dedent(f"""
        IMPL_META = {{"slot": "context", "name": "demo", "version": "1.0.0", "contract": "context-v1",
                      "provides": {provides}}}
    """))
    for rel, text in files.items():
        p = pkg / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text))
    return pkg / "head.py"


def test_a_well_formed_declaration_has_no_problems(tmp_path):
    head = _impl(tmp_path, {"seed": "exports:seed", "entity_profile": "axis.profiles:entity_profile",
                            "admit_context": "axis:admit", "select_contexts": "exports:select"},
                 {"exports.py": "from .axis.profiles import select\ndef seed():\n    return []\n",
                  "axis/__init__.py": "def admit(*a):\n    return True\n",
                  "axis/profiles.py": "class entity_profile:\n    pass\ndef select(*a):\n    return []\n"})
    assert read_impl_meta(head)[1] == []


def test_a_missing_module_or_attribute_is_a_problem(tmp_path):
    head = _impl(tmp_path, {"seed": "nowhere:seed", "axis_encoder": "exports:axis_encoder"},
                 {"exports.py": "def seed():\n    return []\n"})
    problems = read_impl_meta(head)[1]
    assert "IMPL_META provides 'seed': module nowhere is not in the package" in problems
    assert "IMPL_META provides 'axis_encoder': exports.py defines no 'axis_encoder'" in problems


def test_a_provides_that_is_not_static_is_a_problem(tmp_path):
    pkg = tmp_path / "context" / "demo"
    pkg.mkdir(parents=True)
    (pkg / "head.py").write_text('def _p():\n    return {}\nIMPL_META = {"slot": "context", "name": "demo", '
                                 '"version": "1", "contract": "context-v1", "provides": _p()}\n')
    assert read_impl_meta(pkg / "head.py")[1] == [
        "IMPL_META provides is not static (literals and module constants only)"]


def test_kind_is_checked_when_loaded():
    assert X.kind_problem("affordance-v1", "seed", lambda: []) is None
    assert X.kind_problem("affordance-v1", "verb_synonyms", {}) is None
    assert "must be callable" in X.kind_problem("affordance-v1", "seed", [])
    assert "must be a dict" in X.kind_problem("affordance-v1", "verb_synonyms", ["a"])


def test_every_contract_names_a_kind_and_a_fallback_for_each_export():
    for contract, exports in X.CONTRACT_EXPORTS.items():
        for name, (kind, what) in exports.items():
            assert kind in ("callable", "mapping"), (contract, name)
            assert what, (contract, name)
