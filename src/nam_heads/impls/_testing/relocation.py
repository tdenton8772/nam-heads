"""Shared helper for the default heads' relocation tests (docs/design-head-plugins.md section 5).

The plugin factory unpacks a head's package into a cache and imports it under a package name of
its own (e.g. ``nhp_affordance_default_1a2b3c4d``). A default head must therefore work when its
directory is copied anywhere and imported under any name: its own modules imported relatively,
everything outside it by absolute host path, and every file it loads found next to its modules.

``relocated_head`` copies ``impls/<slot>/default/`` (without ``tests/`` and ``__pycache__``) to
``<tmp>/pkgs/relocated_<slot>/``, puts ``<tmp>/pkgs`` on ``sys.path`` (monkeypatch: undone after
the test) and imports ``relocated_<slot>.head``; the package is removed from ``sys.modules``
again on teardown. Nothing here changes process state at import.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from nam_heads.sdk.impl_meta import SLOT_CONTRACTS, impl_meta_problem, read_module_literal

from nam_heads.sdk import EncoderHead

IMPLS_DIR = Path(__file__).resolve().parent.parent


def head_dir(slot: str) -> Path:
    return IMPLS_DIR / slot / "default"

# Every field a default head's IMPL_META carries (docs/design-head-plugins.md section 3).
META_FIELDS = ("slot", "name", "version", "contract", "description", "output", "resources",
               "dependencies")


def package_name(slot: str) -> str:
    return f"relocated_{slot}"


def purge(package: str) -> None:
    """Drop ``package`` and its submodules from ``sys.modules``."""
    for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[name]


def relocated_head(slot: str, tmp_path: Path, monkeypatch) -> Iterator[Any]:
    """Generator for a fixture: yields the ``head`` module of the relocated copy of ``slot``."""
    package = package_name(slot)
    root = tmp_path / "pkgs"
    dest = root / package
    shutil.copytree(head_dir(slot), dest, ignore=shutil.ignore_patterns("tests", "__pycache__"))
    purge(package)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    try:
        yield importlib.import_module(f"{package}.head")
    finally:
        purge(package)


def head_class(module) -> type:
    """The single EncoderHead subclass in ``module``'s namespace (the registry's rule)."""
    found = [obj for _, obj in inspect.getmembers(module, inspect.isclass)
             if issubclass(obj, EncoderHead) and obj is not EncoderHead]
    assert len(found) == 1, f"expected one EncoderHead subclass in {module.__name__}, got {found}"
    return found[0]


def addresses(result) -> List[Tuple[str, str, float, Any, Any]]:
    """What a head emits for a record, comparable across two copies of the head."""
    return [(a.head_type, a.concept, round(float(a.confidence), 9), a.metadata, a.code)
            for a in result.addresses]


def assert_located_inside(package: str, dest: Path, slot: str) -> None:
    """No module of the relocated package names a path inside the original ``impls/<slot>/default``
    (a hard-coded source-tree path), and every module file is inside the copy."""
    original = str(head_dir(slot))
    for name, mod in list(sys.modules.items()):
        if not (name == package or name.startswith(package + ".")) or mod is None:
            continue
        f = getattr(mod, "__file__", None)
        if f:
            assert Path(f).resolve().is_relative_to(dest.resolve()), f"{name} loaded from {f}"
        for attr, val in vars(mod).items():
            if isinstance(val, (str, Path)) and original in str(val):
                raise AssertionError(f"{name}.{attr} points into the source tree: {val}")


def check_impl_meta(slot: str, output: str, version: str = "1.0.1") -> Dict[str, Any]:
    """IMPL_META of ``impls/<slot>/default/head.py``, read statically, with every field the plugin
    contract asks of a default head. Returns it for slot-specific checks."""
    head = head_dir(slot) / "head.py"
    meta = read_module_literal(head, ("IMPL_META",)).get("IMPL_META")
    assert isinstance(meta, dict), f"{head}: no literal IMPL_META"
    missing = [k for k in META_FIELDS if k not in meta]
    assert not missing, f"IMPL_META lacks {missing}"
    assert impl_meta_problem(meta, slot, "default") is None
    assert meta["slot"] == slot and meta["name"] == "default" and meta["version"] == version
    assert meta["contract"] == SLOT_CONTRACTS[slot] == f"{slot}-v1"
    assert isinstance(meta["description"], str) and meta["description"].strip()
    assert "\n" not in meta["description"]
    assert meta["output"] == output
    res = meta["resources"]
    assert set(res) == {"cpu", "memory_mb", "workers"}
    assert isinstance(res["cpu"], float) and res["cpu"] > 0
    assert isinstance(res["memory_mb"], int) and res["memory_mb"] > 0
    assert isinstance(res["workers"], int) and res["workers"] >= 1
    assert meta["dependencies"] == []
    return meta
