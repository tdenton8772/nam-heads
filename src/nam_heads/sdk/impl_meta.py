"""IMPL_META: what an implementation declares, read without importing it.

Every implementation's ``head.py`` carries a module-level literal
``IMPL_META`` dict:

    IMPL_META = {
        "slot": "ontology",            # the slot it fills (= its parent directory)
        "name": "gaius_ref",           # its name in that slot (= its directory)
        "version": "1.2.0",
        "contract": "ontology-v1",     # must equal the slot's contract
        "description": "...",
        "requires": ["numpy"],         # top-level packages beyond the host image
        "assets": [{"name", "env", "default", "files"}],   # model files it loads
        "ontology": {"name": "sdg", "artifact": "artifacts/x.json"},  # ontology slot only
    }

It is read with ``ast`` (``read_module_literal``), so listing implementations
never imports a head module and never loads a model. Values may be literals or
names bound to earlier module-level literals.

Installed implementations are found through the ``nam.heads`` entry-point
group: ``"<slot>.<name>" = "<package of head.py>"`` (``installed``).
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ENTRY_POINT_GROUP = "nam.heads"

SLOT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")
IMPL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

# The contract each standard slot accepts (added slots take FACET_CONTRACT).
SLOT_CONTRACTS: Dict[str, str] = {
    "ontology": "ontology-v1",
    "entity": "entity-v1",
    "attribute": "attribute-v1",
    "affordance": "affordance-v1",
    "context": "context-v1",
}
FACET_CONTRACT = "facet-v1"


def static_value(node: ast.expr, consts: Dict[str, Any], partial: bool = False) -> Any:
    """A literal, a name bound to an earlier module-level literal, or a
    dict/list/tuple of those; anything else raises ValueError. With
    ``partial`` (the top level of IMPL_META only) a dict entry whose value is
    not static is left out instead; nested values are never partial."""
    if isinstance(node, ast.Name):
        if node.id in consts:
            return consts[node.id]
        raise ValueError(node.id)
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                raise ValueError("dict key")
            try:
                out[k.value] = static_value(v, consts)
            except ValueError:
                if not partial:
                    raise
        return out
    if isinstance(node, (ast.List, ast.Tuple)):
        # Every element must be static; a list is never partially read.
        vals = [static_value(e, consts) for e in node.elts]
        return vals if isinstance(node, ast.List) else tuple(vals)
    return ast.literal_eval(node)


def literal_dict_keys(path: Path, name: str) -> List[str]:
    """The keys written in module-level ``name = {...}``, static or not."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                 for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return [k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)]
    return []


def read_module_literal(path: Path, names: Iterable[str]) -> Dict[str, Any]:
    """Module-level ``NAME = <static value>`` assignments, read without
    importing the module (see ``_static_value``)."""
    wanted = set(names)
    out: Dict[str, Any] = {}
    consts: Dict[str, Any] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        targets: List[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            try:
                v = static_value(value, consts, partial=t.id in wanted)
            except ValueError:
                continue  # not static: treated as absent
            consts[t.id] = v
            if t.id in wanted:
                out[t.id] = v
    return out


def impl_meta_problem(meta: Any, slot: str, name: str) -> Optional[str]:
    if not isinstance(meta, dict):
        return "no literal IMPL_META dict"
    for k in ("slot", "name", "version", "contract"):
        if not meta.get(k):
            return f"IMPL_META lacks {k!r}"
    if meta["slot"] != slot:
        return f"IMPL_META slot {meta['slot']!r} != directory {slot!r}"
    if meta["name"] != name:
        return f"IMPL_META name {meta['name']!r} != directory {name!r}"
    if not SLOT_NAME_RE.match(slot) or not IMPL_NAME_RE.match(name):
        return "slot and name must match [a-z][a-z0-9_]{1,31}"
    req = meta.get("requires", ())
    if not isinstance(req, (list, tuple)) or not all(isinstance(r, str) for r in req):
        return "IMPL_META requires must be a list of package names"
    assets = meta.get("assets", ())
    if not isinstance(assets, (list, tuple)) or not all(
            isinstance(a, dict) and isinstance(a.get("files", []), (list, tuple)) for a in assets):
        return "IMPL_META assets must be a list of {name, env, default, files}"
    ont = meta.get("ontology")
    if ont is not None:
        if slot != "ontology":
            return "IMPL_META ontology is for the ontology slot only"
        if (not isinstance(ont, dict) or not isinstance(ont.get("name"), str)
                or not SLOT_NAME_RE.match(ont["name"]) or not isinstance(ont.get("artifact"), str)
                or ont["artifact"].startswith("/") or ".." in Path(ont["artifact"]).parts):
            return "IMPL_META ontology must be {name: [a-z][a-z0-9_]{1,31}, artifact: a path in the package}"
    return None


def missing_assets(assets: Iterable[Dict[str, Any]]) -> Tuple[str, ...]:
    """IMPL_META "assets": [{"name", "env", "default", "files"}]. An asset is
    missing when its directory ($env, else default) lacks any listed file."""
    out = []
    for a in assets:
        d = os.environ.get(str(a.get("env") or ""), "") or str(a.get("default") or "")
        files = [str(f) for f in a.get("files") or ()]
        if not d or not all(os.path.isfile(os.path.join(d, f)) for f in files) or not os.path.isdir(d):
            out.append(f"{a.get('name') or 'asset'} ({d or 'no path'})")
    return tuple(out)


def missing_packages(requires: Iterable[str]) -> Tuple[str, ...]:
    out = []
    for pkg in requires:
        try:
            if importlib.util.find_spec(pkg) is None:
                out.append(pkg)
        except (ImportError, ValueError):
            out.append(pkg)
    return tuple(out)


@dataclass(frozen=True)
class InstalledImpl:
    """One ``nam.heads`` entry point: where its ``head.py`` is, unimported."""
    slot: str
    name: str
    package: str          # import path of the package holding head.py
    head: Path            # that head.py
    dist: str = ""        # distribution that registered it

    @property
    def module(self) -> str:
        return f"{self.package}.head"


def installed(errors: Optional[List[str]] = None) -> List[InstalledImpl]:
    """Every implementation registered under the ``nam.heads`` entry-point
    group, sorted by (slot, name). Finding ``head.py`` imports the package's
    parents (``__init__`` files) but never ``head.py``. An entry that is
    malformed or cannot be found is skipped, and why goes to ``errors``."""
    from importlib.metadata import entry_points

    errs = errors if errors is not None else []
    out: Dict[Tuple[str, str], InstalledImpl] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        where = f"entry point {ep.name!r} ({ep.value})"
        slot, _, name = ep.name.partition(".")
        if not (SLOT_NAME_RE.match(slot) and IMPL_NAME_RE.match(name)):
            errs.append(f"{where}: name must be '<slot>.<name>'")
            continue
        try:
            spec = importlib.util.find_spec(ep.value)
        except (ImportError, ValueError) as e:
            errs.append(f"{where}: {e}")
            continue
        locs = list(spec.submodule_search_locations or ()) if spec else []
        head = Path(locs[0]) / "head.py" if locs else None
        if head is None or not head.is_file():
            errs.append(f"{where}: not a package with a head.py")
            continue
        dist = getattr(getattr(ep, "dist", None), "name", "") or ""
        if (slot, name) in out:
            errs.append(f"{where}: {slot}.{name} is registered twice")
            continue
        out[(slot, name)] = InstalledImpl(slot, name, ep.value, head, dist)
    return [out[k] for k in sorted(out)]


def read_impl_meta(head: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """(IMPL_META, problems) for one head.py, checked against the directory
    it sits in. problems == [] means it is well formed."""
    slot, name = head.parent.parent.name, head.parent.name
    try:
        meta = read_module_literal(head, ("IMPL_META",)).get("IMPL_META")
    except (OSError, SyntaxError) as e:
        return None, [str(e)]
    problem = impl_meta_problem(meta, slot, name)
    if problem:
        return meta, [problem]
    probs = [f"IMPL_META {k} is not static (literals and module constants only)"
             for k in ("requires", "assets", "ontology")
             if k not in meta and k in literal_dict_keys(head, "IMPL_META")]
    want = SLOT_CONTRACTS.get(slot, FACET_CONTRACT)
    if meta["contract"] != want:
        probs.append(f"contract {meta['contract']!r} != slot {slot!r} contract {want!r}")
    ont = meta.get("ontology")
    if ont and not (head.parent / ont["artifact"]).is_file():
        probs.append(f"ontology artifact {ont['artifact']} is not in the package")
    return meta, probs
