"""Exports: what an implementation gives the host besides the head itself.

Some host stages read data that belongs to a head: the bundler folds verbs with the affordance head's
synonyms and admits attributes by the attribute head's rules, query normalizes entity names the way the
entity head does, the supervisor seeds each head's store. The host never imports a head's modules by name
for this. Each contract lists the exports a host may read (``CONTRACT_EXPORTS``); an implementation
declares which of them it provides, and where, in IMPL_META:

    IMPL_META = {..., "provides": {
        "verb_synonyms": "definitions:VERB_SYNONYMS",     # <module in the package>:<attribute>
        "seed": "exports:seed",
    }}

The declaration travels with the package into the plugin registry, so the host lists every installed
implementation's exports from the registry and reads one from the implementation the head-slot manifest
selects (NAM: ``src.encoders.head_exports``). Every export is optional: a host stage that reads one the
selected implementation does not provide runs without it (each export's entry below says what that
means) and the host reports the missing export for the slot.

Kinds: ``callable`` (a function or class, called with the arguments given here) or ``mapping`` (a dict,
read only).
"""

from __future__ import annotations

import ast
import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# name -> (kind, what it is and what the host does without it)
_SEED = ("callable", "seed() -> [(key, payload bytes), ...]: the head's seed entries for its store in the "
                     "session bucket; the host writes the keys that are absent. Without it nothing is seeded.")
_AXIS = ("callable", "axis_encoder() -> AxisEncoder: the encoder the head codes its values with, for host "
                     "stages that have no head instance (health sampling). Without it they skip the axis.")

CONTRACT_EXPORTS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "parser-v1": {
        "pos_dictionary": ("callable", "pos_dictionary() -> {word: POS}: the tagger's dictionary, for the "
                                       "sentence sketch's lexicon. Without it the sketch has no word classes."),
        "countries": ("callable", "countries() -> frozenset of lower-case country names. Without it: empty."),
        "cities": ("callable", "cities() -> frozenset of lower-case city names. Without it: empty."),
    },
    "ontology-v1": {
        "merge_diagnostics": ("callable", "merge_diagnostics([snapshot, ...]) -> snapshot: the head's "
                                          "diagnostics merged across workers. Without it they are not merged."),
    },
    "entity-v1": {
        "normalize_mention": ("callable", "normalize_mention(text) -> str: the form an entity id is hashed "
                                          "from. Without it query lower-cases and strips the name."),
        "comparator": ("callable", "comparator(entity_type) -> comparator for fuzzy surface matching. "
                                   "Without it query and address_heal match exact surface forms only."),
        "open_store": ("callable", "open_store(session_kv) -> the entity store over the session bucket "
                                   "(resolve_surface_form, resolve_fuzzy). Without it address_heal cannot run."),
        "seed": _SEED,
    },
    "attribute-v1": {
        "entity_admits": ("callable", "entity_admits(kinds, lemma_category, ontology) -> bool: may a bundle "
                                      "of this ontology take the value. Without it every value is admitted."),
        "evidence_exclusions_enabled": ("callable", "evidence_exclusions_enabled() -> bool. Without it: False."),
        "title_spans": ("callable", "title_spans(doc, title) -> [(start, end), ...]: token spans naming the "
                                    "record's title. Without it no span is the title."),
        "seed": _SEED,
        "axis_encoder": _AXIS,
    },
    "affordance-v1": {
        "verb_synonyms": ("mapping", "{verb: canonical verb}: the fold the head applies. Without it the "
                                     "host folds nothing."),
        "verb_categories": ("mapping", "{category: [verb, ...]}. Without it query registers no seed verbs."),
        "verb_bases": ("callable", "verb_bases() -> frozenset of known base verbs. Without it: empty."),
        "resolve_base": ("callable", "resolve_base(text, rule_lemma) -> (base, known). Without it the rule "
                                     "lemma is the base and no verb is known."),
        "phrasal_verbs": ("mapping", "{(verb, particle): phrasal verb}. Without it: none."),
        "link_profile": ("callable", "link_profile(ontology, conditioned=True) -> dict: how predicates link "
                                     "to entities of an ontology class. Without it the host links with {}."),
        "senses_enabled": ("callable", "senses_enabled() -> bool: values are verb senses, not folded verbs. "
                                       "Without it: False."),
        "seed": _SEED,
        "axis_encoder": _AXIS,
    },
    "context-v1": {
        "entity_profile": ("callable", "entity_profile(own_places, topic) -> profile (with .dimensions, the "
                                       "dimensions it takes) of a bundle's entity. Without the three profile "
                                       "exports the bundler keeps every context that shares a sentence."),
        "admit_context": ("callable", "admit_context(profile, meta, entity_tokens, own_places, value) -> bool: "
                                      "does a context value apply to the entity."),
        "select_contexts": ("callable", "select_contexts(profile, entity_tokens, contexts, meta, own_places) "
                                        "-> the contexts kept for the entity."),
        "seed": _SEED,
        "axis_encoder": _AXIS,
    },
    "facet-v1": {},
}

REF_RE = re.compile(r"^(?P<module>[a-z_][a-z0-9_]*(\.[a-z_][a-z0-9_]*)*):(?P<attr>[A-Za-z_][A-Za-z0-9_]*)$")


class ProvidesError(ValueError):
    """IMPL_META "provides" is malformed."""


@dataclass(frozen=True)
class ExportRef:
    """One declared export: ``<module>:<attr>`` inside the implementation's package."""
    name: str
    module: str
    attr: str

    @property
    def ref(self) -> str:
        return f"{self.module}:{self.attr}"

    def path(self, head: Path) -> Path:
        """The module's file in the package of ``head`` (a head.py); a package's ``__init__.py`` too."""
        base = head.parent.joinpath(*self.module.split("."))
        return base.with_suffix(".py") if base.with_suffix(".py").is_file() else base / "__init__.py"


def known_exports(contract: str) -> Dict[str, Tuple[str, str]]:
    return CONTRACT_EXPORTS.get(contract, {})


def parse_provides(raw: Any, contract: str) -> Dict[str, ExportRef]:
    """IMPL_META["provides"] -> {name: ExportRef}; {} when absent. Raises ProvidesError: not an object, a
    name the contract does not list, or a reference that is not ``<module>:<attr>``."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ProvidesError("IMPL_META provides must be an object {export: 'module:attribute'}")
    known = known_exports(contract)
    out: Dict[str, ExportRef] = {}
    for name, ref in raw.items():
        if name not in known:
            raise ProvidesError(f"IMPL_META provides {name!r}: not an export of {contract} "
                                f"(known: {sorted(known) or 'none'})")
        m = REF_RE.match(ref) if isinstance(ref, str) else None
        if m is None:
            raise ProvidesError(f"IMPL_META provides {name!r}: {ref!r} is not 'module:attribute'")
        out[name] = ExportRef(name, m.group("module"), m.group("attr"))
    return out


def _top_level_names(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def provides_problems(meta: Dict[str, Any], head: Path) -> List[str]:
    """What is wrong with IMPL_META "provides" of ``head`` (a head.py), read without importing it: malformed,
    not static, or naming a module the package lacks or an attribute that module does not define at its top
    level. [] when it is absent or well formed."""
    from .impl_meta import FACET_CONTRACT, SLOT_CONTRACTS, literal_dict_keys
    if "provides" not in meta:
        if "provides" in literal_dict_keys(head, "IMPL_META"):
            return ["IMPL_META provides is not static (literals and module constants only)"]
        return []
    contract = str(meta.get("contract") or SLOT_CONTRACTS.get(str(meta.get("slot")), FACET_CONTRACT))
    try:
        refs = parse_provides(meta["provides"], contract)
    except ProvidesError as e:
        return [str(e)]
    out = []
    for r in refs.values():
        p = r.path(head)
        if not p.is_file():
            out.append(f"IMPL_META provides {r.name!r}: module {r.module} is not in the package")
            continue
        try:
            names = _top_level_names(p)
        except (OSError, SyntaxError) as e:
            out.append(f"IMPL_META provides {r.name!r}: {e}")
            continue
        if r.attr not in names:
            out.append(f"IMPL_META provides {r.name!r}: {r.module}.py defines no {r.attr!r}")
    return out


def load_export(package: str, ref: ExportRef) -> Any:
    """Import ``<package>.<ref.module>`` and return ``ref.attr``."""
    return getattr(importlib.import_module(f"{package}.{ref.module}"), ref.attr)


def kind_problem(contract: str, name: str, value: Any) -> Optional[str]:
    """Whether a loaded export is of its declared kind (None when it is)."""
    kind = known_exports(contract).get(name, ("", ""))[0]
    if kind == "callable" and not callable(value):
        return f"export {name!r} must be callable, got {type(value).__name__}"
    if kind == "mapping" and not isinstance(value, dict):
        return f"export {name!r} must be a dict, got {type(value).__name__}"
    return None
