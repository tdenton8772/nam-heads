"""The context store as the extractor's lexicon: one batched lookup per batch,
and the learning rule.

Every table the extractor consults -- places, demonyms, organization head
words and names, named periods, field triggers -- lives in the context store
(``heads/context/storage``: session bucket ``ctx::`` keys, L1 dict, optional
LMDB L2, LMDB relay via ``drain_pending_lmdb_writes``). ``seed.py`` seeds it
from the sealed artifact; learning grows it.

Store phrases (all in dimension ``context_axis``), one document per surface
so a window costs one key, not one per table:
    cx:s:<surface lower>  seed:    {"place": [[path, tier, pop], ...],
                                    "dem": [path, tier],
                                    "org": {"Exact Surface": slug},
                                    "period": {"Exact Text": [century path, slug]}}
    cx:t:<token lower>    seed:    {"field": field, "head": ["Exact", ...]}
    cx:L:<surface lower>  learned: {"place" | "amb" | "org:<EXACT>":
                                    {"votes": {value: [witness, ...]}}}

Learning rule (``LEARN_KINDS``)
-------------------------------
An OBSERVATION is (kind, surface, value) settled by evidence inside one
record, computed against the SEED ONLY (never against learned entries), so
the set of observations is a function of the corpus. It is stored as a vote:
the record's witness id (a hash of its text) added to value's witness set,
capped at MAX_WITNESSES. Merging is set union -- commutative, associative,
idempotent -- so any order, batching, pod split or restream of the same
corpus converges to the same store contents.

A learned entry is USED when its top value has >= MIN_WITNESSES distinct
witnesses and more than twice the witnesses of the runner-up. Ingest and
query read the same store with the same rule, so they agree for a given
store state. The trade-off: a record's emitted values depend on the store
state when it is encoded, so a record encoded before an entry reaches the
threshold lacks what an identical later record gets (re-encoding converges).
"""

import hashlib
import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .places import _window_key, MAX_WINDOW, windows  # noqa: F401

DIM = "context_axis"
LEARN_KINDS = ("place", "amb", "org")
MIN_WITNESSES = 2
MAX_WITNESSES = 8
_MAX_WINDOW = max(MAX_WINDOW, 6)


def witness_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class SeedTables:
    """The sealed seed, decoded once per process (``seed.seed_entries``).

    The seed is identical on every pod and immutable, so it is served from
    memory: only LEARNED entries (``cx:L:``) are read through the store
    (L1 -> LMDB L2 -> KV when NAM_CONTEXT_KV_ON_MISS=1). The same documents
    are written to the store by ``seed_store`` / supervisor/seeds.py, so the
    store still holds the whole lexicon."""

    def __init__(self, entries):
        self.places: Dict[str, list] = {}
        self.demonyms: Dict[str, list] = {}
        self.orgs: Dict[tuple, str] = {}
        self.periods: Dict[tuple, tuple] = {}
        self.org_heads: Set[str] = set()
        self.head_kind: Dict[str, str] = {}
        self.fields: Dict[str, str] = {}
        self.legacy_fields: Dict[str, str] = {}
        self.era_words: Dict[str, str] = {}
        self.era_paths: Dict[str, str] = {}       # canonical era slug -> its century path
        self.ner_fields: Dict[str, str] = {}
        for phrase, _dim, val, _c in entries:
            doc = json.loads(val)
            kind, _, key = phrase[3:].partition(":")
            if kind == "s":
                if "place" in doc:
                    self.places[key] = doc["place"]
                if "dem" in doc:
                    self.demonyms[key] = doc["dem"]
                for exact, slug in doc.get("org", {}).items():
                    self.orgs[tuple(exact.split())] = slug
                if "period" in doc:
                    self.periods[tuple(key.split())] = tuple(doc["period"])
            elif kind == "t":
                if key.startswith("ner:"):
                    self.ner_fields[key[4:].upper()] = doc["field"]
                    continue
                if "field" in doc:
                    self.fields[key] = doc["field"]
                if "lfield" in doc:
                    self.legacy_fields[key] = doc["lfield"]
                if "era" in doc:
                    self.era_words[key] = doc["era"]
                    if doc.get("era_path"):
                        self.era_paths[doc["era"]] = doc["era_path"]
                for h in doc.get("head", ()):
                    self.org_heads.add(h)
                    if doc.get("kind"):
                        self.head_kind[h] = doc["kind"]
        self.ambiguous = {k for k, c in self.places.items() if len({x[0] for x in c}) > 1}


_SEED: Optional[SeedTables] = None


def seed_tables() -> SeedTables:
    global _SEED
    if _SEED is None:
        from .seed import seed_entries
        _SEED = SeedTables(seed_entries())
    return _SEED


_ACRONYM = re.compile(r"^[A-Z][A-Z0-9&]{1,7}$")


def candidate_queries(toks, gpe_tokens: Iterable[int] = ()) -> Set[Tuple[str, str]]:
    """The learned entries (``cx:L:``) this doc can USE -- the only store reads.

    A learned entry changes extraction in exactly three cases, so only those
    windows are queried:
      * an ambiguous seed place name (document may not settle it);
      * an unknown 1-3 token capitalized window with place evidence around
        it (a locative word before, a GPE label, or ", " after) -- the only
        windows where a learned new place (cue tier) is accepted;
      * an acronym-shaped token the seed does not know (learned alias)."""
    from .places import LOCATIVES
    seed = seed_tables()
    gpe = set(gpe_tokens)
    q: Set[Tuple[str, str]] = set()
    n = len(toks)
    for i in range(n):
        t = toks[i].text
        if not t or not t[0].isupper():
            continue
        if _ACRONYM.match(t) and (t,) not in seed.orgs:
            q.add((f"cx:L:{t.lower()}", DIM))
        before = i > 0 and toks[i - 1].text.lower() in LOCATIVES
        for j, k in windows(toks, i, _MAX_WINDOW):
            if k in seed.ambiguous:
                q.add((f"cx:L:{k}", DIM))
            elif k not in seed.places and j - i <= 3 and (
                    before or i in gpe or (j < n and toks[j].text == ",")):
                q.add((f"cx:L:{k}", DIM))
    return q


def doc_surfaces(toks) -> Set[str]:
    """Every key extraction of this doc can look up in a learned table:
    each window key (``places.windows`` from every capitalized token, up to
    ``_MAX_WINDOW`` tokens -- ``Lexicon.places`` / ``learned_amb``) and each
    capitalized token text (``Lexicon.orgs``' learned acronyms)."""
    out: Set[str] = set()
    for i in range(len(toks)):
        t = toks[i].text
        if not t or not t[0].isupper():
            continue
        out.add(t)
        out.update(k for _, k in windows(toks, i, _MAX_WINDOW))
    return out


def resolve_votes(entry: Optional[dict], min_witnesses: int = MIN_WITNESSES) -> Optional[str]:
    if not entry:
        return None
    ranked = sorted(((len(set(w)), v) for v, w in entry.get("votes", {}).items()),
                    key=lambda x: (-x[0], x[1]))
    if not ranked or ranked[0][0] < min_witnesses:
        return None
    if len(ranked) > 1 and ranked[0][0] <= 2 * ranked[1][0]:
        return None
    return ranked[0][1]


def merge_votes(entry: Optional[dict], value: str, witnesses: Iterable[str]) -> dict:
    out = {"votes": {v: sorted(set(w)) for v, w in (entry or {}).get("votes", {}).items()}}
    cur = set(out["votes"].get(value, []))
    cur.update(witnesses)
    out["votes"][value] = sorted(cur)[:MAX_WITNESSES]
    return out


class Lexicon:
    """This batch's view: the seed tables plus the learned entries read from
    the store, as the mappings the extractors read."""

    def __init__(self, results: Dict[Tuple[str, str], Optional[str]], use_learned: bool = True,
                 min_witnesses: int = MIN_WITNESSES):
        from collections import ChainMap
        seed = seed_tables()
        self.demonyms = seed.demonyms
        self.periods = seed.periods
        self.org_heads = seed.org_heads
        self.head_kind = seed.head_kind
        self.fields = seed.fields
        self.legacy_fields = seed.legacy_fields
        self.era_words = seed.era_words
        self.era_paths = seed.era_paths
        self.ner_fields = seed.ner_fields
        self.learned_amb: Dict[str, str] = {}
        self.learned_docs: Dict[str, dict] = {}
        learned_place: Dict[str, list] = {}
        learned_org: Dict[tuple, str] = {}
        for (phrase, _dim), val in results.items():
            if val is None or not phrase.startswith("cx:L:"):
                continue
            key = phrase[5:]
            doc = json.loads(val)
            self.learned_docs[key] = doc
            if not use_learned:
                continue
            for lk, entry in doc.items():
                v = resolve_votes(entry, min_witnesses)
                if v is None:
                    continue
                if lk == "place" and key not in seed.places:
                    learned_place[key] = [[v, "city", 0]]
                elif lk == "amb":
                    self.learned_amb[key] = v
                elif lk.startswith("org:") and (lk[4:],) not in seed.orgs:
                    learned_org[(lk[4:],)] = v
        self.uses_learned = bool(learned_place or learned_org or self.learned_amb)
        # Every key a learned entry is reachable under: window keys (places,
        # ambiguous names) and exact acronym tokens (orgs).
        self._learned_surfaces = set(learned_place) | set(self.learned_amb) | \
            {k[0] for k in learned_org}
        # Learned entries never override the seed.
        self.places = ChainMap(learned_place, seed.places) if learned_place else seed.places
        self.orgs = ChainMap(learned_org, seed.orgs) if learned_org else seed.orgs
        self._seed = seed

    def consults_learned(self, toks) -> bool:
        """True if extraction of the doc with these tokens can read a learned
        entry of this lexicon. False means its extraction against this
        lexicon IS its extraction against ``seed_only()``."""
        return bool(self._learned_surfaces) and \
            not self._learned_surfaces.isdisjoint(doc_surfaces(toks))

    def seed_only(self) -> "Lexicon":
        """The same batch with every learned entry removed: what observations
        are computed against (so learning never feeds on itself)."""
        lx = Lexicon({}, use_learned=False)
        lx.learned_docs = self.learned_docs
        return lx


def _canon(doc: Optional[dict]) -> Optional[dict]:
    if doc is None:
        return None
    return {k: {"votes": {v: sorted(set(w)) for v, w in e.get("votes", {}).items()}}
            for k, e in doc.items()}


def learned_writes(observations: Sequence[Tuple[str, str, str, str]],
                   current: Dict[str, dict]) -> List[Tuple[str, str, str, float]]:
    """Fold (kind, surface, value, witness) observations into the current
    learned documents (``Lexicon.learned_docs``, keyed by lowercased surface).
    Returns (phrase, dimension, json, confidence) to put -- only documents
    whose content changed. Confidence: the top value's share of witnesses."""
    acc: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
    for kind, surface, value, w in observations:
        if kind == "org":
            key, sub = surface.lower(), f"org:{surface}"
        else:
            key, sub = surface, kind
        acc.setdefault(key, {}).setdefault(sub, {}).setdefault(value, set()).add(w)
    out = []
    for key, subs in sorted(acc.items()):
        before = _canon(current.get(key))
        doc = dict(before or {})
        for sub, votes in sorted(subs.items()):
            entry = doc.get(sub)
            for value, ws in sorted(votes.items()):
                entry = merge_votes(entry, value, ws)
            doc[sub] = entry
        if doc == before:
            continue
        shares = [max(len(w) for w in e["votes"].values()) / sum(len(w) for w in e["votes"].values())
                  for e in doc.values()]
        out.append((f"cx:L:{key}", DIM, json.dumps(doc, sort_keys=True, separators=(",", ":")),
                    round(min(shares), 3)))
        current[key] = doc
    return out
