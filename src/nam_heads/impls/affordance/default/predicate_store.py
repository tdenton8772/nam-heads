"""The predicate lexicon in the affordance store: seeded, read and grown.

Predicate mode keeps its lexicon in the affordance category store (session
bucket ``vcat::`` keys at ingest, relayed to LMDB by affordance-lmdb-sync,
read by nam-query), next to the legacy verb categories:

  vcat::b:<verb>               verb base   {kind: seed|learned, count, bare}
  vcat::s:<word>               synonym     {kind: seed, target}
  vcat::ph:<verb>:<particle>   phrasal     {kind: seed|learned, value?, count}
  vcat::p:<value>              predicate   {kind, lemma, particle, count,
                                            by_ontology: {CURIE: count}}

Every doc also carries "category"/"confidence" so a legacy reader sees a
well-formed entry.

Seed: ``seed_entries()`` -- data/verb_bases.json (every WordNet verb lemma),
data/verb_synonyms.json (the shared sense-preserving fold), an identity
synonym doc per data/verb_synonyms_unfolded.json word (a fold an older table
had) and predicates.PHRASAL -- written by the store's seeding path
(storage/factory._seed_store): absent keys, and seed docs stamped with
another seed version.

Learning (ingest only; query reads, never writes):
- a verb base the seed lacks is LEARNED from both of its forms: seen bare
  (a predicate token tagged VB/VBP whose text is its own base) at least
  ``LEARN_BASE_MIN_BARE`` time(s) AND seen inflected (an unknown -ed/-ing/-s
  predicate token whose base candidates include it) at least once. Then
  inflected forms repair to it ("livestreamed" -> "livestream"). Bare
  sightings alone are not enough: the rule tagger's suffix rules tag nouns
  in -ate/-ise as VB ("vertebrate", "doctorate", "treatise"), and those
  never occur inflected.
- a phrasal value is LEARNED for a verb + true particle (up, out, down, off,
  away, back, over, apart, together, forth) seen ``LEARN_PHRASAL_MIN`` times
  and on at least ``LEARN_PHRASAL_RATIO`` of the verb's tokens. A learned
  phrasal value adds a value ("end_up"); it never merges two.
- per-value counts by the sentence's ontology class feed the bundler
  (``ont_freq``), used only once a value has ``ONT_FREQ_MIN`` observations.

Determinism: two runs over the same records in the same batches write the
same store. Bare-form and particle observations are counted whether or not a
learned entry was used (a learned base stops its own bare count at the
threshold), so any record order learns the same USABLE entries; counts of
values that a learned entry re-spells can differ with order
(tests/test_predicate_store.py).

One ``get_docs_batch`` per head batch: ``prefetch`` asks for every key the
batch can touch.

Commit window: by default each batch commits its own observations. With a
``PendingObservations`` (``every`` > 1, NAM_AFF_LEXICON_COMMIT_BATCHES) the
observations of ``every`` batches are summed and committed once: a hot key (a
common verb's base and value docs) is merged, encoded and written once per
window instead of once per batch. The totals are the same; an entry learned
inside a window takes part in extraction from the next window on. The window
is counted in batches, so the same records in the same batches still write
the same store.
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

from . import predicates as P

BASE, SYN, PHR, PRED = "b:", "s:", "ph:", "p:"

TRUE_PARTICLES = frozenset({"up", "out", "down", "off", "away", "back", "over",
                            "apart", "together", "forth"})

LEARN_BASE_MIN_BARE = int(os.getenv("NAM_AFF_LEARN_BASE_MIN", "1"))
LEARN_BASE_MIN_INFLECTED = int(os.getenv("NAM_AFF_LEARN_BASE_MIN_INFL", "1"))
LEARN_PHRASAL_MIN = int(os.getenv("NAM_AFF_LEARN_PHRASAL_MIN", "5"))
LEARN_PHRASAL_RATIO = float(os.getenv("NAM_AFF_LEARN_PHRASAL_RATIO", "0.25"))
ONT_FREQ_MIN = int(os.getenv("NAM_AFF_ONT_FREQ_MIN", "20"))


def seed_entries(version: Optional[str] = None) -> List[Tuple[str, dict]]:
    """Seed docs, each stamped with the seed version (storage/seed_version.py).
    A changed seed artifact updates seeded docs; learned docs are never
    replaced, and a seed doc's counters survive its update (merge.py)."""
    from .storage.seed_version import seed_version
    v_ = version or seed_version()
    meta = {"confidence": 1.0, "kind": "seed", "seed_version": v_, "_g": {}}
    out: List[Tuple[str, dict]] = []
    for v in sorted(P.verb_bases()):
        out.append((BASE + v, dict(meta, category="verb_base")))
    for a, b in sorted(P.seed_synonyms().items()):
        out.append((SYN + a, dict(meta, category="synonym", target=b)))
    # Identity docs for the words an older table folded: seeding writes the
    # seed's keys only, so without them a store seeded from that table kept
    # the old fold (ensure -> assure) after the table dropped it.
    for a in sorted(P.seed_unfolded()):
        out.append((SYN + a, dict(meta, category="synonym", target=a)))
    for (v, part), val in sorted(P.PHRASAL.items()):
        out.append((f"{PHR}{v}:{part}", dict(meta, category="phrasal", value=val)))
    return out


def _scoped_lemmas(ontology: Optional[str]) -> FrozenSet[str]:
    """Values that can have an ontology-scoped category row
    (vcat::<value>::<ontology>) under this ontology.

    Scoped rows are written only by seeding, from
    definitions.ONTOLOGY_CATEGORY_OVERRIDES (storage/factory._seed_store,
    supervisor/seeds.py; ``_auto_classify`` writes global rows), and that
    table has only ever gained entries -- older seed versions differ only by
    the pre-CURIE labels no query uses. So a scoped key outside it does not
    exist and is neither prefetched nor read: it was ~200 lookups per
    record, all misses."""
    from .definitions import ONTOLOGY_CATEGORY_OVERRIDES
    if not ontology:
        return frozenset()
    return frozenset(ONTOLOGY_CATEGORY_OVERRIDES.get(ontology, ()))


@lru_cache(maxsize=8192)
def _token_base(text: str, rule_lemma: str):
    """Per (surface, rule lemma): the base candidates (surface candidates,
    the rule lemma, their seed synonym targets), their b:/s:/p: keys and the
    seed phrasal values they can take. Memoized on two words only -- with
    the next words in the key the cache grew to ~50 MB per 2,000 records."""
    syn = P.seed_synonyms()
    cands = P.surface_candidates(text)
    if rule_lemma:
        cands.append(rule_lemma)
    cands += [syn[c] for c in list(cands) if c in syn]
    keys, values = set(), set()
    for c in cands:
        keys.update((BASE + c, SYN + c, PRED + c))
        values.add(c)
    cset = set(cands)
    for (c, _), val in P.PHRASAL.items():
        if c in cset:
            keys.add(PRED + val)
            values.add(val)
    return tuple(dict.fromkeys(cands)), frozenset(keys), frozenset(values)


def _token_keys(text: str, rule_lemma: str, nxt: Tuple[str, ...]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(store keys, producible values) of one VERB token for the prefetch:
    ``_token_base`` plus, per candidate and next word, the phrasal doc and
    value where a phrasal entry can exist (seed pair or true particle)."""
    cands, keys, values = _token_base(text, rule_lemma)
    extra_k, extra_v = [], []
    for w in nxt:
        true = w in TRUE_PARTICLES
        for c in cands:
            if true or (c, w) in P.PHRASAL:
                extra_k += (f"{PHR}{c}:{w}", f"{PRED}{c}_{w}")
                extra_v.append(f"{c}_{w}")
    if not extra_k:
        return keys, values
    return keys | frozenset(extra_k), values | frozenset(extra_v)


def supports_docs(store) -> bool:
    """True if the store implements the predicate-doc API (no I/O)."""
    from .storage.base import CategoryStore
    fn = getattr(type(store), "get_docs_batch", None)
    return fn is not None and fn is not CategoryStore.get_docs_batch


class PendingObservations:
    """Lexicon observations carried across head batches, committed once per
    ``every`` batches. One per head worker; thread-safe, as the worker runs
    two batches at a time (services/nats_worker.run_head executor)."""

    def __init__(self, every: int):
        self.every = max(1, int(every))
        self.delta: Dict[str, dict] = {}
        self.batches = 0
        self.lock = threading.Lock()          # guards delta and batches
        self.commit_lock = threading.Lock()   # one commit at a time

    def take(self, force: bool = False) -> Optional[Dict[str, dict]]:
        """The window's observations when it is due (or ``force``), else
        None. Counts one batch unless forced."""
        with self.lock:
            if not force:
                self.batches += 1
                if self.batches < self.every:
                    return None
            self.batches = 0
            if not self.delta:
                return None
            delta, self.delta = self.delta, {}
            return delta

    def clear(self) -> int:
        """Drop the window (cold session reset). Returns the keys dropped."""
        with self.lock:
            n = len(self.delta)
            self.delta = {}
            self.batches = 0
            return n


class StoreLexicon(P.Lexicon):
    """Lexicon over the affordance store, for one head batch."""

    def __init__(self, store, learn: bool = True, use_learned: bool = True,
                 pending: Optional[PendingObservations] = None, fold: bool = True):
        self.store = store
        self.fold = fold          # False under NAM_AFFORDANCE_SENSES: the sense table replaces the fold
        self.learn = learn
        self.use_learned = use_learned
        self.docs: Dict[str, Optional[dict]] = {}
        self._delta: Dict[str, dict] = {}
        self._pending = pending

    # ---- read ---------------------------------------------------------------
    def prefetch(self, docs: Iterable, sentence_labels: Optional[List[set]] = None) -> None:
        """ONE store read for the batch: every lexicon doc the batch can touch
        and the legacy category keys (vcat::<value>, vcat::<value>::<ontology>)
        of every value it can produce.

        A phrasal doc / phrasal value is asked for only where one can exist:
        a seed pair (predicates.PHRASAL) or a true particle (the only
        particles ``note_verb`` learns and ``phrasal`` accepts unseeded).
        Asking for every next word made ~1,000 keys per record, nearly all
        of which no rule can consult."""
        keys = {PRED + P.COPULAR_NOUN, PRED + P.COPULAR_ADJ}
        self._cat_names = set()
        docs = list(docs)
        sentence_labels = sentence_labels or [set()] * len(docs)
        for doc, labs in zip(docs, sentence_labels):
            if doc is None:
                continue
            values = {P.COPULAR_NOUN, P.COPULAR_ADJ}
            toks = list(doc)
            for k, tok in enumerate(toks):
                if tok.pos_ != "VERB":
                    continue
                nxt = tuple(toks[j].text.lower() for j in (k + 1, k + 2) if j < len(toks))
                tk, tv = _token_keys(tok.text, (getattr(tok, "lemma_", "") or "").lower(), nxt)
                keys |= tk
                values |= tv
            self._cat_names |= values
            for lab in labs:
                scoped = _scoped_lemmas(lab)
                if scoped:
                    self._cat_names.update(f"{v}::{lab}" for v in values & scoped)
        self.docs = self.store.get_docs_batch(list(keys | self._cat_names))

    def categories(self, queries) -> Dict[Tuple[str, Optional[str]], Optional[Tuple[str, float]]]:
        """(value, ontology) -> (category, confidence) from the prefetched
        category keys -- the legacy rule: ontology-scoped key, then global.
        A query the prefetch did not cover is read in one extra batch."""
        names = []
        for v, o in queries:
            if o and v in _scoped_lemmas(o):
                names.append(f"{v}::{o}")
            names.append(v)
        missing = [n for n in names if n not in self.docs]
        if missing:
            self.docs.update(self.store.get_docs_batch(missing))
        out = {}
        for v, o in queries:
            res = None
            for n in ([f"{v}::{o}"] if o else []) + [v]:
                d = self.docs.get(n)
                if d and "category" in d and d.get("category") not in (
                        "predicate", "verb_base", "synonym", "phrasal"):
                    res = (d["category"], float(d.get("confidence", 1.0)))
                    break
            out[(v, o)] = res
        return out

    def _doc(self, name: str) -> Optional[dict]:
        return self.docs.get(name)

    def is_base(self, word: str) -> bool:
        d = self._doc(BASE + word)
        if not d:
            return False
        if d.get("kind") == "seed":
            return True
        return (self.use_learned and d.get("bare", 0) >= LEARN_BASE_MIN_BARE
                and d.get("inflected", 0) >= LEARN_BASE_MIN_INFLECTED)

    def synonym(self, word: str) -> str:
        if not self.fold:
            return word
        d = self._doc(SYN + word)
        return d.get("target", word) if d else word

    def phrasal(self, base: str, particle: str) -> Optional[str]:
        d = self._doc(f"{PHR}{base}:{particle}")
        if not d:
            return None
        if d.get("kind") == "seed":
            return d.get("value")
        if not self.use_learned or particle not in TRUE_PARTICLES:
            return None
        n = d.get("count", 0)
        total = (self._doc(BASE + base) or {}).get("count", 0)
        if n >= LEARN_PHRASAL_MIN and total and n / total >= LEARN_PHRASAL_RATIO:
            return f"{base}_{particle}"
        return None

    def value_doc(self, value: str) -> Optional[dict]:
        return self._doc(PRED + value)

    def ont_freq(self, value: str) -> Optional[Dict[str, int]]:
        d = self._doc(PRED + value)
        if not d or d.get("count", 0) < ONT_FREQ_MIN:
            return None
        return dict(d.get("by_ontology") or {})

    # ---- observe (ingest) ---------------------------------------------------
    def _bump(self, name: str, field: str, n: int = 1, **fixed) -> None:
        if self._pending is not None:
            with self._pending.lock:
                self._bump_into(self._pending.delta, name, field, n, fixed)
        else:
            self._bump_into(self._delta, name, field, n, fixed)

    @staticmethod
    def _bump_into(delta: Dict[str, dict], name: str, field: str, n: int, fixed: dict) -> dict:
        d = delta.setdefault(name, {"_inc": {}, "_fixed": {}, "_ont": {}})
        d["_inc"][field] = d["_inc"].get(field, 0) + n
        for k, v in fixed.items():
            d["_fixed"].setdefault(k, v)
        return d

    def note_verb(self, surface: str, tag: str, base: str, known: bool, particle: str) -> None:
        if not self.learn or not base.isalpha():
            return
        if not known and tag in ("VB", "VBP") and surface == base:
            self._bump(BASE + base, "bare", category="verb_base", kind="learned")
        elif not known and surface != base and surface.endswith(("ed", "ing", "s")):
            for c in P._candidates(surface)[1:]:
                if len(c) >= 3 and c.isalpha() and not (self._doc(BASE + c) or {}).get("kind") == "seed":
                    self._bump(BASE + c, "inflected", category="verb_base", kind="learned")
        if known or (self._doc(BASE + base) or {}).get("bare", 0):
            self._bump(BASE + base, "count")
        if particle in TRUE_PARTICLES:
            self._bump(f"{PHR}{base}:{particle}", "count", category="phrasal", kind="learned")

    def record_value(self, pred: "P.Predicate", ontology: Optional[str]) -> None:
        if not self.learn:
            return
        name = PRED + pred.value
        fixed = {"category": "predicate", "kind": pred.kind,
                 "lemma": pred.lemma, "particle": pred.particle}
        if self._pending is not None:
            with self._pending.lock:
                d = self._bump_into(self._pending.delta, name, "count", 1, fixed)
                if ontology:
                    d["_ont"][ontology] = d["_ont"].get(ontology, 0) + 1
            return
        d = self._bump_into(self._delta, name, "count", 1, fixed)
        if ontology:
            d["_ont"][ontology] = d["_ont"].get(ontology, 0) + 1

    def commit(self, force: bool = False) -> int:
        """Write the observations (one put per touched key): the batch's own,
        or -- with a ``PendingObservations`` -- the window's, when it is due
        (``force`` commits it now).

        Counters are G-counters (storage/merge.py): this writer raises only
        its own witness entry, from its own last value, and the store merges
        per witness with max -- so concurrent pods and replays never lose an
        increment, and the totals are the sum over witnesses."""
        if not self.learn:
            self._delta = {}
            return 0
        if self._pending is not None:
            delta = self._pending.take(force)
            if not delta:
                return 0
            # A window spans batches: the docs this batch prefetched may be
            # older than another batch's commit, and the witness counter must
            # rise from its latest value. Read every touched key now (L1 holds
            # this process's own writes), one commit at a time.
            with self._pending.commit_lock:
                self.docs.update(self.store.get_docs_batch(sorted(delta)))
                return self._write(delta)
        if not self._delta:
            return 0
        delta, self._delta = self._delta, {}
        # A touched key the prefetch did not cover (a synonym target such as
        # depict -> portray) must be read before it is rewritten, or the
        # merge would replace a seed doc with a bare counter.
        missing = [k for k in sorted(delta) if k not in self.docs]
        if missing:
            self.docs.update(self.store.get_docs_batch(missing))
        return self._write(delta)

    def _write(self, delta: Dict[str, dict]) -> int:
        from .storage import merge as M
        me = M.witness_id()
        written = 0
        hint = getattr(self.store, "accepts_merge_hint", False)
        for name in sorted(delta):
            d = delta[name]
            cur = self.docs.get(name) or {}
            out = {k: v for k, v in cur.items() if k not in ("_g", "_go") + M.COUNTER_FIELDS + ("by_ontology",)}
            if not out:
                out = {"confidence": 1.0}
                out.update(d["_fixed"])
            g = {}
            for f, n in d["_inc"].items():
                g[f] = {me: int(((cur.get("_g") or {}).get(f) or {}).get(me, 0)) + n}
            out["_g"] = g
            if d["_ont"]:
                out["_go"] = {o: {me: int(((cur.get("_go") or {}).get(o) or {}).get(me, 0)) + n}
                              for o, n in d["_ont"].items()}
            out.setdefault("category", d["_fixed"].get("category", "predicate"))
            out.setdefault("confidence", 1.0)
            merged = M.merge_docs(cur, out)
            if hint:
                # The store merges with what it holds; when that is ``cur``
                # (the doc this batch read), the result is ``merged``.
                self.store.put_doc(name, out, merged_from=(self.docs.get(name), merged))
            else:
                self.store.put_doc(name, out)        # the store merges with what it holds
            self.docs[name] = merged
            written += 1
        return written


def store_fingerprint(store) -> str:
    """Canonical JSON of every predicate doc (memory store; tests)."""
    return json.dumps(store.doc_items(), sort_keys=True)
