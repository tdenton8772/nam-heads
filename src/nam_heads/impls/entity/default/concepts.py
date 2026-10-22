# src/encoders/heads/entity/concepts.py

"""Concepts: the lower-case multi-word mentions the entity head indexes.

A lower-case phrase that is not a name is either a term the corpus talks
about ("sedimentary rock", "country music", "economic diversification") or
a description written for one sentence ("powerful neighbors", "resilient
resistance", "regional scope"). Word shape cannot tell the two apart, so the
head asks two sources, and a phrase that neither knows gets no entity:

  the seed     A sealed, versioned list built offline from a Wikipedia title
               dump (scripts/axes/build_concept_lexicon.py). A common-noun
               term with its own article or redirect is titled in sentence
               case ("Sedimentary_rock"); a name capitalises every word. The
               list is the sorted 64-bit hashes of the title keys
               (codebook/concept_lexicon_v1.npy, provenance in the manifest
               beside it). The head never changes it at run time.

  learning     A phrase the seed lacks is learned when it is shaped like a
               term (``learnable``): it passes the compound rules, it is
               adjectives or nouns ending in a noun, and its modifier is not
               a free one -- a word that combines with many nouns and almost
               never into a term ("pivotal", "intricate", "powerful"; a
               sealed list, codebook/concept_free_modifiers_v1.json). Such a
               phrase becomes an entity on first sight and grows the store
               like any entity; this is how a domain corpus teaches the head
               its own terms.

               ``NAM_ENTITY_CONCEPT_K`` (default 1) can ask for K distinct
               records first. The count is the session document
               ``cpt::<canonical>`` -- {"recs": [record fingerprints, at most
               K], "learned": bool} -- written through the entity store's
               pending writes like the surf:: keys, so every node's LMDB gets
               it; a phrase is learned on its K-th record and the earlier
               records keep no bundle for it. Two workers that
               read-modify-write one document at once can lose an
               observation, which only delays learning.

Measured on fact gold v2 dev and the offline question sets (2026-10-08,
scripts/axes/fact_gold_eval.py, scripts/axes/data/fact_gold_v2/; questions
with and without an answer in a 25k-record index): K=1 kept every answered
question the head answered without the gate (description 85, paraphrase 28,
title 96 of 100/32/100) and cut junk entities per passage 1.26 -> 0.98;
K=3 cut them to 0.68 but lost 3 description and 1 paraphrase answers -- the
phrases a description question finds its article by ("neoteric style",
"dicroglossid frogs") occur once. Questions without an answer were the same
under both. Recurrence also admits stock description on its own: K=3 over
3,000 records without the free list promoted "pivotal moment" and "rich
tapestry".

Entity ids do not depend on either source (they hash the canonical form);
only whether a record indexes the entity does.

The gate (``needs_concept``) covers only lower-case mentions of two or more
words with no proper noun in them. Names, the record title, single words and
anything with a PROPN token are decided elsewhere, as before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

ARTIFACT = Path(__file__).parent / "codebook" / "concept_lexicon_v1.npy"
MANIFEST = ARTIFACT.with_suffix(".manifest.json")
FREE_MODIFIERS = Path(__file__).parent / "codebook" / "concept_free_modifiers_v1.json"

KEY_PREFIX = "cpt::"

_ARTICLES = frozenset({"a", "an", "the"})
_STRIP = re.compile(r"[^a-z0-9'\-]")
_NOT_PLURAL_ENDINGS = ("ss", "us", "is")


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

def _fold(word: str) -> str:
    w = word.replace("’", "'").replace("‘", "'")
    w = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode("ascii")
    return _STRIP.sub("", w.lower()).strip("'-")


def concept_key_words(words: Sequence[str]) -> str:
    """The lookup key of a phrase given as words: ASCII-folded, lower case,
    characters other than a-z 0-9 ' - removed, leading articles dropped,
    joined by single spaces. The seed builder and the head both use it."""
    out = [w for w in (_fold(x) for x in words) if w]
    while out and out[0] in _ARTICLES:
        out = out[1:]
    return " ".join(out)


def concept_key(text: str) -> str:
    """``concept_key_words`` of a mention as written. A possessive the parser
    split off ("children 's literature") is joined back to its word."""
    t = re.sub(r"\s+(['’]s)\b", r"\1", text or "")
    return concept_key_words(t.split())


def concept_hash(key: str) -> int:
    """64-bit blake2b of a key, as stored in the seed array."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "little")


def _singular_forms(word: str) -> List[str]:
    """Candidate singulars of a plural last word, most likely first."""
    if len(word) < 4 or not word.endswith("s") or word.endswith(_NOT_PLURAL_ENDINGS):
        return []
    out = []
    if word.endswith("ies"):
        out.append(word[:-3] + "y")
    if word.endswith("ves"):
        out += [word[:-3] + "f", word[:-3] + "fe"]
    if word.endswith("es"):
        out.append(word[:-2])
    out.append(word[:-1])
    return out


def key_variants(key: str) -> List[str]:
    """The key as written, then with its last word singular. Wikipedia
    titles a term in the singular ("Coffee bean"); a mention is often
    plural ("coffee beans"). Plural titles and redirects match as written."""
    if not key:
        return []
    words = key.split(" ")
    out = [key]
    for s in _singular_forms(words[-1]):
        v = " ".join(words[:-1] + [s])
        if v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# The seed
# --------------------------------------------------------------------------

_seed = None
_seed_lock = threading.Lock()


def _load_seed():
    global _seed
    if _seed is None:
        with _seed_lock:
            if _seed is None:
                import numpy as np
                arr = np.load(ARTIFACT, mmap_mode="r")
                if arr.dtype != np.uint64 or arr.ndim != 1:
                    raise ValueError(f"{ARTIFACT.name}: expected a 1-d uint64 array, got {arr.dtype} {arr.shape}")
                _seed = arr
    return _seed


def seed_meta() -> dict:
    """Provenance of the seed (dump, its sha256, the title rule, count)."""
    return json.loads(MANIFEST.read_text())


def in_seed(text: str) -> bool:
    """The mention (as written) is a seed concept, in any of its variants."""
    return seed_has_key(concept_key(text))


def seed_has_key(key: str) -> bool:
    if not key or " " not in key:
        return False
    import numpy as np
    seed = _load_seed()
    for v in key_variants(key):
        h = np.uint64(concept_hash(v))
        i = int(np.searchsorted(seed, h))
        if i < seed.shape[0] and seed[i] == h:
            return True
    return False


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def gate_enabled() -> bool:
    return os.environ.get("NAM_ENTITY_CONCEPT_GATE", "1").lower() in ("1", "true", "yes")


def needs_concept(cand, sent_starts: Optional[Set[int]] = None) -> bool:
    """The candidate is a lower-case phrase of two or more words with no
    proper noun: it is an entity only as a concept (seed or learned).

    ``sent_starts``: token indices that open a sentence in the parse. A
    capitalised first word is lower case for this rule only when it opens a
    sentence ("Student leaders ..."); elsewhere it marks a name the parser did
    not label ("Voyager 2", "Vasa bryggeri", "Normandy coast"). When the
    tagger read that sentence-initial word as a proper noun, only the seed
    can tell a term ("Arsenic contamination") from a name-led phrase
    ("Quetzaltenango vendors"): the gate covers it when the seed holds it,
    and otherwise leaves it to the rules that decided it before."""
    if cand.is_name or cand.is_propn:
        return False
    words = (cand.text or "").split()
    if len(words) < 2:
        return False
    tags = (cand.pos_pattern or "").split("_")
    if any(w[:1].isupper() for w in words[1:]) or "PROPN" in tags[1:]:
        return False
    if words[0][:1].isupper():
        if not (cand.start >= 0 and cand.start in (sent_starts or ())):
            return False
        if tags[0] == "PROPN":
            return in_seed(cand.text)
    return tags[0] != "PROPN"


# --------------------------------------------------------------------------
# What may be learned
# --------------------------------------------------------------------------

_free: Optional[frozenset] = None


def free_modifiers() -> frozenset:
    """The sealed free-modifier list (scripts/build_entity_free_modifiers.py)."""
    global _free
    if _free is None:
        data = json.loads(FREE_MODIFIERS.read_text())
        if data.get("version") != 1:
            raise ValueError(f"{FREE_MODIFIERS.name}: version {data.get('version')!r}, expected 1")
        _free = frozenset(data["modifiers"])
    return _free


_TERM_TAGS = frozenset({"ADJ", "NOUN"})


def learnable_shape(cand, entity_type: Optional[str]) -> bool:
    """Shaped like a term: the compound rules pass (no verb, determiner,
    generic modifier or root, at most two words), the words are adjectives
    or nouns ending in a noun, and no noun before the last is plural. A
    plural noun modifier is the tagger reading a verb or adverb as a noun
    ("audiences worldwide", "nations shape", "origins trace"); the terms
    that have one ("sales tax") are in the seed. An article does not count
    against the phrase: "the uniliteral sign" is an occurrence of the term
    (that mention itself stays a description, as for a seed term)."""
    from .codebook.compound_filter import is_meaningful_compound
    if not is_meaningful_compound(cand.normalized, cand.pos_pattern, cand.text,
                                  cand.token_count, cand.is_propn, entity_type=entity_type,
                                  is_name=cand.is_name, has_article=False):
        return False
    tags = (cand.pos_pattern or "").split("_")
    if not tags or tags[-1] != "NOUN" or not all(t in _TERM_TAGS for t in tags):
        return False
    words = concept_key(cand.text).split(" ")
    if len(words) != len(tags):
        return True
    return not any(t == "NOUN" and _singular_forms(w) for w, t in zip(words[:-1], tags[:-1]))


def learnable(cand, entity_type: Optional[str]) -> bool:
    """A gated phrase outside the seed that recurrence may promote."""
    if not learnable_shape(cand, entity_type):
        return False
    key = concept_key(cand.text)
    return bool(key) and key.split(" ")[0] not in free_modifiers()


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------

def record_fingerprint(text: str) -> str:
    """A record's identity for counting distinct records: 8 bytes of
    blake2b over its text, hex. The same record reloaded counts once."""
    return hashlib.blake2b((text or "").encode("utf-8"), digest_size=8).hexdigest()


def learn_k() -> int:
    return max(1, int(os.environ.get("NAM_ENTITY_CONCEPT_K", "1")))


def learning_enabled() -> bool:
    return os.environ.get("NAM_ENTITY_CONCEPT_LEARN", "1").lower() in ("1", "true", "yes")


class ConceptLearner:
    """Counts the distinct records a non-seed phrase occurs in, and promotes
    it at ``k``. One per head; thread-safe.

    Reads go through the store's ``get_raw_batch`` (LMDB, then KV where the
    store reads through); writes through ``put_raw_batch`` (the pending
    writes the worker publishes for lmdb-sync and the session bucket).
    Because those writes reach LMDB asynchronously, the learner keeps what
    it last wrote for each key and prefers it to a read. Both in-process
    maps are bounded by ``cache_max`` and cleared past it; they are caches,
    a miss re-reads the store.
    """

    def __init__(self, k: Optional[int] = None, cache_max: Optional[int] = None):
        self.k = k if k is not None else learn_k()
        self.cache_max = cache_max if cache_max is not None else int(
            os.environ.get("NAM_ENTITY_CONCEPT_CACHE_MAX", "200000"))
        self._learned: Set[str] = set()
        self._docs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.stats = {"observed": 0, "learned": 0, "promoted": 0}

    def reset(self) -> Dict[str, int]:
        """Forget what this process read and wrote: the session bucket was
        flushed (cold reset, src/storage/session_reset.py)."""
        with self._lock:
            out = {"learned": len(self._learned), "docs": len(self._docs)}
            self._learned.clear()
            self._docs.clear()
        return out

    def _trim(self) -> None:
        if self.cache_max <= 0:
            return
        if len(self._docs) > self.cache_max:
            self._docs.clear()
        if len(self._learned) > self.cache_max:
            self._learned.clear()

    def observe(self, store, items: Iterable[Tuple[str, str]], learn: bool = True) -> Set[str]:
        """``items``: (canonical, record fingerprint) pairs from one batch.
        Returns the canonicals that are concepts after this batch. With
        ``learn`` false (a query), nothing is counted or written."""
        items = [(c, fp) for c, fp in items if c]
        if not items:
            return set()
        if self.k <= 1:
            # First sight is enough: nothing to count, nothing to store.
            return {c for c, _ in items}
        with self._lock:
            known = {c for c, _ in items if c in self._learned}
            todo = sorted({c for c, _ in items if c not in known})
            have = {c: self._docs[c] for c in todo if c in self._docs}
        missing = [c for c in todo if c not in have]
        if missing:
            raw = _raw_get(store, [KEY_PREFIX + c for c in missing])
            for c in missing:
                v = raw.get(KEY_PREFIX + c)
                if v is None:
                    continue
                try:
                    have[c] = json.loads(v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v)
                except (ValueError, UnicodeDecodeError):
                    logger.warning("concepts: unreadable %s%s", KEY_PREFIX, c)
        with self._lock:
            for c, d in have.items():
                if d.get("learned"):
                    known.add(c)
                    self._learned.add(c)
            if not learn:
                self._trim()
                return known
            writes = []
            for c, fp in items:
                if c in known:
                    continue
                d = have.get(c) or {"recs": [], "learned": False}
                recs = list(d.get("recs") or [])
                if fp in recs:
                    continue
                recs.append(fp)
                self.stats["observed"] += 1
                d = {"recs": recs[: self.k], "learned": len(recs) >= self.k}
                have[c] = d
                self._docs[c] = d
                writes.append((KEY_PREFIX + c, json.dumps(d, separators=(",", ":")).encode("utf-8")))
                if d["learned"]:
                    known.add(c)
                    self._learned.add(c)
                    self.stats["promoted"] += 1
            self.stats["learned"] = len(self._learned)
            self._trim()
        if writes:
            _raw_put(store, writes)
        return known


def _raw_get(store, keys: List[str]) -> Dict[str, bytes]:
    fn = getattr(store, "get_raw_batch", None)
    return fn(keys) if fn is not None else {}


def _raw_put(store, items: List[Tuple[str, bytes]]) -> None:
    fn = getattr(store, "put_raw_batch", None)
    if fn is not None:
        fn(items)
