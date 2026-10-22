"""Context head work-saving changes give the same values, observations and
store writes.

* one extraction per doc unless that doc can reach a usable learned entry
  (``Lexicon.consults_learned``); it was a second, seed-only pass for EVERY
  doc of a batch that read any learned entry;
* sentence lookup by binary search, the seed gazetteer's longest key and the
  period index cached, windows shared between the store query and extraction.
"""
import os
import random
from collections import ChainMap
from types import SimpleNamespace

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.context.default.context_axis import doc as D
from nam_heads.impls.context.default.context_axis import institutions, places, temporal
from nam_heads.impls.context.default.context_axis.encoder import ParticularContextEncoder
from nam_heads.impls.context.default.context_axis.lexicon import Lexicon, doc_surfaces
from nam_heads.impls.context.default.context_axis.seed import seed_store
from nam_heads.impls.context.default.storage.memory_store import MemoryContextStore


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


def seeded():
    st = MemoryContextStore()
    seed_store(st)
    return st


def learned(store):
    return {k: v for k, v in store._store.items() if k[0].startswith("cx:L:")}


WARM = ["Smith was born in Waunakee , Wisconsin in 1950 .",
        "The band formed in Waunakee , Wisconsin and toured .",
        "He founded the Northern Rail Workers Union ( NRWU ) in 1920 .",
        "The Northern Rail Workers Union ( NRWU ) struck in 1926 .",
        "The team is based in Richmond , Virginia near the river .",
        "He taught school in Virginia and later lived in Richmond for years ."]
BATCH = ["Jones later lived in Waunakee for many years .",       # uses a learned place
         "Members of the NRWU marched in 1930 .",                 # uses a learned acronym
         "She retired in Richmond after the war .",               # uses a learned ambiguity
         "Abraham Lincoln was born in Kentucky in 1809 .",        # reaches none
         "The Roman general fought in 44 BC near Rome .",         # reaches none
         "The cathedral in Paris was completed in 1345 ."]        # reaches none


class TwoPass(ParticularContextEncoder):
    """The previous rule: a second, seed-only extraction for every doc of a
    batch that read any usable learned entry."""

    def _run(self, texts, contexts, mode):
        orig = Lexicon.consults_learned
        Lexicon.consults_learned = lambda self_, toks: self_.uses_learned
        try:
            return super()._run(texts, contexts, mode)
        finally:
            Lexicon.consults_learned = orig


def _encode(cls, store, nlp, texts):
    enc = cls(head_type="context", config={"store": store})
    out = enc.encode_batch_texts(texts, [{"spacy_doc": nlp.parse(t)} for t in texts])
    return [[(a.concept, a.metadata) for a in r.addresses] for r in out]


def test_one_pass_matches_two_pass_values_and_learning(nlp):
    a, b = seeded(), seeded()
    for st in (a, b):
        _encode(ParticularContextEncoder, st, nlp, WARM)
    assert learned(a) == learned(b) and learned(a)
    ra = _encode(ParticularContextEncoder, a, nlp, BATCH + WARM)
    rb = _encode(TwoPass, b, nlp, BATCH + WARM)
    assert ra == rb
    assert learned(a) == learned(b)


def test_learned_entries_are_used_in_the_batch(nlp):
    st = seeded()
    _encode(ParticularContextEncoder, st, nlp, WARM)
    r = _encode(ParticularContextEncoder, st, nlp, BATCH)
    concepts = [{c for c, _ in rec} for rec in r]
    assert "g:NA/US/US-WI/n-waunakee" in concepts[0]
    assert "i:northern_rail_workers_union" in concepts[1]
    assert any(c.startswith("g:NA/US/US-VA/gn") for c in concepts[2])


def test_consults_learned_per_doc(nlp):
    st = seeded()
    _encode(ParticularContextEncoder, st, nlp, WARM)
    from nam_heads.impls.context.default.context_axis.lexicon import candidate_queries
    docs = [list(nlp.parse(t)) for t in BATCH]
    q = set().union(*(candidate_queries(t) for t in docs))
    lex = Lexicon(st.get_batch(sorted(q)))
    assert lex.uses_learned
    flags = [lex.consults_learned(t) for t in docs]
    assert flags == [True, True, True, False, False, False]


def test_doc_surfaces_cover_every_window_and_capitalized_token(nlp):
    toks = list(nlp.parse("The New York Times reported from Baden - Württemberg and NRWU ."))
    s = doc_surfaces(toks)
    for i, t in enumerate(toks):
        if t.text[:1].isupper():
            assert t.text in s
            for _, k in places._windows(toks, i, 6):
                assert k in s


def test_sentence_lookup_matches_linear_scan():
    rng = random.Random(5)
    for _ in range(200):
        n, cuts = rng.randint(1, 60), sorted(rng.sample(range(1, 80), rng.randint(0, 8)))
        bounds = [0] + [c for c in cuts if c < n] + [n]
        sents = [(a, b) for a, b in zip(bounds, bounds[1:]) if a < b]
        if rng.random() < 0.2 and sents:                 # gaps / odd shapes
            sents = [(a + 1, b) for a, b in sents if a + 1 < b]
        if rng.random() < 0.1:
            rng.shuffle(sents)                           # unordered: linear fallback
        f = D.sentence_lookup(sents)
        for i in range(-1, n + 2):
            assert f(i) == D.sentence_of(sents, i)


def _old_gazetteer_match(toks, i, j, gaz):
    """The previous gazetteer loop (every length up to the longest name)."""
    glen = max((len(k) for k in gaz), default=1)
    out, k = [], i
    while k < j:
        hit = None
        for ln in range(min(glen, j - k), 0, -1):
            w = tuple(toks[k:k + ln])
            if w in gaz:
                hit = (k, k + ln, gaz[w])
                break
        if hit:
            out.append(hit)
            k = hit[1]
        else:
            k += 1
    return out


def _new_gazetteer_match(toks, i, j, gaz):
    first = institutions._first_word_lengths(gaz)
    out, k = [], i
    while k < j:
        hit = None
        for ln in first.get(toks[k], ()):
            if ln > j - k:
                continue
            w = tuple(toks[k:k + ln])
            if w in gaz:
                hit = (k, k + ln, gaz[w])
                break
        if hit:
            out.append(hit)
            k = hit[1]
        else:
            k += 1
    return out


def test_first_word_lengths_match_the_full_scan():
    rng = random.Random(9)
    words = ["A", "B", "C", "D", "E"]
    for _ in range(200):
        seed = {tuple(rng.choices(words, k=rng.randint(1, 4))): f"s{n}" for n in range(rng.randint(0, 8))}
        learned_ = {(rng.choice(words),): "l"} if rng.random() < 0.5 else {}
        gaz = ChainMap(learned_, seed) if learned_ else seed
        toks = rng.choices(words, k=rng.randint(1, 12))
        assert _new_gazetteer_match(toks, 0, len(toks), gaz) == _old_gazetteer_match(toks, 0, len(toks), gaz)
    seed = {("A",): 1}
    assert institutions._first_word_lengths(seed) == {"A": (1,)}
    seed[("A", "B", "C")] = 2                             # a changed dict is re-indexed
    assert institutions._first_word_lengths(seed) == {"A": (3, 1)}


def test_period_index_matches_a_fresh_scan():
    table = {("world", "war", "ii"): ("C20", "world_war_ii"), ("renaissance",): ("", "r")}
    plen, first = temporal._period_index(table)
    assert plen == 3 and first == {"world", "renaissance"}
    assert temporal._period_index({}) == (1, set())


def test_windows_memo_returns_what_windows_builds(nlp):
    toks = list(nlp.parse("Lake Michigan and the City of New York met Baden - Württemberg ."))
    for i in range(len(toks)):
        assert places.windows(toks, i) == places._windows(toks, i, places.MAX_WINDOW)
        assert places.windows(toks, i) is places.windows(toks, i)
    other = list(nlp.parse("Paris , France ."))
    assert places.windows(other, 0) == places._windows(other, 0, places.MAX_WINDOW)
