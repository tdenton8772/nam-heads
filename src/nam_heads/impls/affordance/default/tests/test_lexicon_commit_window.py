"""The lexicon commit window (predicate_store.PendingObservations): N batches'
observations committed once, same totals, fewer writes."""
import os
import threading

import pytest

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

from nam_heads.impls.affordance.default import predicate_store as PS
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from nam_heads.impls.affordance.default.storage.memory_store import MemoryCategoryStore
from nam_heads.impls.affordance.default.storage.factory import _seed_store


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


class CountingStore(MemoryCategoryStore):
    """Counts the docs written (one put_doc per touched key per commit)."""

    def __init__(self):
        super().__init__()
        self.puts = 0

    def put_doc(self, name, doc, merged_from=None):
        self.puts += 1
        return super().put_doc(name, doc, merged_from=merged_from) if merged_from is not None \
            else super().put_doc(name, doc)


def seeded(cls=MemoryCategoryStore):
    s = cls()
    _seed_store(s)
    return s


def head(store, **cfg):
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "emit_mode": "predicate", **cfg})
    h.category_store = store
    return h


def run(h, nlp, texts, batch=1):
    out = []
    for k in range(0, len(texts), batch):
        chunk = texts[k:k + batch]
        out += h.encode_batch_texts(chunk, [{"spacy_doc": nlp.parse(t)} for t in chunk])
    return [[a.concept for a in r.addresses] for r in out]


# No entry crosses a learning threshold in these: the window cannot change
# what is extracted, so the store must come out byte-identical.
PLAIN = [
    "Smith founded the company and joined the board.",
    "The firm was depicted in a film.",
    "Jones wrote the book and founded a school.",
    "The museum displays paintings and founded a library.",
    "Smith joined the army in 1940.",
    "The company built ships and wrote software.",
]


def test_default_commits_every_batch():
    h = head(seeded())
    assert h._lex_pending is None
    assert h._lexicon("ingest")._pending is None


def test_env_and_config_set_the_window(monkeypatch):
    monkeypatch.setenv("NAM_AFF_LEXICON_COMMIT_BATCHES", "3")
    assert head(seeded())._lex_pending.every == 3
    assert head(seeded(), lexicon_commit_batches=5)._lex_pending.every == 5
    monkeypatch.setenv("NAM_AFF_LEXICON_COMMIT_BATCHES", "1")
    assert head(seeded())._lex_pending is None


def test_query_mode_never_joins_the_window():
    h = head(seeded(), lexicon_commit_batches=4)
    assert h._lexicon("query")._pending is None
    assert h._lexicon("ingest")._pending is h._lex_pending


def test_take_is_due_on_the_nth_batch_and_force_takes_a_partial_window():
    p = PS.PendingObservations(3)
    p.delta["b:x"] = {"_inc": {"count": 1}, "_fixed": {}, "_ont": {}}
    assert p.take() is None and p.take() is None
    assert set(p.take()) == {"b:x"}                 # third batch: due
    assert p.take() is None                         # empty window
    p.delta["b:y"] = {"_inc": {"count": 1}, "_fixed": {}, "_ont": {}}
    assert set(p.take(force=True)) == {"b:y"}
    p.delta["b:z"] = {"_inc": {"count": 1}, "_fixed": {}, "_ont": {}}
    assert p.clear() == 1 and p.take(force=True) is None


def test_window_writes_the_same_store_as_per_batch_commits(nlp):
    a, b = seeded(), seeded()
    va = run(head(a), nlp, PLAIN * 3)
    hb = head(b, lexicon_commit_batches=4)
    vb = run(hb, nlp, PLAIN * 3)
    hb.flush_lexicon()
    assert va == vb
    assert PS.store_fingerprint(a) == PS.store_fingerprint(b)


def test_window_writes_each_hot_doc_once_per_window(nlp):
    # The distinct docs one pass over PLAIN touches.
    u = seeded(CountingStore)
    u.puts = 0
    hu = head(u, lexicon_commit_batches=1000)
    run(hu, nlp, PLAIN)
    hu.flush_lexicon()
    distinct = u.puts
    # Per batch: every batch writes its own keys. A window of 12 batches
    # (two passes) writes each distinct doc once.
    a, b = seeded(CountingStore), seeded(CountingStore)
    a.puts = b.puts = 0
    run(head(a), nlp, PLAIN * 4)
    hb = head(b, lexicon_commit_batches=12)
    run(hb, nlp, PLAIN * 4)
    assert b.puts == 2 * distinct, (a.puts, b.puts, distinct)
    assert a.puts > 3 * b.puts


def test_nothing_reaches_the_store_until_the_window_is_due(nlp):
    s = seeded()
    snap = PS.store_fingerprint(s)
    h = head(s, lexicon_commit_batches=4)
    run(h, nlp, PLAIN[:3])                          # 3 batches of 1: window open
    assert PS.store_fingerprint(s) == snap
    run(h, nlp, PLAIN[3:4])                         # 4th batch: committed
    assert PS.store_fingerprint(s) != snap


def test_flush_commits_the_open_window(nlp):
    s = seeded()
    h = head(s, lexicon_commit_batches=100)
    run(h, nlp, PLAIN)
    snap = PS.store_fingerprint(s)
    assert h.flush_lexicon() > 0
    assert PS.store_fingerprint(s) != snap
    assert h.flush_lexicon() == 0                   # nothing left


def test_cold_reset_drops_the_open_window(nlp):
    s = seeded()
    h = head(s, lexicon_commit_batches=100)
    run(h, nlp, PLAIN)
    snap = PS.store_fingerprint(s)
    assert h.reset_session_state()["lexicon_pending"] > 0
    assert h.flush_lexicon() == 0
    assert PS.store_fingerprint(s) == snap


def test_entry_learned_in_a_window_is_used_from_the_next(nlp):
    laid = ["The ship was laid down in 1910.", "Her keel was laid down at Belfast.",
            "The cruiser was laid down in March.", "The battleship was laid down by Vickers.",
            "The destroyer was laid down in 1940."]
    assert len(laid) == PS.LEARN_PHRASAL_MIN
    s = seeded()
    h = head(s, lexicon_commit_batches=len(laid) + 1)
    run(h, nlp, laid)
    probe = ["The frigate was laid down in 1801."]
    assert "lay_down" not in run(h, nlp, probe)[0]  # still in the open window
    h.flush_lexicon()
    assert "lay_down" in run(h, nlp, probe)[0]


def test_witness_counter_rises_across_windows():
    """Each window raises this writer's counter from its latest value -- a
    window must not start from a doc another commit has since moved past."""
    s = seeded()
    p = PS.PendingObservations(1 << 30)
    me = __import__("nam_heads.impls.affordance.default.storage.merge", fromlist=["x"]).witness_id()
    stale = PS.StoreLexicon(s, pending=p)
    stale.docs = dict(s.doc_items())                # prefetched before the commits below
    for _ in range(3):
        lex = PS.StoreLexicon(s, pending=p)
        lex.note_verb("founded", "VBD", "found", True, "")
        lex.commit(force=True)
    stale.note_verb("founded", "VBD", "found", True, "")
    stale.commit(force=True)
    assert s.doc_items()["b:found"]["_g"]["count"][me] == 4


def test_concurrent_batches_lose_no_observation():
    s = seeded()
    p = PS.PendingObservations(1 << 30)

    def bump():
        lex = PS.StoreLexicon(s, pending=p)
        for _ in range(500):
            lex.note_verb("founded", "VBD", "found", True, "")

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert p.delta["b:found"]["_inc"]["count"] == 2000


# ---- the session-bucket store (the ingest path), two pods ---------------------

POD_TEXTS = [
    "Smith founded the company and joined the board.",
    "The ship was laid down in 1910 and launched in 1911.",
    "Jones is an actor who starred in the film Jaws.",
    "The bridge was built in 1890 and rebuilt in 1950.",
    "She was born in Paris and studied at the Sorbonne.",
] * 4      # "laid down" 4 times: below LEARN_PHRASAL_MIN, nothing is learned


def _kv_pod(kv, name, window, monkeypatch):
    from nam_heads.impls.affordance.default.storage.couchbase_kv_store import CouchbaseKVCategoryStore
    st = CouchbaseKVCategoryStore(kv, lmdb_cache=None, kv_on_cache_miss=True)
    _seed_store(st)
    h = AffordanceEncoder(head_type="affordance",
                          config={"category_store_backend": "memory", "category_store_seed": False,
                                  "lexicon_commit_batches": window})
    h.category_store = st
    h._witness = name
    return h


def _kv_ingest(h, nlp, texts, monkeypatch, batch=2):
    monkeypatch.setenv("NAM_AFF_WITNESS", h._witness)
    for k in range(0, len(texts), batch):
        chunk = texts[k:k + batch]
        h.encode_batch_texts(chunk, [{"spacy_doc": nlp.parse(t)} for t in chunk])


def _p_counts(kv):
    import json
    return {k[6:]: json.loads(v).get("count", 0)
            for k, v in kv._data.items() if k.startswith("vcat::p:")}


def test_two_pods_with_windows_count_what_one_pod_counts_per_batch(nlp, monkeypatch):
    from nam_heads.impls._testing.fake_kv import FakeKV
    single = FakeKV()
    one = _kv_pod(single, "pod-0", 1, monkeypatch)
    _kv_ingest(one, nlp, POD_TEXTS, monkeypatch)
    one.category_store.get_docs_batch([])           # flush deferred writes
    want = _p_counts(single)

    kv = FakeKV()
    a = _kv_pod(kv, "pod-a", 3, monkeypatch)
    b = _kv_pod(kv, "pod-b", 3, monkeypatch)
    for k in range(0, len(POD_TEXTS), 4):           # interleaved: stale L1s
        _kv_ingest(a, nlp, POD_TEXTS[k:k + 2], monkeypatch, batch=1)
        _kv_ingest(b, nlp, POD_TEXTS[k + 2:k + 4], monkeypatch, batch=1)
    for h in (a, b):
        monkeypatch.setenv("NAM_AFF_WITNESS", h._witness)
        h.flush_lexicon()
    assert _p_counts(kv) == want and sum(want.values()) >= 36
