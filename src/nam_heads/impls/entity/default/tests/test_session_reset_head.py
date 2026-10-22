"""A load after a session flush writes every entity again only after the cold
reset (src/storage/session_reset.py).

The setup mirrors the reference deployment's: the entity head reads a node LMDB through
LmdbEntityStore, and the worker relays the head's writes to lmdb-sync, which
puts them in LMDB and the session bucket. Without the reset, an entity seen
before the flush resolves from LMDB, the head has already "ensured" it, and
nothing reaches the emptied bucket. That is what the 2026-10-08 rebuild did:
surf::paris and surf::albert_einstein were missing afterwards.
"""
import pytest

from src.nlp import get_nlp_service
from src.storage.lmdb_node_cache import LmdbAttributeCache
from ..head import EntityEncoder
from ..storage.lmdb_store import LmdbEntityStore
from .. import concepts as C

TEXT = "Albert Einstein was born in Ulm."


@pytest.fixture(scope="module")
def nlp():
    return get_nlp_service(backend="rule")


def _encoder(store):
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.2
    enc.enable_fuzzy = True
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = store
    enc._concepts = C.ConceptLearner(k=1)
    return enc


class _Node:
    """One node: its LMDB (written by lmdb-sync), the session bucket, and an
    entity head worker reading the LMDB."""

    def __init__(self, path):
        self.lmdb = LmdbAttributeCache(str(path), map_size=16 * 1024 * 1024, readonly=False)
        self.store = LmdbEntityStore(self.lmdb)
        self.enc = _encoder(self.store)
        self.bucket = {}

    def load(self, nlp, text):
        """Encode one batch and relay its writes; returns the keys written."""
        self.enc.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])
        writes = self.store.drain_pending_lmdb_writes()
        self.lmdb.batch_put(writes)
        self.bucket.update(writes)
        return {k for k, _v in writes}

    def cold_reset(self):
        self.lmdb.clear()                    # the lmdb-sync worker
        self.store.reset_session_state()     # the head worker
        self.enc.reset_session_state()


def _einstein(keys):
    return {k for k in keys if "albert_einstein" in k}


def test_without_the_reset_a_reload_writes_nothing_for_known_entities(tmp_path, nlp):
    node = _Node(tmp_path)
    assert _einstein(node.load(nlp, TEXT)), "the first load must create the entity"
    node.bucket.clear()                      # the session flush, nothing else
    assert not _einstein(node.load(nlp, TEXT))
    assert not _einstein(node.bucket)        # the delta the 2026-10-08 export held


def test_after_the_reset_a_reload_writes_the_entity_again(tmp_path, nlp):
    node = _Node(tmp_path)
    first = _einstein(node.load(nlp, TEXT))
    node.bucket.clear()
    node.cold_reset()
    assert node.lmdb.stat()["entries"] == 0
    again = _einstein(node.load(nlp, TEXT))
    assert again == first
    assert any(k.startswith("ent::") for k in node.bucket)
    assert _einstein(node.bucket) == first


def test_reset_session_state_reports_and_clears_the_head_sets(tmp_path, nlp):
    node = _Node(tmp_path)
    node.load(nlp, TEXT)
    out = node.enc.reset_session_state()
    assert out["ensured"] >= 1
    assert node.enc._ensured_entities == set()
    assert node.enc.reset_session_state()["ensured"] == 0


def test_reset_on_a_head_that_never_encoded():
    enc = EntityEncoder.__new__(EntityEncoder)
    assert enc.reset_session_state() == {"ensured": 0}
