"""Head stores: the merge set, store declarations, the in-memory store, and a
head reaching its stores. With NAM_SRC set, the vote_set and g_counter merges
are checked byte for byte against NAM's own context and affordance merges."""

import json
import random
import textwrap

import pytest

from nam_heads.sdk import EncoderHead, EncodingResult, SemanticAddress, host
from nam_heads.sdk import merge as M
from nam_heads.sdk.impl_meta import read_impl_meta
from nam_heads.sdk.store import (Family, MemoryStore, ReadOnlyStore, StoreSpec, StoreSpecError,
                                 parse_specs)


@pytest.fixture(autouse=True)
def fresh_stores():
    host.reset_memory_stores()
    saved = host._store_factory
    host.set_store_factory(None)
    yield
    host.set_store_factory(saved)
    host.reset_memory_stores()


# ---- merges -------------------------------------------------------------------------

def _votes(d):
    return json.dumps({"canonical_value": json.dumps(d), "confidence": 0.5}).encode()


def test_lww_and_first_writer():
    assert M.lww(b"a", b"b") == b"b" and M.lww(None, b"b") == b"b"
    assert M.first_writer(b"a", b"b") == b"a" and M.first_writer(None, b"b") == b"b"


def test_append_set_unions_in_order():
    assert json.loads(M.append_set(b'[["e1","x"]]', b'[["e2","y"],["e1","x"]]')) == [["e1", "x"], ["e2", "y"]]
    assert M.append_set(None, b"[1]") == b"[1]"
    assert M.append_set(b"[1]", b"not json") == b"not json"


def test_vote_set_is_a_crdt():
    a = _votes({"place": {"votes": {"Paris": ["w1"]}}})
    b = _votes({"place": {"votes": {"Paris": ["w2"], "Lyon": ["w3"]}}})
    ab, ba = M.vote_set(a, b), M.vote_set(b, a)
    assert ab == ba                                      # commutative
    assert M.vote_set(ab, a) == ab                       # idempotent
    doc = json.loads(json.loads(ab)["canonical_value"])
    assert doc["place"]["votes"] == {"Lyon": ["w3"], "Paris": ["w1", "w2"]}
    assert json.loads(ab)["confidence"] == round(2 / 3, 3)
    assert M.vote_set(a, b"plain") == b"plain"           # not a vote set: lww


def test_g_counter_is_a_crdt():
    a = json.dumps({"kind": "learned", "_g": {"count": {"p1": 3}}}).encode()
    b = json.dumps({"kind": "learned", "_g": {"count": {"p1": 2, "p2": 5}}}).encode()
    ab = M.g_counter(a, b)
    assert ab == M.g_counter(b, a) and M.g_counter(ab, b) == ab
    d = json.loads(ab)
    assert d["_g"]["count"] == {"p1": 3, "p2": 5} and d["count"] == 8
    seed = json.dumps({"kind": "seed", "lemma": "run"}).encode()
    assert json.loads(M.g_counter(ab, seed))["lemma"] == "run"     # seed fields win


def test_unknown_merge_is_refused():
    with pytest.raises(KeyError, match="unknown merge"):
        M.get("max")


def _random_votes(rng):
    return _votes({k: {"votes": {v: [f"w{rng.randint(0, 12)}" for _ in range(rng.randint(1, 4))]
                                 for v in rng.sample(["a", "b", "c"], rng.randint(1, 3))}}
                   for k in rng.sample(["place", "amb", "org:X"], rng.randint(1, 2))})


def _random_counter(rng):
    d = {"kind": rng.choice(["learned", "seed"]), "lemma": rng.choice(["run", "go"]),
         "_g": {f: {f"p{rng.randint(0, 3)}": rng.randint(0, 9)} for f in ("count", "bare")}}
    if rng.random() < 0.5:
        d["_go"] = {"cco:Person": {f"p{rng.randint(0, 3)}": rng.randint(1, 5)}}
    if rng.random() < 0.3:
        d["seed_version"] = rng.choice(["a1", "b2"])
    return json.dumps(d).encode()


def test_merges_match_nam_byte_for_byte():
    from nam_heads.sdk import host as H
    if not H.has_parser():   # set by conftest.py when NAM_SRC points at a NAM checkout
        pytest.skip("needs NAM_SRC")
    from src.storage.merges import counters as AFF
    from src.storage.merges import votes as CTX

    rng = random.Random(3)
    for _ in range(2000):
        a, b = _random_votes(rng), _random_votes(rng)
        assert M.vote_set(a, b) == CTX.merge_payloads(a, b)
        a, b = _random_counter(rng), _random_counter(rng)
        assert M.g_counter(a, b) == AFF.merge_payloads(a, b)


# ---- declarations ---------------------------------------------------------------------

def test_parse_specs():
    specs = parse_specs([{"name": "lexicon", "families": [{"prefix": "", "merge": "lww"},
                                                          {"prefix": "v:", "merge": "vote_set"}],
                          "miss_ttl_s": 300},
                         {"name": "private_one", "scope": "impl"}])
    assert [s.name for s in specs] == ["lexicon", "private_one"]
    assert specs[0].merge_for("v:paris") is M.vote_set and specs[0].merge_for("x") is M.lww
    assert specs[1].scope == "impl" and specs[1].families == (Family("", "lww"),)
    assert parse_specs(None) == []


@pytest.mark.parametrize("raw, why", [
    ({"name": "x"}, "must be a list"),
    ([{"name": "X"}], "name must match"),
    ([{"name": "ab"}, {"name": "ab"}], "declared twice"),
    ([{"name": "ab", "scope": "global"}], "scope"),
    ([{"name": "ab", "families": [{"prefix": "", "merge": "max"}]}], "unknown merge"),
    ([{"name": "ab", "families": []}], "non-empty"),
    ([{"name": "ab", "miss_ttl_s": -1}], "miss_ttl_s"),
])
def test_bad_specs_are_refused(raw, why):
    with pytest.raises(StoreSpecError, match=why):
        parse_specs(raw)


def test_impl_meta_with_a_bad_store_is_a_problem(tmp_path):
    d = tmp_path / "attribute" / "toy"
    d.mkdir(parents=True)
    (d / "head.py").write_text(textwrap.dedent('''
        IMPL_META = {"slot": "attribute", "name": "toy", "version": "1", "contract": "attribute-v1",
                     "stores": [{"name": "cats", "families": [{"prefix": "", "merge": "max"}]}]}
    '''))
    _, problems = read_impl_meta(d / "head.py")
    assert any("unknown merge" in p for p in problems)


# ---- the in-memory store ------------------------------------------------------------------

def test_memory_store_merges_and_reads_its_writes():
    st = MemoryStore(StoreSpec("lex", families=(Family("", "lww"), Family("v:", "vote_set"))))
    st.put("v:k", _votes({"place": {"votes": {"Paris": ["w1"]}}}))
    st.put("v:k", _votes({"place": {"votes": {"Paris": ["w2"]}}}))
    votes = json.loads(json.loads(st.get("v:k"))["canonical_value"])["place"]["votes"]["Paris"]
    assert votes == ["w1", "w2"]
    st.put_doc("plain", {"category": "color"})
    assert st.get_docs(["plain", "missing"]) == {"plain": {"category": "color"}, "missing": None}
    reads = st.reads
    st.get_many(["a", "b", "c"])
    assert st.reads == reads + 1                       # one batch, one read
    st.delete("plain")
    assert st.get("plain") is None


def test_read_only_store_refuses_writes():
    st = MemoryStore(StoreSpec("lex"), writable=False)
    with pytest.raises(ReadOnlyStore):
        st.put("k", b"v")
    with pytest.raises(ReadOnlyStore):
        st.delete("k")
    with pytest.raises(TypeError):
        MemoryStore(StoreSpec("lex")).put("k", "str")


# ---- a head and its stores ----------------------------------------------------------------

IMPL_META = {"slot": "attribute", "name": "counting", "version": "1", "contract": "attribute-v1",
             "stores": [{"name": "counts", "scope": "slot"},
                        {"name": "mine", "scope": "impl"}]}


class _Counting(EncoderHead):
    def _initialize(self):
        pass

    def encode(self, text, context=None):
        st = self.store("counts")
        for w in text.split():
            n = int(st.get(w) or b"0")
            st.put(w, str(n + 1).encode())
        return EncodingResult(self.head_type, [SemanticAddress(self.head_type, "x")], {})

    def encode_query(self, query, context=None):
        return self.encode(query, context)


def test_a_head_opens_its_declared_stores_through_the_host():
    h = _Counting("attribute")
    h.encode("red red blue")
    assert h.store("counts").get("red") == b"2"
    with pytest.raises(KeyError, match="declares no store"):
        h.store("other")
    # a second instance of the slot sees the slot-scoped store, not the impl one
    h2 = _Counting("attribute")
    assert h2.store("counts").get("red") == b"2"
    h.store("mine").put("k", b"v")
    assert h2.store("mine").get("k") == b"v"           # same impl: same private store


def test_query_mode_heads_get_read_only_stores():
    q = _Counting("attribute", config={"store_mode": "query"})
    with pytest.raises(ReadOnlyStore):
        q.encode("red")


def test_a_host_factory_is_used():
    opened = []

    def factory(spec, slot, impl, mode):
        opened.append((spec.name, slot, impl, mode))
        return MemoryStore(spec)

    host.set_store_factory(factory)
    _Counting("attribute").encode("x")
    assert opened == [("counts", "attribute", "counting", "ingest")]


def test_stores_handed_in_win():
    given = MemoryStore(StoreSpec("counts"))
    h = _Counting("attribute", stores={"counts": given})
    h.encode("a")
    assert given.get("a") == b"1"
