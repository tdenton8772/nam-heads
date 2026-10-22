"""The concept gate: seed lookup, keys, the gate's coverage, and learning."""

import json

import numpy as np
import pytest

from .. import concepts as C
from ..normalization import EntityCandidate
from ..storage.memory_store import MemoryEntityStore


def _cand(text, *, is_name=False, is_propn=False, pos=None, start=5):
    words = text.split()
    return EntityCandidate(
        text=text, normalized="_".join(w.lower() for w in words), confidence=0.6,
        pos_pattern=pos or "_".join("ADJ" if i < len(words) - 1 else "NOUN" for i in range(len(words))),
        token_count=len(words), is_propn=is_propn, is_name=is_name, start=start,
    )


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

class TestKeys:
    def test_fold_lower_and_articles(self):
        assert C.concept_key("The Sedimentary  Rock") == "sedimentary rock"
        assert C.concept_key("an Économie mixte") == "economie mixte"

    def test_punctuation_stripped_hyphen_and_apostrophe_kept(self):
        assert C.concept_key("horse-drawn omnibus,") == "horse-drawn omnibus"
        assert C.concept_key("children's literature") == "children's literature"

    def test_split_possessive_rejoined(self):
        assert C.concept_key("children 's literature") == "children's literature"

    def test_curly_apostrophe(self):
        assert C.concept_key("children’s literature") == "children's literature"

    def test_words_and_text_agree(self):
        assert C.concept_key_words(["Dark", "rum"]) == C.concept_key("dark rum")

    def test_empty(self):
        assert C.concept_key("") == ""
        assert C.concept_key("the") == ""

    def test_hash_is_64_bit_and_stable(self):
        h = C.concept_hash("dark rum")
        assert 0 <= h < 2 ** 64
        assert h == C.concept_hash("dark rum")
        assert h != C.concept_hash("dark rums")

    @pytest.mark.parametrize("key,expect", [
        ("coffee beans", ["coffee beans", "coffee bean"]),
        ("baby teeth", ["baby teeth"]),
        ("grey wolves", ["grey wolves", "grey wolf", "grey wolfe", "grey wolv", "grey wolve"]),
        ("city libraries", ["city libraries", "city library", "city librari", "city librarie"]),
        ("sales tax", ["sales tax"]),
        ("tennis class", ["tennis class"]),
        ("high campus", ["high campus"]),
        ("", []),
    ])
    def test_variants(self, key, expect):
        assert C.key_variants(key) == expect


# --------------------------------------------------------------------------
# The seed
# --------------------------------------------------------------------------

class TestSeed:
    def test_artifact_is_sorted_unique_uint64(self):
        arr = np.load(C.ARTIFACT, mmap_mode="r")
        assert arr.dtype == np.uint64 and arr.ndim == 1
        assert bool(np.all(arr[1:] > arr[:-1]))

    def test_manifest_matches(self):
        meta = C.seed_meta()
        assert meta["concepts"] == np.load(C.ARTIFACT, mmap_mode="r").shape[0]
        assert len(meta["source_sha256"]) == 64

    @pytest.mark.parametrize("text", [
        "sedimentary rock", "Sedimentary rocks", "country music", "economic diversification",
        "the cold war", "coffee beans", "gross domestic product",
    ])
    def test_terms_in(self, text):
        assert C.in_seed(text)

    @pytest.mark.parametrize("text", [
        "powerful neighbors", "resilient resistance", "regional scope",
        "awareness underscores", "charming Bavarian city",
    ])
    def test_descriptions_out(self, text):
        assert not C.in_seed(text)

    def test_single_word_never(self):
        assert not C.in_seed("rock")
        assert not C.seed_has_key("rock")


# --------------------------------------------------------------------------
# Coverage of the gate
# --------------------------------------------------------------------------

class TestNeedsConcept:
    def test_lower_case_phrase(self):
        assert C.needs_concept(_cand("powerful neighbors"))

    def test_single_word(self):
        assert not C.needs_concept(_cand("competence"))

    def test_name(self):
        assert not C.needs_concept(_cand("Royal Navy", is_name=True))

    def test_propn_flag_or_pattern(self):
        assert not C.needs_concept(_cand("sirius star", is_propn=True))
        assert not C.needs_concept(_cand("sirius star", pos="PROPN_NOUN"))

    def test_capital_after_first_word(self):
        assert not C.needs_concept(_cand("southern Sweden"))

    def test_sentence_initial_capital_is_lower_case(self):
        assert C.needs_concept(_cand("Student leaders", start=7), {0, 7})

    def test_capital_mid_sentence_marks_a_name(self):
        assert not C.needs_concept(_cand("Voyager 2", start=9), {0, 7})
        assert not C.needs_concept(_cand("Vasa bryggeri", start=9), {0, 7})

    def test_capital_without_parse_position(self):
        assert not C.needs_concept(_cand("Student leaders", start=-1), {0})

    def test_sentence_initial_proper_noun_tag_covered_only_by_the_seed(self):
        """The tagger reads a sentence-initial capital as PROPN: a seed term
        is covered, a name is left to the old rules."""
        assert C.needs_concept(_cand("Sedimentary rock", pos="PROPN_NOUN", start=0), {0})
        assert not C.needs_concept(_cand("Quetzaltenango vendors", pos="PROPN_NOUN", start=0), {0})

    def test_proper_noun_tag_after_the_first_word(self):
        assert not C.needs_concept(_cand("the angola rebels", pos="NOUN_PROPN_NOUN"))

    def test_switch(self, monkeypatch):
        monkeypatch.setenv("NAM_ENTITY_CONCEPT_GATE", "0")
        assert not C.gate_enabled()
        monkeypatch.setenv("NAM_ENTITY_CONCEPT_GATE", "1")
        assert C.gate_enabled()


# --------------------------------------------------------------------------
# Learning
# --------------------------------------------------------------------------

class TestLearner:
    def test_promoted_on_the_kth_distinct_record(self):
        store = MemoryEntityStore()
        lr = C.ConceptLearner(k=3)
        for n, text in enumerate(["rec a", "rec b"]):
            assert lr.observe(store, [("quantum_widget", C.record_fingerprint(text))]) == set()
        assert lr.observe(store, [("quantum_widget", C.record_fingerprint("rec c"))]) == {"quantum_widget"}
        doc = json.loads(store.raw["cpt::quantum_widget"])
        assert doc["learned"] is True and len(doc["recs"]) == 3
        assert lr.stats["promoted"] == 1

    def test_same_record_counts_once(self):
        store = MemoryEntityStore()
        lr = C.ConceptLearner(k=2)
        fp = C.record_fingerprint("one record")
        for _ in range(5):
            assert lr.observe(store, [("quantum_widget", fp)]) == set()
        assert json.loads(store.raw["cpt::quantum_widget"])["recs"] == [fp]

    def test_learned_state_is_read_by_another_process(self):
        store = MemoryEntityStore()
        a = C.ConceptLearner(k=2)
        a.observe(store, [("quantum_widget", "f1")])
        a.observe(store, [("quantum_widget", "f2")])
        b = C.ConceptLearner(k=2)
        assert b.observe(store, [("quantum_widget", "f9")]) == {"quantum_widget"}

    def test_partial_count_is_continued_by_another_process(self):
        store = MemoryEntityStore()
        C.ConceptLearner(k=2).observe(store, [("quantum_widget", "f1")])
        assert C.ConceptLearner(k=2).observe(store, [("quantum_widget", "f2")]) == {"quantum_widget"}

    def test_own_writes_preferred_to_a_lagging_store(self):
        """Writes reach LMDB asynchronously: the learner keeps what it wrote."""
        class Lagging(MemoryEntityStore):
            def put_raw_batch(self, items):
                pass                                # never visible to reads
        store = Lagging()
        lr = C.ConceptLearner(k=2)
        lr.observe(store, [("quantum_widget", "f1")])
        assert lr.observe(store, [("quantum_widget", "f2")]) == {"quantum_widget"}

    def test_first_sight_needs_no_store(self):
        class Untouchable(MemoryEntityStore):
            def get_raw_batch(self, keys):
                raise AssertionError("read")

            def put_raw_batch(self, items):
                raise AssertionError("write")
        lr = C.ConceptLearner(k=1)
        assert lr.observe(Untouchable(), [("quantum_widget", "f1"), ("", "f1")]) == {"quantum_widget"}

    def test_default_is_first_sight(self, monkeypatch):
        monkeypatch.delenv("NAM_ENTITY_CONCEPT_K", raising=False)
        assert C.ConceptLearner().k == 1

    def test_query_reads_but_never_counts(self):
        store = MemoryEntityStore()
        lr = C.ConceptLearner(k=2)
        assert lr.observe(store, [("quantum_widget", "f1")], learn=False) == set()
        assert store.raw == {}

    def test_one_batch_with_k_records(self):
        store = MemoryEntityStore()
        lr = C.ConceptLearner(k=3)
        got = lr.observe(store, [("quantum_widget", f"f{i}") for i in range(3)])
        assert got == {"quantum_widget"}

    def test_unreadable_doc_is_a_fresh_count(self):
        store = MemoryEntityStore()
        store.raw["cpt::quantum_widget"] = b"{not json"
        lr = C.ConceptLearner(k=2)
        assert lr.observe(store, [("quantum_widget", "f1")]) == set()
        assert json.loads(store.raw["cpt::quantum_widget"])["recs"] == ["f1"]

    def test_caches_are_bounded(self):
        store = MemoryEntityStore()
        lr = C.ConceptLearner(k=5, cache_max=10)
        lr.observe(store, [(f"phrase_{i}", "f1") for i in range(25)])
        assert len(lr._docs) <= 10

    def test_store_without_raw_hooks(self):
        class Bare:
            pass
        assert C.ConceptLearner(k=1).observe(Bare(), [("quantum_widget", "f1")]) == {"quantum_widget"}

    def test_k_from_env(self, monkeypatch):
        monkeypatch.setenv("NAM_ENTITY_CONCEPT_K", "5")
        assert C.ConceptLearner().k == 5
        monkeypatch.setenv("NAM_ENTITY_CONCEPT_K", "0")
        assert C.learn_k() == 1


class TestMemoryStoreRaw:
    def test_round_trip(self):
        s = MemoryEntityStore()
        s.put_raw_batch([("cpt::x", b"1"), ("cpt::y", b"2")])
        assert s.get_raw_batch(["cpt::x", "cpt::z"]) == {"cpt::x": b"1"}


class TestLearnable:
    def test_term_shapes(self):
        assert C.learnable_shape(_cand("cancellation errors", pos="NOUN_NOUN"), "compound_noun")
        assert C.learnable_shape(_cand("uniliteral sign", pos="ADJ_NOUN"), "compound_noun")

    def test_an_article_still_counts(self):
        c = _cand("uniliteral sign", pos="ADJ_NOUN")
        c.has_article = True
        assert C.learnable_shape(c, "compound_noun")

    @pytest.mark.parametrize("text,pos", [
        ("audiences worldwide", "NOUN_NOUN"),     # adverb read as a noun
        ("nations shape", "NOUN_NOUN"),           # verb read as a noun
        ("cancellation erroneous", "NOUN_ADJ"),   # no noun at the end
        ("which became", "PRON_VERB"),
        ("important role", "ADJ_NOUN"),           # generic modifier and root
        ("perpetuate authority", "VERB_NOUN"),
        ("long distance runner", "ADJ_NOUN_NOUN"),  # three words
    ])
    def test_not_term_shapes(self, text, pos):
        assert not C.learnable_shape(_cand(text, pos=pos), "compound_noun")

    def test_free_modifier_never_learnable(self):
        assert "pivotal" in C.free_modifiers()
        assert not C.learnable(_cand("pivotal moment", pos="ADJ_NOUN"), "compound_noun")
        assert C.learnable(_cand("cancellation errors", pos="NOUN_NOUN"), "compound_noun")

    def test_free_list_is_sealed_and_versioned(self):
        data = json.loads(C.FREE_MODIFIERS.read_text())
        assert data["version"] == 1 and data["count"] == len(data["modifiers"])
        assert data["modifiers"] == sorted(data["modifiers"])
