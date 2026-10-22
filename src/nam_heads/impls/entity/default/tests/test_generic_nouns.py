"""The frozen generic-noun list and how the head applies it."""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from .. import generic_nouns as GN
from ..codebook.compound_filter import is_meaningful_compound
from ..head import EntityEncoder
from ..normalization import EntityCandidate
from ..storage.factory import create_entity_store

# The build tool is NAM's (scripts/build_entity_generic_nouns.py, it reads NAM's corpora): found through
# NAM_SRC (<nam checkout>/nam), else its tests skip.
_NAM = os.environ.get("NAM_SRC")
BUILD_SCRIPT = Path(_NAM).resolve().parent / "scripts" / "build_entity_generic_nouns.py" if _NAM else Path("/nonexistent")


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service(backend="rule")


@pytest.fixture
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.20
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


def _c(text, normalized=None, pos="NOUN", **kw):
    norm = normalized or text.lower().replace(" ", "_")
    return EntityCandidate(text=text, normalized=norm, confidence=0.25, pos_pattern=pos,
                           token_count=len(text.split()), **kw)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------

def test_artifact_is_versioned_with_provenance():
    data = json.loads(GN.ARTIFACT.read_text())
    assert data["version"] == GN.VERSION == 1
    for k in ("corpus", "corpus_sha256", "slice", "articles", "min_article_share", "parser", "count"):
        assert k in data, k
    assert data["count"] == len(data["nouns"]) == len(GN.generic_nouns())
    assert data["nouns"] == sorted(set(data["nouns"]))            # sorted, no duplicates
    assert GN.artifact_meta()["min_article_share"] == data["min_article_share"]


def test_artifact_holds_the_audit_partitions_and_not_domain_terms():
    nouns = GN.generic_nouns()
    for w in ("career", "year", "city", "community", "commitment", "dedication", "town", "influence"):
        assert w in nouns, w
    for w in ("gene", "inflation", "psychology", "terrorism", "cabbage", "haggis"):
        assert w not in nouns, w
    assert "busines" not in nouns and "campu" not in nouns        # no mangled singulars


def test_wrong_version_is_refused(tmp_path, monkeypatch):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"version": 99, "nouns": ["career"]}))
    monkeypatch.setattr(GN, "ARTIFACT", p)
    monkeypatch.setattr(GN, "_nouns", None)
    with pytest.raises(ValueError):
        GN.generic_nouns()


# ---------------------------------------------------------------------------
# is_generic_noun
# ---------------------------------------------------------------------------

def test_single_listed_word_is_generic():
    assert GN.is_generic_noun(_c("career"), "career")
    assert GN.is_generic_noun(_c("years"), "year")                 # by canonical form
    assert GN.is_generic_noun(_c("Community"), "community")        # sentence-initial capital
    assert not GN.is_generic_noun(_c("gene"), "gene")


def test_a_name_is_never_generic():
    assert not GN.is_generic_noun(_c("Career", is_name=True), "career")
    assert not GN.is_generic_noun(_c("Town", is_name=True), "town")


def test_adjectives_in_front_of_a_generic_noun():
    assert GN.is_generic_noun(_c("political landscape", pos="ADJ_NOUN"), "political_landscape")
    assert GN.is_generic_noun(_c("early years", pos="ADJ_NOUN"), "early_year")
    assert GN.is_generic_noun(_c("Scottish culture", pos="ADJ_NOUN"), "scottish_culture")
    # a noun modifier makes a term: kept
    assert not GN.is_generic_noun(_c("computer science", pos="NOUN_NOUN"), "computer_science")
    assert not GN.is_generic_noun(_c("video games", pos="NOUN_NOUN"), "video_game")
    # a head noun not on the list: kept
    assert not GN.is_generic_noun(_c("liquid mercury", pos="ADJ_NOUN"), "liquid_mercury")
    assert not GN.is_generic_noun(_c("x y", pos="UNK"), "x_y")


# ---------------------------------------------------------------------------
# Through the head
# ---------------------------------------------------------------------------

def _kept(res):
    return {(a.metadata or {})["normalized"] for a in res.addresses
            if (a.metadata or {}).get("meaningful", True)}


TEXT = ("Throughout her career she showed dedication to the community. "
        "The political landscape changed. She studied the gene and inflation.")


def test_ingest_drops_generic_nouns(encoder, nlp):
    res = encoder.encode_batch_texts([TEXT], [{"spacy_doc": nlp.parse(TEXT)}])[0]
    norms = {(a.metadata or {})["normalized"] for a in res.addresses}
    assert not norms & {"career", "dedication", "community", "political_landscape"}
    assert "inflation" in _kept(res)
    # nothing generic reached the store
    assert encoder.store.resolve_surface_form("career") is None


def test_single_path_matches_batch(encoder, nlp):
    doc = nlp.parse(TEXT)
    single = encoder._encode_internal(TEXT, mode="ingest", context={"spacy_doc": doc})
    batch = encoder.encode_batch_texts([TEXT], [{"spacy_doc": doc}])[0]
    assert [a.concept for a in single.addresses] == [a.concept for a in batch.addresses]


def test_a_title_on_the_list_is_still_the_subject(encoder, nlp):
    text = "Behavior. Behavior is how organisms respond to their surroundings."
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text), "title": "Behavior"}])[0]
    assert "behavior" in _kept(res)


def test_query_keeps_what_the_user_typed(encoder):
    res = encoder.encode_query("career")
    assert "career" in {(a.metadata or {}).get("normalized") for a in res.addresses}


# ---------------------------------------------------------------------------
# Compound filter: names pass, the article rule is for lower-case descriptions
# ---------------------------------------------------------------------------

def test_compound_filter_names_and_articles():
    # a name always passes, article or not, three words or not
    assert is_meaningful_compound("english_civil_war", "PROPN_PROPN_PROPN", "English Civil War",
                                  3, True, entity_type="compound", is_name=True)
    assert is_meaningful_compound("european_union", "ADJ_NOUN", "European Union", 2, False,
                                  entity_type="compound_noun", is_name=True, has_article=True)
    # a lower-case description introduced by an article is not
    assert not is_meaningful_compound("decisive_engagement", "ADJ_NOUN", "decisive engagement",
                                      2, False, entity_type="compound_noun", has_article=True)
    assert is_meaningful_compound("decisive_engagement", "ADJ_NOUN", "decisive engagement",
                                  2, False, entity_type="compound_noun")


def test_rule_six_counts_words_as_written():
    assert is_meaningful_compound("anti_aircraft_gun", "ADJ_NOUN", "anti-aircraft gun", 2, False,
                                  entity_type="compound_noun")
    assert not is_meaningful_compound("network_address_translation", "NOUN_NOUN_NOUN",
                                      "network address translation", 3, False,
                                      entity_type="compound_noun")


# ---------------------------------------------------------------------------
# The build script
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def build():
    if not BUILD_SCRIPT.is_file():
        pytest.skip("set NAM_SRC to a NAM checkout (its scripts/build_entity_generic_nouns.py)")
    spec = importlib.util.spec_from_file_location("build_entity_generic_nouns", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_generic_threshold(build):
    sets = {"career": {"a", "b", "c"}, "gene": {"a"}, "town": {"b", "c"}}
    assert build.select_generic(sets, 4, 0.5) == ["career", "town"]
    assert build.select_generic(sets, 4, 0.75) == ["career"]
    assert build.select_generic(sets, 0, 0.5) == []


def test_record_words_reads_the_head_candidates(build, nlp):
    enc = EntityEncoder.__new__(EntityEncoder)
    enc._rule_nlp = None
    words = build.record_words(enc, nlp, "Jane Doe",
                               "Jane Doe built a career in the community of Paris in 1990.")
    assert {"career", "community"} <= words
    assert "paris" not in words and "jane_doe" not in words and "1990" not in words


def test_build_is_deterministic(build, tmp_path, nlp):
    corpus = tmp_path / "c.jsonl"
    rows = [{"title": f"T{i}", "text": f"T{i} built a career in the town of Paris."} for i in range(6)]
    rows += [{"title": "Gene", "text": "A gene is a unit of heredity."}]
    corpus.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    outs = []
    for k in range(2):
        out = tmp_path / f"o{k}.json"
        assert build.main([str(corpus), "--seed", "1", "--skip", "0", "--take", "100",
                           "--min-article-share", "0.5", "--out", str(out)]) == 0
        outs.append(json.loads(out.read_text()))
    assert outs[0]["nouns"] == outs[1]["nouns"]
    assert "career" in outs[0]["nouns"] and "town" in outs[0]["nouns"]
    assert "gene" not in outs[0]["nouns"]
    assert outs[0]["articles"] == 7 and outs[0]["version"] == 1
