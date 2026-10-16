"""CCO codebook: features, scoring, resolution, artifact validation, roll-up."""

import json
from collections import Counter

import numpy as np
import pytest

from nam_heads.sdk import host

# These tests run the codebook against real parses and its parser fingerprint,
# so they need the host's parser (NAM_SRC=<nam checkout>/nam, see conftest.py).
# tests/test_conformance.py covers the head without one.
pytestmark = pytest.mark.skipif(not host.has_parser(), reason="needs a host parser")

from nam_heads.impls.ontology.codebook import features
from nam_heads.impls.ontology.codebook.codebook import (
    UNCLASSIFIED, Codebook, CodebookError, parser_fingerprint,
)
from nam_heads.impls.ontology.codebook.training.cco_train import rollup
from nam_heads.sdk.testing import parser_or_skip
from nam_heads.ontology import bindings as B

PERSON, CITY_PARENT, ACT = B.PERSON, B.GEOSPATIAL_REGION, B.ACT
CITY = "cco:ont00000887"           # City < ... < Geospatial Region
GEO_LOCATION = B.GEOSPATIAL_LOCATION


@pytest.fixture(scope="module")
def nlp():
    return parser_or_skip()


def _span(nlp, text, i=0):
    return list(nlp.parse(text).sents)[i]


# ---- features -------------------------------------------------------------

def test_predicate_noun_is_the_phrase_head(nlp):
    sp = _span(nlp, "Albert Einstein was a German-born theoretical physicist.")
    c = dict(features.cues(list(sp), sp.ents))
    assert c["pred:physicist"] == features.WEIGHTS["pred"]
    assert "predmod:einstein" not in c          # subject side is not a modifier


def test_subject_pronoun_cue(nlp):
    sp = _span(nlp, "He received the Nobel Prize in Physics in 1921.")
    keys = {k for k, _ in features.cues(list(sp), sp.ents)}
    assert "subjpron:person" in keys
    sp = _span(nlp, "It borders Kentucky to the north.")
    assert "subjpron:thing" in {k for k, _ in features.cues(list(sp), sp.ents)}


def test_subject_head_for_passive(nlp):
    sp = _span(nlp, "The Battle of Waterloo was fought on Sunday 18 June 1815.")
    assert "subjhead:battle" in {k for k, _ in features.cues(list(sp), sp.ents)}


def test_vectors_are_unit_and_length_invariant(nlp):
    a = features.vectorize_span(_span(nlp, "Tennessee is a state."))
    b = features.vectorize_span(_span(nlp, "Tennessee is a landlocked state in the Southeastern region."))
    assert abs(np.linalg.norm(a) - 1) < 1e-5 and abs(np.linalg.norm(b) - 1) < 1e-5
    assert float(a @ b) > 0.6                    # same predicate dominates both


def test_empty_sentence_vector_is_zero():
    assert not np.any(features.vectorize([], []))


def test_feature_spec_tracks_weights(monkeypatch):
    import hashlib
    spec = hashlib.sha256((features.FEATURE_VERSION + repr(features.DIM)
                           + repr(sorted(features.WEIGHTS.items()))
                           + repr(features.MIN_TOKENS)
                           + features.POINTER_RE.pattern
                           + repr([(k, r.pattern) for k, r in features.TITLE_KINDS])
                           ).encode()).hexdigest()
    assert spec == features.FEATURE_SPEC


# ---- list-style title cue ---------------------------------------------------

class _T:
    def __init__(self, text, punct=False):
        self.text, self.is_punct = text, punct
        self.lemma_, self.pos_, self.dep_, self.like_num = text.lower(), "PROPN", "", False


def _toks(*words):
    return [_T(w, punct=(w in ".,")) for w in words]


@pytest.mark.parametrize("first,kind", [
    ("List_of_Polydor_Records_artists", "list"),
    ("Lists_of_animated_television_series", "list"),
    ("Timeline_of_Italian_music", "timeline"),
    ("National_Register_of_Historic_Places_listings_in_Allegan_County", "listings"),
    ("The_Corrs_discography", "discography"),
    ("Glossary_of_chess", "glossary"),
    ("Albert_Einstein", None),
    ("Listing_agent", None),
])
def test_title_kind(first, kind):
    assert features.title_kind(_toks(first, ".")) == kind


def test_running_text_never_gets_a_title_cue():
    assert features.title_kind(_toks("List", "of", "things", ".")) is None
    assert features.title_kind(_toks("The", "list_of", "x")) is None


def test_title_fragment_gets_only_the_title_cue():
    assert features.span_cues(_toks("List_of_anthropologists", ".")) == [("title:list", 4.0)]


def test_plain_fragment_and_pointer_get_none():
    assert features.span_cues(_toks("Albert_Einstein", ".")) == []
    assert features.span_cues(_toks("See", "also", "the", "list", ".")) == []


def test_title_merged_into_a_sentence_keeps_its_own_cues():
    out = features.span_cues(_toks("Timeline_of_Italian_music", ".", "1000", "AD", "music", "."))
    assert out[0] == ("title:timeline", 4.0) and len(out) > 1


def test_sparse_and_dense_agree_on_title_spans():
    class Span(list):
        ents = []
    for sp in (Span(_toks("List_of_anthropologists", ".")),
               Span(_toks("Timeline_of_Italian_music", ".", "1000", "AD", "music", "."))):
        idx, val = features.sparse_span(sp)
        dense = np.zeros(features.DIM, np.float32)
        dense[idx] = val
        assert np.allclose(dense, features.vectorize_span(sp))
        assert abs(np.linalg.norm(dense) - 1.0) < 1e-5


# ---- codebook -------------------------------------------------------------

def _toy(tmp_path, accept=0.6, temperature=0.05):
    """Three classes, one centroid each, on disjoint cue directions."""
    def unit(keys):
        v = np.zeros(features.DIM, np.float32)
        for k in keys:
            i, s = features._bucket(k)
            v[i] += s
        return v / np.linalg.norm(v)
    C = np.vstack([unit(["pred:physicist", "subjpron:person"]),
                   unit(["pred:city", "pred:town"]),
                   unit(["pred:village", "pred:hamlet"])])
    cb = Codebook(C, np.array([0, 1, 2]), [PERSON, CITY, GEO_LOCATION],
                  temperature=temperature, accept=accept, meta={"parser": parser_fingerprint()})
    cb.save(tmp_path)
    return Codebook.load(tmp_path)


def test_roundtrip_and_predict(tmp_path, nlp):
    cb = _toy(tmp_path)
    p = cb.predict_span(_span(nlp, "He was a physicist."))
    assert p.cls == PERSON and not p.abstained and p.confidence >= cb.accept


def test_no_cues_abstains(tmp_path):
    cb = _toy(tmp_path)
    p = cb.resolve(None)
    assert p.cls == UNCLASSIFIED and p.abstained


def test_backoff_to_common_ancestor(tmp_path):
    """City vs Geospatial Location split 50/50: neither clears 0.6, but both are
    Geospatial Regions, so the head says Geospatial Region."""
    cb = _toy(tmp_path)
    p = cb.resolve(np.array([0.0, 0.5, 0.5]))
    assert p.cls == CITY_PARENT and p.backed_off and p.confidence == pytest.approx(1.0)


def test_disagreement_across_branches_abstains(tmp_path):
    cb = _toy(tmp_path)
    p = cb.resolve(np.array([0.5, 0.5, 0.0]))     # Person vs City: share only BFO top classes
    # the deepest shared ancestor below the root is 'independent continuant'
    assert p.backed_off or p.abstained
    assert p.cls in (B.INDEPENDENT_CONTINUANT, UNCLASSIFIED)


def test_record_class_from_first_sentence_person(tmp_path, nlp):
    cb = _toy(tmp_path)
    doc = nlp.parse("He was a physicist. He was a physicist. It is a city.")
    rec, per = cb.predict_spans(list(doc.sents))
    assert [p.cls for p in per] == [PERSON, PERSON, CITY]
    assert rec.cls == PERSON


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m.update(schema="other/1"), "schema"),
    (lambda m: m.update(feature_spec="x"), "feature spec"),
    (lambda m: m.update(tbox="x"), "TBox"),
    (lambda m: m.update(parser="x"), "parser"),
    (lambda m: m.update(classes=["person", "x", "y"]), "registry"),
    (lambda m: m.update(temperature=0), "temperature"),
])
def test_artifact_rejected_on_mismatch(tmp_path, mutate, match):
    _toy(tmp_path)
    meta_path = tmp_path / "cco_codebook.json"
    meta = json.loads(meta_path.read_text())
    mutate(meta)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(CodebookError, match=match):
        Codebook.load(tmp_path)


def test_missing_artifact_is_an_error(tmp_path):
    with pytest.raises(CodebookError, match="missing"):
        Codebook.load(tmp_path)


def test_class_without_centroid_rejected(tmp_path):
    _toy(tmp_path)
    z = dict(np.load(tmp_path / "cco_codebook.npz"))
    z["centroid_class"] = np.array([0, 1, 1])
    np.savez_compressed(tmp_path / "cco_codebook.npz", **z)
    with pytest.raises(CodebookError, match="no centroid"):
        Codebook.load(tmp_path)


# ---- roll-up --------------------------------------------------------------

def test_rollup_moves_short_class_to_parent():
    # City short, Geospatial Region has support of its own
    t = rollup(Counter({CITY: 10, CITY_PARENT: 500, PERSON: 400}), 150)
    assert t[CITY] != CITY and t[PERSON] == PERSON
    from nam_heads.ontology.registry import get_registry
    assert get_registry().is_a(CITY, t[CITY])


def test_rollup_accumulates_children():
    t = rollup(Counter({CITY: 100, GEO_LOCATION: 100}), 150)
    # neither alone has support; together they do under a shared ancestor
    assert t[CITY] == t[GEO_LOCATION]


def test_rollup_keeps_supported_classes():
    t = rollup(Counter({PERSON: 200, ACT: 300}), 150)
    assert t == {PERSON: PERSON, ACT: ACT}


@pytest.mark.parametrize("text", [
    "See also, Abatement.",
    "This article is about communications systems in Anguilla.",
    "This is a list of characters in Ayn Rand's 1957 novel Atlas Shrugged.",
    "The following is a list of current and past artificial intelligence projects.",
    "Burgundian can refer to any of the following: someone from Burgundy.",
    "For other uses, see Mercury.",
])
def test_pointer_sentences_carry_no_cues(nlp, text):
    assert not np.any(features.vectorize_span(_span(nlp, text)))


@pytest.mark.parametrize("text", [
    "Cave diving is a form of underwater diving in caves.",
    "See Here, Private Hargrove is a 1942 book by Marion Hargrove.",
    "Seeing is the perception of light by the eye.",
])
def test_content_sentences_are_not_pointers(nlp, text):
    assert np.any(features.vectorize_span(_span(nlp, text)))


def test_record_class_is_the_first_real_sentence(tmp_path, nlp):
    cb = _toy(tmp_path)
    doc = nlp.parse("It is a city. He was a physicist. He was a physicist.")
    rec, per = cb.predict_spans(list(doc.sents))
    assert [p.cls for p in per] == [CITY, PERSON, PERSON]
    assert rec.cls == CITY            # the definition, not the majority


def test_parser_fingerprint_is_behavioural_and_pinned():
    """The shipped codebook was trained on a parser whose cues on the probe set
    match the current parser's: a CPU refactor keeps it loadable, a change in
    parse output does not."""
    from nam_heads.impls.ontology.codebook.codebook import Codebook, parser_fingerprint
    meta = Codebook.load().meta
    assert meta["parser"] == parser_fingerprint()
    assert len(parser_fingerprint()) == 64


def test_sparse_path_is_bit_identical_to_dense(nlp):
    """The head scores sparse vectors; they must give exactly the dense result."""
    from nam_heads.impls.ontology.codebook.codebook import Codebook
    cb = Codebook.load()
    doc = nlp.parse("Albert_Einstein. Albert Einstein was a German-born theoretical physicist. "
                    "He received the Nobel Prize. See also, relativity. It is a city in France.")
    for sp in doc.sents:
        a = cb.probabilities(features.vectorize_span(sp))
        b = cb.probabilities_sparse(*features.sparse_span(sp))
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a, b)


def test_cue_hash_cache_is_small_and_bounded():
    """A per-process cache must not grow with the corpus: the ontology
    container runs under a tight memory limit."""
    info = features._bucket.cache_info()
    assert info.maxsize is not None and info.maxsize <= 16384
    for i in range(info.maxsize + 500):
        features._bucket(f"noun:bound-test-{i}")
    assert features._bucket.cache_info().currsize <= info.maxsize
    assert features._bucket("pred:city") == features._bucket.__wrapped__("pred:city")
