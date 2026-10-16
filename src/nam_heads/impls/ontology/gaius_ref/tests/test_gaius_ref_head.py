"""gaius_ref in the ontology slot: gaius's input, score and output, and the model pin."""

import inspect

import numpy as np
import pytest

from nam_heads.impls.ontology.gaius_ref import head as H
from nam_heads.impls.ontology.gaius_ref import tbox as T
from nam_heads.sdk import EncoderHead
from nam_heads.sdk.sentence_ontology import attach_sentence_ontologies
from nam_heads.ontology.registry import base_registry, get_registry

from .conftest import FakeEncoder

EINSTEIN = ("Albert_Einstein. Albert Einstein was a German-born theoretical physicist who is "
            "widely held to be one of the greatest scientists of all time.")


# ---- discovery -------------------------------------------------------------------------

def test_impl_meta_is_discovered_in_the_ontology_slot():
    from nam_heads.sdk.impl_meta import installed, read_impl_meta

    errors = []
    impl = next(i for i in installed(errors) if (i.slot, i.name) == ("ontology", "gaius_ref"))
    assert not [e for e in errors if "gaius_ref" in e]
    meta, problems = read_impl_meta(impl.head)
    assert problems == []
    assert (meta["slot"], meta["name"], meta["contract"]) == ("ontology", "gaius_ref", "ontology-v1")
    assert impl.module == "nam_heads.impls.ontology.gaius_ref.head"
    assert set(H.IMPL_META["requires"]) == {"numpy", "onnxruntime", "tokenizers"}
    assert H.IMPL_META["assets"][0]["env"] == "NAM_GAIUS_REF_MODEL_DIR"
    assert H.IMPL_META["ontology"] == {"name": "sdg", "artifact": "artifacts/sdg_ontology.json"}


def test_exactly_one_encoder_head_in_the_module():
    subs = [o for _, o in inspect.getmembers(H, inspect.isclass)
            if issubclass(o, EncoderHead) and o is not EncoderHead and o.__module__ == H.__name__]
    assert subs == [H.GaiusRefOntologyEncoder]


# ---- the model pin -----------------------------------------------------------------------

def test_weights_pin_accepts_colbert_zero_and_names_what_else_it_found():
    assert H.weights_problem({"source": H.MODEL_PIN["source"]}) is None
    msg = H.weights_problem({"source": {"model.safetensors": next(iter(H.KNOWN_WEIGHTS))}})
    assert "ModernColBERT-embed-base" in msg and "ColBERT-Zero" in msg
    assert "unknown weights" in H.weights_problem({"source": {}})


def test_refuses_a_substitute_model_by_default(monkeypatch, substitute_dir):
    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "_late_interaction_encoder", lambda d, t: FakeEncoder(d, t))
    with pytest.raises(H.GaiusRefError, match="ModernColBERT"):
        H.GaiusRefOntologyEncoder(head_type="ontology", config={"model_dir": str(substitute_dir)})


def test_runs_a_substitute_only_when_asked_and_says_so(monkeypatch, substitute_dir):
    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "_late_interaction_encoder", lambda d, t: FakeEncoder(d, t))
    h = H.GaiusRefOntologyEncoder(head_type="ontology", config={
        "model_dir": str(substitute_dir), "require_colbert_zero": False})
    assert "ModernColBERT" in h.config_summary()["model_substitute"]
    assert h.encode(EINSTEIN).metadata["model_substitute"] is True


def test_missing_model_dir_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(H, "_SHARED", {})
    with pytest.raises(H.GaiusRefError, match="not found"):
        H.GaiusRefOntologyEncoder(head_type="ontology", config={"model_dir": str(tmp_path / "none")})


def test_check_available_reports_a_substitute(monkeypatch, substitute_dir):
    monkeypatch.setenv("NAM_GAIUS_REF_MODEL_DIR", str(substitute_dir))
    assert "ModernColBERT" in H.check_available()


def test_check_available_with_the_pinned_export(monkeypatch, pinned_dir):
    monkeypatch.setenv("NAM_GAIUS_REF_MODEL_DIR", str(pinned_dir))
    assert H.check_available() is None


# ---- input: the text as given ---------------------------------------------------------------

def test_the_record_text_is_typed_verbatim_and_no_parse_is_needed(fake_head):
    h = fake_head()
    a = h.encode(EINSTEIN)
    b = h.encode(EINSTEIN, {"spacy_doc": object()})
    assert [x.concept for x in a.addresses] == [x.concept for x in b.addresses]
    assert h.encoder.queries == [EINSTEIN]          # one query, the whole text (cached after)


def test_query_path_types_the_query_the_same_way(fake_head):
    h = fake_head()
    assert ([x.concept for x in h.encode("who won the nobel prize").addresses]
            == [x.concept for x in h.encode_query("who won the nobel prize").addresses])


# ---- score and output ---------------------------------------------------------------------------

def test_score_is_summed_best_cosine_over_query_tokens_divided_by_their_count(fake_head):
    h = fake_head()
    q = h.encoder.encode_query(EINSTEIN)
    rows = T.load()["classes"]
    want = []
    for r in rows:
        d = h.encoder.encode_document(r["text"].strip())
        want.append(float((q @ d.T).max(axis=1).sum() / q.shape[0]))
    got = h.scores(EINSTEIN)
    assert np.allclose(got, want, atol=1e-5)


def test_top8_best_first_with_gaius_scores_as_confidences(fake_head):
    h = fake_head()
    s = h.scores(EINSTEIN)
    r = h.encode(EINSTEIN)
    assert len(r.addresses) == 1                              # the record class: gaius's rank 1
    top_k = r.metadata["top_k"]
    assert len(top_k) == 8 == H.TOP_K                         # gaius's grounding set
    confs = [score for _c, _iri, score in top_k]
    assert confs == sorted(confs, reverse=True)
    assert confs[0] == pytest.approx(float(s.max()))
    assert r.addresses[0].concept == h.classes[int(np.argmax(s))]
    assert r.metadata["top_confidence"] == pytest.approx(confs[0])
    assert all(a.metadata["iri"] == h.iris[h.classes.index(a.concept)] for a in r.addresses)


def test_no_threshold_low_scores_still_commit(fake_head):
    r = fake_head().encode("zz qq xx")
    assert r.addresses[0].concept != H.UNCLASSIFIED and r.metadata["abstained"] is False


def test_empty_text_returns_nothing_which_is_bfo_entity(fake_head):
    for text in ("", "   \n"):
        r = fake_head().encode(text)
        assert [a.concept for a in r.addresses] == [H.UNCLASSIFIED]
        assert r.metadata["abstained"] == "no_text"


def test_top_k_is_configurable(fake_head):
    r = fake_head(top_k=3).encode(EINSTEIN)
    assert len(r.metadata["top_k"]) == 3 and len(r.addresses) == 1


def test_items_not_sentences(fake_head):
    h = fake_head()
    r = h.encode(EINSTEIN)
    assert r.metadata["sentence_ontologies"] == {} and r.metadata["sentence_confidences"] == {}

    class Doc:
        sents = [object()]

    called = []
    h.classify_nearest = lambda *a, **k: called.append(1)
    assert attach_sentence_ontologies(h, Doc(), r) == {} and not called


def test_classes_are_the_sdg_ontologys_none_mapped(fake_head):
    h = fake_head()
    em = h.emittable_classes()
    ext = T.load_extension()
    # bfo:entity is itself one of the SDG TBox's classes, so it is not extra
    assert em == sorted(ext["emittable"]) and len(em) == 905 and H.UNCLASSIFIED in h.classes
    assert sum(c.startswith("sdg:") for c in em) == 735
    assert int(h.bfo_cco.sum()) == 168 and h.ontology_name == "sdg"
    base = base_registry()
    for c, iri, bc in zip(h.classes, h.iris, h.bfo_cco):
        assert bool(bc) is base.is_class(c)
        assert (base.iri(c) if bc else ext["classes"][c]["iri"]) == iri     # same IRI, never another class
    r = h.encode(EINSTEIN)
    assert r.metadata["outside_bfo_cco"] is (not r.addresses[0].metadata["bfo_cco"])


def test_while_selected_the_registry_knows_every_class_it_emits(fake_head):
    from nam_heads.ontology import registry as R
    h = fake_head()
    R.use_extension(str(T.EXTENSION_PATH))
    try:
        reg = get_registry()
        assert all(reg.is_class(c) for c in h.emittable_classes())
        r = h.encode(EINSTEIN)
        assert all(reg.is_class(a.concept) for a in r.addresses)
    finally:
        R.use_extension(None)


def test_deterministic(fake_head):
    a = fake_head().encode(EINSTEIN)
    b = fake_head().encode(EINSTEIN)
    assert [(x.concept, x.confidence) for x in a.addresses] == [(x.concept, x.confidence) for x in b.addresses]


# ---- the real model (skips without the pinned export) ------------------------------------------------

def test_real_model_types_text(real_head):
    r = real_head.encode(EINSTEIN)
    assert len(r.addresses) == 1 and len(r.metadata["top_k"]) == 8
    assert -1.0 <= r.addresses[0].confidence <= 1.0
    assert real_head.encode(EINSTEIN).addresses[0].concept == r.addresses[0].concept
