"""The gaius head is a drop-in for the ontology slot: same contract as the
codebook head, deterministic, abstains as bfo:entity, emits only active classes."""

import json

import pytest

from nam_heads.impls.ontology.gaius import head as H
from nam_heads.sdk import EncoderHead, EncodingResult
from nam_heads.ontology import bindings as B
from nam_heads.ontology.active import emittable_classes as active_classes
from nam_heads.ontology.registry import get_registry

EINSTEIN = ("Albert_Einstein. Albert Einstein was a German-born theoretical physicist. "
            "He received the Nobel Prize in Physics in 1921. He was a member of the Prussian Academy.")
TEXTS = [
    EINSTEIN,
    "Nashville. Nashville is the capital of the U.S. state of Tennessee. It is a city on the Cumberland River.",
    "List_of_Polydor_Records_artists. This is a list of artists who have recorded for Polydor Records.",
    "Agrotis_ipsilon. Agrotis ipsilon is a species of moth in the family Noctuidae.",
    "The Battle of Waterloo was fought on Sunday 18 June 1815, near Waterloo.",
    "who won the nobel prize in physics",
]
# metadata the NATS stage, bundler, attribute/affordance/context heads and the
# dashboard read from an ontology result (same keys as the codebook head)
CONTRACT_META = {"mode", "sentence_ontologies", "sentence_confidences", "backed_off",
                 "abstained", "top_confidence", "diagnostics"}


def _encode(head, nlp, text, mode="ingest"):
    ctx = {"spacy_doc": nlp.parse(text)}
    return head.encode(text, ctx) if mode == "ingest" else head.encode_query(text, ctx)


# ---- interface ---------------------------------------------------------------

def test_impl_meta_names_the_slot_and_contract():
    m = H.IMPL_META
    assert (m["slot"], m["name"], m["contract"]) == ("ontology", "gaius", "ontology-v1")
    assert m["version"] == H.VERSION and m["classes"].startswith("bfo2020-cco2.2:")
    assert set(m["requires"]) == {"numpy", "onnxruntime", "tokenizers"}
    assert "torch" not in m["requires"] and "pylate" not in m["requires"]


def test_exactly_one_encoder_head_in_head_module():
    import inspect
    subs = [o for _, o in inspect.getmembers(H, inspect.isclass)
            if issubclass(o, EncoderHead) and o is not EncoderHead]
    assert subs == [H.GaiusOntologyEncoder]


def test_constructed_like_the_codebook_head(fake_head):
    h = fake_head()
    assert isinstance(h, EncoderHead)
    assert h.head_type == "ontology" == H.GaiusOntologyEncoder.HEAD_TYPE
    for name in ("encode", "encode_query", "classify_nearest", "emittable_classes",
                 "get_metrics", "config_summary"):
        assert callable(getattr(h, name))


@pytest.mark.parametrize("mode", ["ingest", "query"])
def test_result_shape_matches_the_contract(fake_head, nlp, mode):
    h = fake_head()
    reg = get_registry()
    for text in TEXTS:
        res = _encode(h, nlp, text, mode)
        assert isinstance(res, EncodingResult) and res.head_type == "ontology"
        assert res.addresses and all(a.head_type == "ontology" for a in res.addresses)
        assert all(reg.is_class(a.concept) for a in res.addresses)
        confs = [a.confidence for a in res.addresses[1:]]
        assert confs == sorted(confs, reverse=True)
        assert len({a.concept for a in res.addresses}) == len(res.addresses)
        assert CONTRACT_META <= set(res.metadata)
        assert res.metadata["mode"] == mode
        so, sc = res.metadata["sentence_ontologies"], res.metadata["sentence_confidences"]
        assert set(so) == set(sc) and all(isinstance(k, int) for k in so)
        assert all(reg.is_class(v) and v != B.UNCLASSIFIED for v in so.values())
        json.dumps(res.metadata)                    # crosses NATS as JSON


def test_title_fragment_does_not_get_a_sentence_class(fake_head, nlp):
    meta = _encode(fake_head(), nlp, EINSTEIN).metadata
    assert 0 not in meta["sentence_ontologies"]
    assert 1 in meta["sentence_ontologies"]            # the lead carries the record class


def test_lead_sentence_carries_the_record_class(fake_head, nlp):
    res = _encode(fake_head(), nlp, EINSTEIN)
    assert res.metadata["sentence_ontologies"][1] == res.addresses[0].concept


def test_max_sentences_bounds_the_work(fake_head, nlp):
    h = fake_head(max_sentences=1, text_cache_size=0)
    res = _encode(h, nlp, EINSTEIN)
    assert list(res.metadata["sentence_ontologies"]) == [1]
    assert h.encoder.calls == 1


def test_query_and_ingest_classify_alike(fake_head, nlp):
    h = fake_head()
    for text in TEXTS:
        a, b = _encode(h, nlp, text, "ingest"), _encode(h, nlp, text, "query")
        assert [x.concept for x in a.addresses] == [x.concept for x in b.addresses]


def test_classify_nearest(fake_head, nlp):
    h = fake_head()
    spans = list(nlp.parse(EINSTEIN).sents)
    assert h.classify_nearest("x", spacy_doc=None) is None
    assert h.classify_nearest("", spacy_doc=spans[0]) is None       # the title fragment
    cls = h.classify_nearest("", spacy_doc=spans[1])
    assert cls in h.emittable_classes()


def test_attach_sentence_ontologies_keeps_the_heads_map(fake_head, nlp):
    from nam_heads.sdk.sentence_ontology import attach_sentence_ontologies
    h = fake_head()
    doc = nlp.parse(EINSTEIN)
    res = h.encode(EINSTEIN, {"spacy_doc": doc})
    before = dict(res.metadata["sentence_ontologies"])
    assert attach_sentence_ontologies(h, doc, res) == before


# ---- determinism -------------------------------------------------------------

def test_same_input_same_output(fake_head, nlp):
    a, b = fake_head(text_cache_size=0), fake_head(text_cache_size=0)
    for text in TEXTS:
        ra, rb = _encode(a, nlp, text), _encode(b, nlp, text)
        assert [(x.concept, x.confidence) for x in ra.addresses] == \
               [(x.concept, x.confidence) for x in rb.addresses]
        assert ra.metadata["sentence_ontologies"] == rb.metadata["sentence_ontologies"]
        assert ra.metadata["sentence_confidences"] == rb.metadata["sentence_confidences"]


def test_cache_does_not_change_the_answer(fake_head, nlp):
    cached, uncached = fake_head(), fake_head(text_cache_size=0)
    for text in TEXTS + TEXTS:
        assert [x.concept for x in _encode(cached, nlp, text).addresses] == \
               [x.concept for x in _encode(uncached, nlp, text).addresses]


# ---- abstain -----------------------------------------------------------------

def test_no_parse_abstains_and_says_so(fake_head):
    res = fake_head().encode("Paris is the capital of France.", {})
    assert res.addresses[0].concept == B.UNCLASSIFIED
    assert res.metadata["abstained"] == "no_parse"
    assert res.addresses[0].metadata["abstained"] is True


def test_record_without_text_abstains(fake_head, nlp):
    res = _encode(fake_head(), nlp, "Hi.")
    assert res.addresses[0].concept == B.UNCLASSIFIED
    assert res.metadata["abstained"] is True
    assert res.metadata["sentence_ontologies"] == {}


def test_skip_nlp_flag_cannot_degrade_silently(fake_head, monkeypatch):
    monkeypatch.setenv("NAM_SKIP_HEAD_NLP", "1")
    res = fake_head().encode("Paris is the capital of France.", None)
    assert res.metadata["abstained"] == "no_parse"


def test_unreachable_acceptance_abstains_or_backs_off(fake_head, nlp):
    strict = fake_head(accept=1.0)
    res = _encode(strict, nlp, EINSTEIN)
    assert res.addresses[0].concept == B.UNCLASSIFIED and res.metadata["abstained"] is True
    assert res.metadata["sentence_ontologies"] == {}            # abstaining sentences are absent
    loose = fake_head(accept=0.0)
    assert _encode(loose, nlp, EINSTEIN).metadata["abstained"] is False


def test_back_off_lands_on_an_ancestor(fake_head, nlp):
    reg = get_registry()
    h = fake_head(accept=0.2)
    for text in TEXTS:
        res = _encode(h, nlp, text)
        if res.metadata["backed_off"]:
            top = res.addresses[1].concept if len(res.addresses) > 1 else None
            assert top is None or reg.is_a(top, res.addresses[0].concept)


# ---- class set ---------------------------------------------------------------

def test_emits_only_active_classes(fake_head, nlp):
    h = fake_head()
    active = set(active_classes())
    assert set(h.emittable_classes()) <= active
    assert B.UNCLASSIFIED in h.emittable_classes()
    for text in TEXTS:
        for mode in ("ingest", "query"):
            res = _encode(h, nlp, text, mode)
            assert {a.concept for a in res.addresses} <= set(h.emittable_classes())
            assert set(res.metadata["sentence_ontologies"].values()) <= active


def test_emittable_set_is_closed_under_ancestors(fake_head):
    reg = get_registry()
    em = set(fake_head().emittable_classes())
    for c in em:
        assert reg.ancestors(c) <= em


def test_class_set_id_matches_impl_meta_and_is_head_independent(fake_head):
    h = fake_head()
    assert h.class_set_id == H.IMPL_META["classes"] == H.class_set_id(reversed(h.emittable_classes()))


def test_same_partitions_as_the_active_class_set(fake_head):
    """Swapping heads must not change which type partitions exist. The
    candidates were taken from active.py at calibration; if the active set
    moves (the codebook retrained), re-run tools/calibrate.py."""
    assert fake_head().class_set_id == H.class_set_id(active_classes())


def test_retrievable_classes_exclude_bfo_top_categories():
    from nam_heads.ontology.registry import BRANCHES
    cal = H.load_calibration()
    assert not ({B.ENTITY, B.CONTINUANT, B.OCCURRENT} | set(BRANCHES)) & set(cal["candidates"])


# ---- start-up refusals -------------------------------------------------------

def test_refuses_without_calibration(fake_head, monkeypatch, tmp_path):
    monkeypatch.setattr(H, "CALIBRATION_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(H.load_calibration, "__defaults__", (tmp_path / "missing.json",))
    with pytest.raises(H.GaiusError, match="calibration missing"):
        fake_head()


def test_refuses_a_calibration_for_other_verbalizations(fake_head, monkeypatch, tmp_path):
    cal = json.loads(H.CALIBRATION_PATH.read_text())
    cal["verbalizations_sha256"] = "0" * 64
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    monkeypatch.setattr(H.load_calibration, "__defaults__", (p,))
    with pytest.raises(H.GaiusError, match="verbalizations"):
        fake_head()


def test_refuses_a_calibration_for_another_tbox(monkeypatch, tmp_path):
    cal = json.loads(H.CALIBRATION_PATH.read_text())
    cal["tbox"] = {"bfo": {"sha256": "x"}}
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    with pytest.raises(H.GaiusError, match="TBox"):
        H.load_calibration(p)


def test_refuses_other_weights_or_a_model_that_scores_the_probes_differently(monkeypatch):
    from .conftest import FakeEncoder
    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "LateInteractionEncoder", FakeEncoder)    # _check_model left on
    with pytest.raises(H.GaiusError, match="weights differ"):
        H.GaiusOntologyEncoder(head_type="ontology")

    source = H.load_calibration()["model"]["source"]

    class SameWeights(FakeEncoder):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.meta = dict(self.meta, source=source)

    monkeypatch.setattr(H, "_SHARED", {})
    monkeypatch.setattr(H, "LateInteractionEncoder", SameWeights)
    with pytest.raises(H.GaiusError, match="probe"):
        H.GaiusOntologyEncoder(head_type="ontology")


def test_check_available_reports_a_missing_model(monkeypatch):
    from nam_heads.impls.ontology.gaius import encoder as E
    monkeypatch.setattr(E, "DEFAULT_DIRS", ())
    monkeypatch.delenv(E.ENV_MODEL_DIR, raising=False)
    assert "not found" in H.check_available()


# ---- metrics -----------------------------------------------------------------

def test_metrics_carry_the_slot_diagnostics_schema(fake_head, nlp):
    from nam_heads.impls.ontology.codebook.diagnostics import merge_snapshots
    h = fake_head()
    for text in TEXTS:
        _encode(h, nlp, text)
    m = h.get_metrics()
    assert m["head_type"] == "ontology"
    assert m["diagnostics"]["window"]["count"] == len(TEXTS)
    for k in ("ontology_entropy_mean", "ontology_unclassified_rate", "ontology_overactivated_rate",
              "ontology_class_entropy", "ontology_top1_share"):
        assert isinstance(m[k], float)
    merged = merge_snapshots([m["diagnostics"], m["diagnostics"]])
    assert merged["window"]["count"] == 2 * len(TEXTS)
    s = h.config_summary()
    assert s["impl"] == "gaius" and s["class_set"] == h.class_set_id


# ---- real model --------------------------------------------------------------

def test_real_model_passes_its_probe_pin_and_types_a_record(real_head, nlp):
    res = _encode(real_head, nlp, EINSTEIN)
    assert res.addresses[0].concept in real_head.emittable_classes()
    assert res.addresses[0].concept != B.UNCLASSIFIED
    assert 0 < res.addresses[0].confidence <= 1


def test_real_model_is_deterministic(real_head, nlp):
    real_head._cache.size = 0
    try:
        for text in TEXTS[:4]:
            a, b = _encode(real_head, nlp, text), _encode(real_head, nlp, text)
            assert [(x.concept, x.confidence) for x in a.addresses] == \
                   [(x.concept, x.confidence) for x in b.addresses]
    finally:
        real_head._cache.size = 4096


def test_real_model_types_clear_phrases(real_head):
    # sanity on phrases the validation set types correctly
    assert real_head._predict("a river in Poland").cls == "cco:ont00000563"   # Hydrographic Feature
    assert real_head._predict("an American actor").cls == B.PERSON
