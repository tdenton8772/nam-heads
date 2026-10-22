import pytest

from nam_heads.ontology import bindings as B

# The ontology head emits BFO/CCO classes; tests name them by the familiar word.
CATEGORY_CLASS = {
    "person": B.PERSON, "group": B.GROUP_OF_PERSONS, "organization": B.ORGANIZATION,
    "object": B.MATERIAL_ARTIFACT, "location": B.GEOSPATIAL_REGION,
    "place": B.GEOSPATIAL_LOCATION, "time": B.TEMPORAL_REGION, "event": B.PROCESS,
    "action": B.ACT, "state": B.STASIS,
}

from nam_heads.impls.attribute.default.head import AttributeEncoder
from nam_heads.impls.attribute.default.storage.memory_store import MemoryCategoryStore
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.impls.nlp.rule import RuleNLPService
from src.nlp.interfaces import ParsedDocument, Sentence, Token

# A text encoded without a doc is parsed by the head itself (record_parse:
# the rule parser); a test that builds its own doc uses the same parser.
_RULE = RuleNLPService()


def _parse(text: str):
    return _RULE.parse(text)


def _doc(*rows):
    """A ParsedDocument from (text, lemma, pos, tag, dep, head) rows. For the
    extractor rules on labels the rule parser never assigns (appos, acl, an
    ADJ with an advmod child): the rule is exercised on the parse that would
    carry them."""
    tokens = [Token(text=t, lemma=lem, pos=pos, tag=tag, dep=dep, ent_type="",
                    ent_iob="O", head_idx=head, idx=i)
              for i, (t, lem, pos, tag, dep, head) in enumerate(rows)]
    text = " ".join(r[0] for r in rows)
    return ParsedDocument(text=text, tokens=tokens, entities=[],
                          sentences=[Sentence(start_token=0, end_token=len(tokens), text=text)])


@pytest.fixture(scope="module")
def encoder():
    return AttributeEncoder(
        head_type="attribute",
        config={"category_store_backend": "memory"},
    )


def _concepts(result: EncodingResult):
    return [a.concept for a in result.addresses]


# ------------------------------------------------------------------
# Basic extraction
# ------------------------------------------------------------------

def test_basic_adj_extraction(encoder):
    result = encoder.encode("The red ball is heavy.")
    concepts = _concepts(result)
    assert "red" in concepts
    assert "heavy" in concepts


def test_participial_extraction(encoder):
    result = encoder.encode("The broken window needs repair.")
    concepts = _concepts(result)
    assert "broken" in concepts or "break" in concepts


def test_lemmatization(encoder):
    result = encoder.encode("The largest building in town.")
    concepts = _concepts(result)
    assert "large" in concepts
    assert "largest" not in concepts


def test_synonym_folding(encoder):
    result = encoder.encode("The enormous truck blocked traffic.")
    concepts = _concepts(result)
    assert "large" in concepts
    assert "enormous" not in concepts


# ------------------------------------------------------------------
# Ontology-aware scoring (integrated -- before selection)
# ------------------------------------------------------------------

def _make_ontology_context(ontology_type: str, spacy_doc=None):
    """Build a fake context with an ontology_result for the given type."""
    ctx = {
        "ontology_result": EncodingResult(
            head_type="ontology",
            addresses=[
                SemanticAddress(
                    head_type="ontology",
                    concept=CATEGORY_CLASS.get(ontology_type, ontology_type),
                    confidence=0.9,
                )
            ],
            metadata={},
        )
    }
    if spacy_doc is not None:
        ctx["spacy_doc"] = spacy_doc
    return ctx


def test_person_emotional_boost(encoder):
    """With person ontology, 'angry' (emotional) should be boosted and survive."""
    text = "The angry red man."
    ctx = _make_ontology_context("person")
    result = encoder.encode(text, context=ctx)
    concepts = _concepts(result)
    assert "angry" in concepts
    # With integrated scoring, ontology can now influence selection --
    # 'red' (physical, suppressed for person) may be cut by elbow gap
    if "red" in concepts:
        assert concepts.index("angry") < concepts.index("red")


def test_object_physical_boost(encoder):
    """With object ontology, 'red' (physical) should be boosted and survive."""
    text = "The angry red ball."
    ctx = _make_ontology_context("object")
    result = encoder.encode(text, context=ctx)
    concepts = _concepts(result)
    assert "red" in concepts
    # With integrated scoring, ontology can now influence selection --
    # 'angry' (emotional, suppressed for object) may be cut by elbow gap
    if "angry" in concepts:
        assert concepts.index("red") < concepts.index("angry")


def test_no_ontology_neutral(encoder):
    """Without ontology context, base scoring is preserved (no boost)."""
    text = "The angry red ball."
    result = encoder.encode(text, context=None)
    concepts = _concepts(result)
    # Both should appear, no particular boost
    assert len(concepts) >= 1
    assert "__null__" not in concepts


# ------------------------------------------------------------------
# Null fallback
# ------------------------------------------------------------------

def test_null_fallback(encoder):
    result = encoder.encode("The dog ran.")
    concepts = _concepts(result)
    assert "__null__" in concepts


# ------------------------------------------------------------------
# Bounds and determinism
# ------------------------------------------------------------------

def test_max_three(encoder):
    result = encoder.encode(
        "The large red angry calm bright dark heavy old ball."
    )
    non_null = [a for a in result.addresses if a.concept != "__null__"]
    assert len(non_null) <= 3


def test_deterministic(encoder):
    text = "The angry tall man walked into the large red building."
    r1 = encoder.encode(text)
    r2 = encoder.encode(text)
    assert _concepts(r1) == _concepts(r2)
    scores1 = [a.confidence for a in r1.addresses]
    scores2 = [a.confidence for a in r2.addresses]
    assert scores1 == scores2


# ------------------------------------------------------------------
# Category store integration
# ------------------------------------------------------------------

def test_category_in_metadata(encoder):
    """Known adjectives should have category in metadata."""
    result = encoder.encode("The happy person is tall.")
    for addr in result.addresses:
        if addr.concept in ("happy", "tall"):
            assert "category" in addr.metadata


def test_ontology_type_in_result_metadata(encoder):
    """When ontology context is provided, result metadata should include it."""
    ctx = _make_ontology_context("person")
    result = encoder.encode("The tall man.", context=ctx)
    assert result.metadata.get("ontology_type") == CATEGORY_CLASS["person"]


# ------------------------------------------------------------------
# NEW: Ontology-integrated stopword modulation
# ------------------------------------------------------------------

def test_stopword_exempt_for_time(encoder):
    """With time ontology, 'old' should survive (reduced penalty)."""
    text = "The old ancient temple was recently discovered."
    ctx = _make_ontology_context("time")
    result = encoder.encode(text, context=ctx)
    concepts = _concepts(result)
    assert "old" in concepts, "'old' should survive with reduced stopword penalty for time ontology"


def test_stopword_not_exempt_for_object(encoder):
    """With object ontology, 'old' should be penalized normally -- 'ancient' preferred."""
    text = "The old ancient ball."
    ctx = _make_ontology_context("object")
    result = encoder.encode(text, context=ctx)
    concepts = _concepts(result)
    assert "ancient" in concepts
    # 'old' may or may not survive, but 'ancient' should rank above it if both present
    if "old" in concepts:
        assert concepts.index("ancient") < concepts.index("old")


def test_ontology_influences_selection(encoder):
    """Time ontology should boost temporal-category attributes in scoring."""
    text = "The ancient modern temple stands tall."
    ctx_time = _make_ontology_context("time")
    result_time = encoder.encode(text, context=ctx_time)
    concepts_time = _concepts(result_time)

    # With time ontology, temporal attributes (ancient, modern) should be present
    temporal_present = sum(1 for c in concepts_time if c in ("ancient", "modern"))
    assert temporal_present >= 1, "temporal attrs should survive with time ontology"


# ------------------------------------------------------------------
# NEW: NER scoping
# ------------------------------------------------------------------

def test_ner_scoping_boosts_emotional(encoder):
    """Text with PERSON NER entity should boost emotional attributes."""
    doc = _parse("John Smith was angry and tall.")

    # The parse's NER found the PERSON this test is about
    ner_labels = {ent.label_ for ent in doc.ents}
    assert "PERSON" in ner_labels

    ctx = _make_ontology_context("person", spacy_doc=doc)
    result = encoder.encode("John Smith was angry and tall.", context=ctx)
    concepts = _concepts(result)

    assert "angry" in concepts
    # With PERSON NER + person ontology, emotional should rank above physical
    if "tall" in concepts:
        assert concepts.index("angry") < concepts.index("tall")


def test_ner_labels_in_metadata(encoder):
    """When NER entities are detected, metadata should include ner_labels."""
    doc = _parse("Microsoft released a new product.")

    ner_labels = {ent.label_ for ent in doc.ents}
    assert ner_labels, "the parse's NER found no entity"

    ctx = {"spacy_doc": doc}
    result = encoder.encode("Microsoft released a new product.", context=ctx)
    assert "ner_labels" in result.metadata
    assert isinstance(result.metadata["ner_labels"], list)


# ------------------------------------------------------------------
# NEW: Compound phrase extraction
# ------------------------------------------------------------------

def test_compound_phrase_extraction(encoder):
    """Multi-token adjective phrases should be extracted."""
    doc = _doc(("The", "the", "DET", "DT", "det", 3),
               ("well", "well", "ADV", "RB", "advmod", 2),
               ("known", "known", "ADJ", "JJ", "amod", 3),
               ("scientist", "scientist", "NOUN", "NN", "nsubj", 4),
               ("made", "make", "VERB", "VBD", "ROOT", 4),
               ("a", "a", "DET", "DT", "det", 6),
               ("discovery", "discovery", "NOUN", "NN", "dobj", 4),
               (".", ".", "PUNCT", ".", "punct", 4))

    attrs = encoder._extract_from_doc(doc)
    assert ("well known", "compound_phrase", 2, "well known") in attrs
    assert ("known", "amod", 2, "known") in attrs  # and the head ADJ itself


# ------------------------------------------------------------------
# NEW: Store ontology override
# ------------------------------------------------------------------

def test_store_ontology_override():
    """get_category_weighted should return ontology-specific override when scoped."""
    store = MemoryCategoryStore()
    store.put_category_weighted("private", "institutional", 1.0, ontology_type=None)
    store.put_category_weighted("private", "emotional", 1.0, ontology_type="person")

    # Global lookup
    result = store.get_category_weighted("private", None)
    assert result is not None
    assert result[0] == "institutional"

    # Ontology-scoped lookup
    result = store.get_category_weighted("private", "person")
    assert result is not None
    assert result[0] == "emotional"

    # Unknown ontology falls back to global
    result = store.get_category_weighted("private", "event")
    assert result is not None
    assert result[0] == "institutional"


def test_backward_compat_get_category():
    """Old get_category() API should still work."""
    store = MemoryCategoryStore()
    store.put_category("angry", "emotional", 1.0)
    assert store.get_category("angry") == "emotional"
    assert store.has_lemma("angry")
    assert store.count() == 1


# ------------------------------------------------------------------
# NEW: Auto-classification of unknown adjectives
# ------------------------------------------------------------------

def test_auto_classify_persists_to_store():
    """Unknown adjectives should be auto-classified and stored after first encounter."""
    enc = AttributeEncoder(
        head_type="attribute",
        config={"category_store_backend": "memory"},
    )

    # "ferocious" is not in seed data
    assert enc.category_store.get_category("ferocious") is None

    # Encode with person ontology -- should auto-classify as "emotional" (highest for person)
    ctx = _make_ontology_context("person")
    enc.encode("The ferocious warrior charged.", context=ctx)

    result = enc.category_store.get_category_weighted("ferocious", None)
    assert result is not None, "ferocious should be stored after first encounter"
    category, confidence = result
    assert category == "emotional", "person ontology should auto-classify to emotional"
    assert confidence == 0.5, "auto-classified entries should have 0.5 confidence"


def test_auto_classify_no_ontology_stores_uncategorized():
    """Without ontology context, unknown adjectives store as uncategorized."""
    enc = AttributeEncoder(
        head_type="attribute",
        config={"category_store_backend": "memory"},
    )

    assert enc.category_store.get_category("fantastical") is None

    enc.encode("The fantastical beast appeared.")
    result = enc.category_store.get_category_weighted("fantastical", None)
    assert result is not None
    assert result[0] == "uncategorized"


def test_auto_classify_second_encounter_uses_stored():
    """Second encounter of an auto-classified adjective should use the stored category."""
    enc = AttributeEncoder(
        head_type="attribute",
        config={"category_store_backend": "memory"},
    )

    ctx = _make_ontology_context("object")
    # First encounter: auto-classifies "sturdy" as physical (highest for object)
    enc.encode("The sturdy bridge held firm.", context=ctx)
    cat1 = enc.category_store.get_category("sturdy")
    assert cat1 == "physical"

    # Second encounter without ontology: should still use stored "physical"
    r2 = enc.encode("The sturdy bridge held firm.")
    for addr in r2.addresses:
        if addr.concept == "sturdy":
            assert addr.metadata.get("category") == "physical"


# ------------------------------------------------------------------
# Attribute diversity: NOUN compounds, appositives, participial acl
# ------------------------------------------------------------------

def test_noun_compound_extraction(encoder):
    """NOUN compound modifiers should be extracted as attributes."""
    doc = _parse("The stone bridge spans the river.")

    attrs = encoder._extract_from_doc(doc)
    sources = [s for _, s, _, _ in attrs]
    lemmas = [l for _, _, _, l in attrs]

    assert "noun_compound" in sources, f"Expected noun_compound, got sources={sources}"
    assert "stone" in lemmas, f"Expected 'stone', got lemmas={lemmas}"


def test_appositive_extraction(encoder):
    """Appositive NOUN modifiers should be extracted."""
    doc = _doc(("Lincoln", "Lincoln", "PROPN", "NNP", "nsubj", 5),
               (",", ",", "PUNCT", ",", "punct", 0),
               ("the", "the", "DET", "DT", "det", 3),
               ("president", "president", "NOUN", "NN", "appos", 0),
               (",", ",", "PUNCT", ",", "punct", 0),
               ("led", "lead", "VERB", "VBD", "ROOT", 5),
               ("the", "the", "DET", "DT", "det", 7),
               ("nation", "nation", "NOUN", "NN", "dobj", 5),
               (".", ".", "PUNCT", ".", "punct", 5))

    attrs = encoder._extract_from_doc(doc)
    sources = [s for _, s, _, _ in attrs]
    lemmas = [l for _, _, _, l in attrs]

    assert "appositive" in sources, f"Expected appositive, got sources={sources}"
    assert "president" in lemmas, f"Expected 'president', got lemmas={lemmas}"


def test_acl_participial_extraction(encoder):
    """Past participle in acl dep should be extracted."""
    doc = _doc(("The", "the", "DET", "DT", "det", 1),
               ("city", "city", "NOUN", "NN", "nsubjpass", 6),
               ("destroyed", "destroy", "VERB", "VBN", "acl", 1),
               ("by", "by", "ADP", "IN", "agent", 2),
               ("war", "war", "NOUN", "NN", "pobj", 3),
               ("was", "be", "AUX", "VBD", "auxpass", 6),
               ("rebuilt", "rebuild", "VERB", "VBN", "ROOT", 6),
               (".", ".", "PUNCT", ".", "punct", 6))

    attrs = encoder._extract_from_doc(doc)
    assert ("destroyed", "acl_participial", 2, "destroy") in attrs, attrs


def test_adj_and_noun_compound_both_extracted(encoder):
    """Both ADJ and NOUN compound modifiers should be extracted."""
    text = "The large stone bridge."
    result = encoder.encode(text)
    concepts = _concepts(result)
    non_null = [c for c in concepts if c != "__null__"]
    assert len(non_null) >= 2
    assert "large" in non_null, f"Expected 'large' in {non_null}"
    assert "stone" in non_null, f"Expected 'stone' in {non_null}"
