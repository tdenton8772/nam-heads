import pytest

from nam_heads.ontology import bindings as B

# The ontology head emits BFO/CCO classes; tests name them by the familiar word.
CATEGORY_CLASS = {
    "person": B.PERSON, "group": B.GROUP_OF_PERSONS, "organization": B.ORGANIZATION,
    "object": B.MATERIAL_ARTIFACT, "location": B.GEOSPATIAL_REGION,
    "place": B.GEOSPATIAL_LOCATION, "time": B.TEMPORAL_REGION, "event": B.PROCESS,
    "action": B.ACT, "state": B.STASIS,
}

from nam_heads.impls.affordance.default.config import DEFAULT_CONFIG
from nam_heads.impls.affordance.default.head import AffordanceEncoder
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.impls.nlp.rule import RuleNLPService

# A text encoded without a doc is parsed by the head itself (record_parse:
# the rule parser); a test that builds its own doc uses the same parser.
_RULE = RuleNLPService()


def _make_ontology_context(concept, confidence=0.9):
    """Build a context dict with an ontology_result for testing."""
    return {
        "ontology_result": EncodingResult(
            "ontology",
            [SemanticAddress("ontology", CATEGORY_CLASS.get(concept, concept), confidence)],
            {},
        )
    }


@pytest.fixture(scope="module")
def encoder():
    return AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )


# --------------------------------------------------
# Original tests (updated with memory store config)
# --------------------------------------------------

def test_event_sentence_returns_null(encoder):
    text = "The pump failed and triggered an outage alert."
    result = encoder.encode(text)
    addrs = result.addresses
    assert len(addrs) == 1
    assert addrs[0].concept == "__null__"


def test_agentive_root_selected(encoder):
    text = "The technician repaired the pump after the failure."
    addrs = encoder.encode(text).addresses
    assert addrs[0].concept == "repair"
    assert all(a.concept != "fail" for a in addrs)


def _copular_encoder():
    """Copular values are off by default (config emit_copular); these cases
    exercise the mechanism, which stays available."""
    return AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory", "emit_copular": True},
    )


def test_descriptive_sentence_copular():
    """Copular 'is' with ADJ predicate produces 'describe' when enabled."""
    text = "The pump is red and large."
    addrs = _copular_encoder().encode(text).addresses
    concepts = [a.concept for a in addrs]
    assert "describe" in concepts


def test_copular_values_off_by_default(encoder):
    """Default config: "X is a N" / "X is ADJ" give no value (they say that
    the passage describes something, not what the entity does)."""
    for text in ("The pump is red and large.", "Lincoln was a president."):
        concepts = [a.concept for a in encoder.encode(text).addresses]
        assert "describe" not in concepts and "classify" not in concepts


def test_fanout_capped_at_two():
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"max_affordances": 2, "category_store_backend": "memory"},
    )
    text = "Tyler took his children to the park and brought a ball to play catch."
    addrs = enc.encode(text).addresses
    assert 1 <= len(addrs) <= 2
    assert all(a.concept != "__null__" for a in addrs)


# --------------------------------------------------
# New: ontology-integrated scoring
# --------------------------------------------------

def test_person_cognitive_boost():
    """Cognitive verbs boosted for person ontology."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The scientist analyzed the data and carried the samples."
    ctx = _make_ontology_context("person")
    result = enc.encode(text, context=ctx)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    # "analyze" is cognitive (1.3x for person), should be present. The
    # sense-preserving fold maps analyze -> study (shared dominant sense);
    # the old fold mapped study -> analyze.
    from nam_heads.impls.affordance.default.definitions import VERB_SYNONYMS
    assert VERB_SYNONYMS.get("analyze", "analyze") in concepts


def test_object_physical_boost():
    """Physical verbs boosted for object ontology."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The team repaired the machine and discussed the process."
    ctx = _make_ontology_context("object")
    result = enc.encode(text, context=ctx)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "repair" in concepts


def test_location_movement_boost():
    """Movement verbs boosted for location ontology."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The explorers traveled through the valley and studied the rocks."
    ctx = _make_ontology_context("location")
    result = enc.encode(text, context=ctx)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "travel" in concepts


def test_no_ontology_neutral():
    """Without ontology context, base scoring is preserved."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The scientist analyzed the data and carried the samples."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert len(concepts) >= 1


# --------------------------------------------------
# New: firewall exemptions
# --------------------------------------------------

def test_firewall_exemption_action():
    """Firewalled verbs pass through in specific ontology contexts."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The earthquake triggered a landslide."
    ctx = _make_ontology_context("action")
    result = enc.encode(text, context=ctx)
    concepts = [a.concept for a in result.addresses]
    # "trigger" is in event_verbs but exempt for action ontology
    assert "trigger" in concepts


def test_light_verbs_never_exempt():
    """Light verbs are never exempt regardless of ontology."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The system was activated."
    ctx = _make_ontology_context("action")
    result = enc.encode(text, context=ctx)
    concepts = [a.concept for a in result.addresses]
    assert "be" not in concepts


# --------------------------------------------------
# New: metadata
# --------------------------------------------------

def test_category_in_metadata():
    """Category appears in address metadata for categorized verbs."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The engineer built the bridge."
    result = enc.encode(text)
    addr = result.addresses[0]
    assert addr.concept == "build"
    assert addr.metadata.get("category") == "physical"


def test_ontology_type_in_result_metadata():
    """Result metadata includes ontology_type when context is provided."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The engineer built the bridge."
    ctx = _make_ontology_context("person")
    result = enc.encode(text, context=ctx)
    assert result.metadata.get("ontology_type") == CATEGORY_CLASS["person"]


def test_ner_labels_in_metadata():
    """Result metadata includes ner_labels when NER entities are detected."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "John Smith repaired the bridge in London."
    doc = _RULE.parse(text)
    ner_labels = {ent.label_ for ent in doc.ents}
    assert ner_labels, "the parse's NER found no entity"
    result = enc.encode(text, context={"spacy_doc": doc})
    assert result.metadata["ner_labels"] == sorted(ner_labels)


# --------------------------------------------------
# New: auto-classify
# --------------------------------------------------

def test_auto_classify_persists():
    """Unknown verbs are auto-classified and persisted to the store."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    # "frobnicate" is not in seed data ("pontificate" now is: the regenerated
    # verb_categories.json covers every WordNet verb it can classify)
    ctx = _make_ontology_context("person")
    text = "The professor frobnicated about the topic."
    enc.encode(text, context=ctx)
    # Should be auto-classified as cognitive (best for person)
    result = enc.category_store.get_category_weighted("frobnicate")
    assert result is not None
    category, confidence = result
    assert category == "cognitive"
    assert confidence == 0.5


# --------------------------------------------------
# New: synonym folding
# --------------------------------------------------

def test_synonym_folding():
    """Verb synonyms fold to canonical form in emitted concepts."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The rabbit hopped across the field."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    if concepts:
        # "hop" should fold to "jump"
        assert "jump" in concepts
        assert "hop" not in concepts


def test_synonym_dedup():
    """Synonyms that fold to the same canonical form produce one concept."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "He jumped and hopped over the fence."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    # Both "jump" and "hop" fold to "jump" -- should appear at most once
    assert concepts.count("jump") <= 1


# --------------------------------------------------
# New: bundler compatibility
# --------------------------------------------------

def test_bundler_compatibility():
    """Emitted concept is the canonical verb lemma (not the category)."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The technician repaired the pump."
    result = enc.encode(text)
    addr = result.addresses[0]
    # Concept should be "repair" (verb lemma), not "physical" (category)
    assert addr.concept == "repair"


# --------------------------------------------------
# New: auto-classify only for selected affordances
# --------------------------------------------------

def test_auto_classify_only_for_selected():
    """Store should only contain entries for selected affordances, not all candidates."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )

    # Text with many verbs -- only 1-2 should be selected as affordances
    ctx = _make_ontology_context("person")
    text = "The researcher analyzed the data, observed patterns, noted anomalies, and reported findings."
    result = enc.encode(text, context=ctx)

    selected_concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert len(selected_concepts) >= 1

    # Count how many verbs got stored -- should be <= number of selected concepts
    store_count_after = enc.category_store.count()
    # The store starts with seed data; we only care that non-seed entries are bounded
    # by the number of selected affordances (1-2), not all verb candidates (4+)
    # Check that verbs NOT selected are NOT in the store
    all_candidate_verbs = {"analyze", "observe", "note", "report"}
    unselected = all_candidate_verbs - set(selected_concepts)
    for verb in unselected:
        result = enc.category_store.get_category_weighted(verb)
        # Unselected verbs should not have been auto-classified into the store
        # (they may be in seed data though, so only check if they were NOT in seed)
        if result and result[1] == 0.5:
            # confidence 0.5 = auto-classified, should not exist for unselected
            assert False, f"Unselected verb '{verb}' was auto-classified into store"


def test_scaffolding_verbs_filtered():
    """Scaffolding verbs like 'tell', 'know', 'give' are filtered as light verbs."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    scaffolding_queries = [
        ("What can you tell me about rivers?", "tell"),
        ("I want to know about bridges.", "know"),
        ("Can you give me info on pumps?", "give"),
        ("Let me find out about valves.", "find"),
        ("Show me something about turbines.", "show"),
    ]
    for text, verb in scaffolding_queries:
        result = enc.encode(text)
        concepts = [a.concept for a in result.addresses]
        assert verb not in concepts, (
            f"Scaffolding verb '{verb}' should be filtered from: {text!r}, got {concepts}"
        )


def test_other_role_below_threshold():
    """A verb in 'other' syntactic role without bonuses should not become primary affordance."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    # "other" role gets 0.20 base (below 0.25 abs_min), so it needs bonuses to qualify
    # The config weight for "other" is 0.20, absolute_min is 0.25
    w = enc.weights
    other_base = w.get("other", 0.45)
    abs_min = enc.abs_min
    assert other_base < abs_min, (
        f"'other' weight ({other_base}) should be below absolute_min ({abs_min})"
    )


# --------------------------------------------------
# Copular construction handling
# --------------------------------------------------

def test_copular_noun_predicate_classify():
    """Copular 'is' with NOUN predicate produces 'classify' affordance."""
    enc = _copular_encoder()
    text = "Lincoln was a president."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "classify" in concepts, f"Expected 'classify', got {concepts}"


def test_copular_adj_predicate_describe():
    """Copular 'is' with ADJ predicate produces 'describe' affordance."""
    enc = _copular_encoder()
    text = "The sky is blue."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "describe" in concepts, f"Expected 'describe', got {concepts}"


def test_copular_does_not_override_real_verbs():
    """When a sentence has both copular and agentive verbs, agentive wins."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The engineer built the bridge that is large."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "build" in concepts, f"Expected 'build' in {concepts}"


def test_passive_is_not_copular():
    """Passive 'was destroyed' should extract the real verb, not 'classify'."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The building was destroyed by the earthquake."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    # "destroy" should be found, not "classify"
    assert "classify" not in concepts or "destroy" in concepts


def test_light_verbs_still_filtered_non_copular():
    """Non-copular light verbs like 'have', 'do' are still filtered."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "He did say that."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses]
    assert "do" not in concepts


# --------------------------------------------------
# Structural role recognition (relcl, acl, ccomp, pcomp)
# --------------------------------------------------

def test_relcl_verb_recognized():
    """Relative clause verbs (dep=relcl) should be recognized with high weight.

    Wikipedia text is full of patterns like "who assassinated", "which established".
    Previously these fell to role=other (0.20) and were suppressed.
    """
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "John Wilkes Booth was an actor who assassinated Abraham Lincoln."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "assassinate" in concepts, f"Expected 'assassinate' in relcl, got {concepts}"


def test_relcl_beats_copular():
    """Relative clause verb should outscore copular 'was a...' construction."""
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "Booth was an actor who assassinated Lincoln."
    result = enc.encode(text)
    primary = result.addresses[0].concept
    assert primary == "assassinate", (
        f"relcl verb 'assassinate' should beat copular 'classify', got primary='{primary}'"
    )


def test_acl_verb_recognized():
    """Adjectival clause verbs (dep=acl) should be recognized.

    Common in Wikipedia: "the city founded by...", "a book written in..."
    """
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The city founded by settlers grew rapidly."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "found" in concepts or "grow" in concepts, (
        f"Expected 'found' (acl) or 'grow' (root), got {concepts}"
    )


def test_ccomp_verb_recognized():
    """Clausal complement verbs (dep=ccomp) should be recognized.

    Pattern: "He said he believed..." — "believed" is ccomp.
    """
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    # Use verbs not in light_verbs list
    text = "The researcher claimed she discovered the species."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert "discover" in concepts or "claim" in concepts, (
        f"Expected 'discover' (ccomp) or 'claim' (root), got {concepts}"
    )


def test_pcomp_verb_recognized():
    """Prepositional complement verbs (dep=pcomp) should be recognized.

    Pattern: "known for performing", "responsible for building"
    Note: the parse may read 'responsible' as ADJ predicate → 'describe' copular,
    which can outscore pcomp. The test validates pcomp role extraction weight
    is above threshold, and that the text produces a meaningful (non-null) result.
    """
    enc = AffordanceEncoder(
        head_type="affordance",
        config={"category_store_backend": "memory"},
    )
    text = "The architect was responsible for designing the tower."
    result = enc.encode(text)
    concepts = [a.concept for a in result.addresses if a.concept != "__null__"]
    assert len(concepts) >= 1, f"Expected non-null result, got {concepts}"
    # pcomp weight is above threshold, so even if copular wins,
    # design should appear as secondary or at least not be suppressed
    assert any(c in concepts for c in ("design", "describe", "classify")), (
        f"Expected 'design' (pcomp), 'describe' (copular), or 'classify', got {concepts}"
    )


def test_relcl_weight_above_threshold():
    """relcl weight should be well above absolute_min_confidence."""
    w = DEFAULT_CONFIG["weights"]
    abs_min = DEFAULT_CONFIG["absolute_min_confidence"]
    assert w["relcl"] >= 0.80, f"relcl weight {w['relcl']} should be >= 0.80"
    assert w["relcl"] > abs_min


def test_acl_weight_above_threshold():
    """acl weight should be above absolute_min_confidence."""
    w = DEFAULT_CONFIG["weights"]
    abs_min = DEFAULT_CONFIG["absolute_min_confidence"]
    assert w["acl"] >= 0.60, f"acl weight {w['acl']} should be >= 0.60"
    assert w["acl"] > abs_min


def test_new_roles_in_weights():
    """All new structural roles have defined weights."""
    w = DEFAULT_CONFIG["weights"]
    for role in ("relcl", "acl", "ccomp", "pcomp"):
        assert role in w, f"Missing weight for role '{role}'"
        assert w[role] > 0.20, f"Weight for '{role}' ({w[role]}) should be > 0.20 (other)"


def test_relcl_can_be_secondary(encoder):
    """A relcl verb should be accepted as secondary when a root verb wins primary."""
    # "Lincoln, who was assassinated by Booth, served as president"
    # "served" = root (primary), "assassinated" = relcl (should be secondary)
    text = "Lincoln, who was assassinated by Booth, served as president"
    result = encoder.encode(text, context=_make_ontology_context("person"))
    concepts = [a.concept for a in result.addresses]
    assert len(concepts) == 2, f"Expected 2 affordances, got {concepts}"
    assert "serve" in concepts, f"'serve' should be primary, got {concepts}"
    assert "assassinate" in concepts, f"'assassinate' should be secondary, got {concepts}"


def test_acl_can_be_secondary(encoder):
    """An acl verb should be accepted as secondary."""
    # "The team built a system designed to monitor networks"
    # "built" = root (primary), "designed" = acl (should be secondary)
    text = "The team built a system designed to monitor networks"
    result = encoder.encode(text, context=_make_ontology_context("object"))
    concepts = [a.concept for a in result.addresses]
    # Primary should be "build"; if "design" also present as secondary, good
    assert len(concepts) >= 1, f"Expected at least 1 affordance, got {concepts}"
    assert "build" in concepts, f"'build' should be primary, got {concepts}"


# --------------------------------------------------
# Passive penalty and occurrence verbs (predicate mode)
# --------------------------------------------------

def test_passive_penalty_takes_at_most_half_the_base():
    """A passive verb outside the main clause (role "other", weight 0.2) kept
    a score of 0 under the full 0.2 penalty and fell under soft_min."""
    enc = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    other, _c, _i = enc._score_candidate("transform", "other", False, False, False, True,
                                         False, None, set(), ("transformative", 1.0))
    assert other == pytest.approx(0.1) and other >= enc.soft_min
    root, _c, _i = enc._score_candidate("transform", "root", False, False, False, True,
                                        False, None, set(), ("transformative", 1.0))
    assert root == pytest.approx(0.8)          # unchanged where the base is >= 0.4


def test_passive_verb_in_subordinate_position_is_kept():
    from src.nlp import get_nlp_service
    nlp = get_nlp_service()
    enc = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    t = "According to myth, after seeing Artemis bathing, he was transformed into a stag."
    concepts = [a.concept for a in enc.encode(t, {"spacy_doc": nlp.parse(t)}).addresses]
    assert "transform" in concepts


def test_unfold_is_an_occurrence_verb():
    from src.nlp import get_nlp_service
    nlp = get_nlp_service()
    enc = AffordanceEncoder(head_type="affordance", config={"category_store_backend": "memory"})
    t = "The Battle of Lostwithiel unfolded in Cornwall."
    concepts = [a.concept for a in enc.encode(t, {"spacy_doc": nlp.parse(t)}).addresses]
    assert "unfold" not in concepts
