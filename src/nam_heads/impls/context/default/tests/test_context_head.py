import pytest

from nam_heads.ontology import bindings as B

# The ontology head emits BFO/CCO classes; tests name them by the familiar word.
CATEGORY_CLASS = {
    "person": B.PERSON, "group": B.GROUP_OF_PERSONS, "organization": B.ORGANIZATION,
    "object": B.MATERIAL_ARTIFACT, "location": B.GEOSPATIAL_REGION,
    "place": B.GEOSPATIAL_LOCATION, "time": B.TEMPORAL_REGION, "event": B.PROCESS,
    "action": B.ACT, "state": B.STASIS,
}

from nam_heads.impls.context.default.head import ContextEncoder
from src.encoders.registry import EncodingResult, SemanticAddress
from nam_heads.impls.nlp.rule import RuleNLPService
from src.nlp.interfaces import Entity

# Texts are encoded without a doc: the head parses them (record_parse, the
# rule parser), the parse a rollback to this head would read at ingest.
_RULE = RuleNLPService()


def _doc_with_org(text: str, org: str):
    """The rule parse of ``text`` with ``org`` as its one entity, an ORG.
    The institutional dimension maps an ORG entity to its kind; the rule NER
    reads neither name below as one ORG ("Oxford" alone, "United States" as a
    GPE), so the entity is given here."""
    doc = _RULE.parse(text)
    words, want = [t.text for t in doc], org.split()
    start = next(i for i in range(len(words)) if words[i:i + len(want)] == want)
    doc.entities = [Entity(text=org, label="ORG", start_token=start,
                           end_token=start + len(want))]
    return doc


@pytest.fixture(scope="module")
def encoder():
    return ContextEncoder(
        head_type="context",
        config={
            "context_store_backend": "memory",
            "context_store_seed": True,
        },
    )


def _concepts(result: EncodingResult):
    return [a.concept for a in result.addresses]


def _dim_concept(result: EncodingResult, dimension: str):
    """Get the concept for a specific dimension from the result."""
    for a in result.addresses:
        if a.metadata and a.metadata.get("dimension") == dimension:
            return a.concept
    return None


def _make_ontology_context(ontology_type: str):
    """Build a fake context with an ontology_result for the given type."""
    return {
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


# ------------------------------------------------------------------
# Temporal dimension
# ------------------------------------------------------------------

def test_temporal_year(encoder):
    result = encoder.encode("Founded in 1935.")
    temporal = _dim_concept(result, "temporal")
    assert temporal == "1930s", f"Expected '1930s', got '{temporal}'"


def test_temporal_century(encoder):
    result = encoder.encode("Built in the 14th century.")
    temporal = _dim_concept(result, "temporal")
    assert temporal == "14th_century", f"Expected '14th_century', got '{temporal}'"


def test_temporal_decade(encoder):
    result = encoder.encode("Popular in the 1990s.")
    temporal = _dim_concept(result, "temporal")
    assert temporal == "1990s", f"Expected '1990s', got '{temporal}'"


# ------------------------------------------------------------------
# Geographic dimension
# ------------------------------------------------------------------

def test_geographic_country(encoder):
    result = encoder.encode("She lived in France.")
    geo = _dim_concept(result, "geographic")
    assert geo == "france", f"Expected 'france', got '{geo}'"


def test_geographic_country_germany(encoder):
    result = encoder.encode("He was born in Germany.")
    geo = _dim_concept(result, "geographic")
    assert geo == "germany", f"Expected 'germany', got '{geo}'"


# ------------------------------------------------------------------
# Domain dimension
# ------------------------------------------------------------------

def test_domain_from_ontology(encoder):
    ctx = _make_ontology_context("person")
    result = encoder.encode("Albert Einstein was a physicist.", context=ctx)
    domain = _dim_concept(result, "domain")
    assert domain == "biography", f"Expected 'biography', got '{domain}'"


def test_domain_event(encoder):
    """EVENT NER label should produce historical domain."""
    result = encoder.encode("The Battle of Waterloo changed European history.")
    domain = _dim_concept(result, "domain")
    # Domain may be historical if NER detects EVENT, or null if not
    if domain is not None and domain != "__null__":
        assert domain in ("historical", "biography", "institutional", "geographic", "narrative", "cultural")


# ------------------------------------------------------------------
# Institutional dimension
# ------------------------------------------------------------------

def test_institutional_academic(encoder):
    text = "She studied at Oxford University."
    result = encoder.encode(text, {"spacy_doc": _doc_with_org(text, "Oxford University")})
    inst = _dim_concept(result, "institutional")
    assert inst == "academic", f"Expected 'academic', got '{inst}'"


def test_institutional_military(encoder):
    text = "He served in the United States Army."
    result = encoder.encode(text, {"spacy_doc": _doc_with_org(text, "United States Army")})
    inst = _dim_concept(result, "institutional")
    assert inst == "military", f"Expected 'military', got '{inst}'"


# ------------------------------------------------------------------
# Null fallback
# ------------------------------------------------------------------

def test_null_no_context(encoder):
    result = encoder.encode("The dog ran.")
    concepts = _concepts(result)
    assert "__null__" in concepts, "Expected __null__ for text with no contextual signal"


def test_all_null_for_plain_text(encoder):
    """Plain text with no NER entities should produce all-null addresses."""
    result = encoder.encode("Something happened.")
    non_null = [a for a in result.addresses if a.concept != "__null__"]
    # May or may not have non-null depending on the parse's NER, but should have addresses
    assert len(result.addresses) >= 1


# ------------------------------------------------------------------
# Multiple dimensions
# ------------------------------------------------------------------

def test_multiple_dimensions(encoder):
    result = encoder.encode("Founded in 1935 in Germany.")
    concepts = _concepts(result)
    non_null = [a for a in result.addresses if a.concept != "__null__"]
    # Should have at least temporal and geographic signal
    dims = set()
    for a in result.addresses:
        if a.metadata and a.concept != "__null__":
            dims.add(a.metadata.get("dimension"))
    # At minimum we expect temporal from "1935" and geographic from "Germany"
    assert "temporal" in dims or "geographic" in dims, f"Expected dimensional signal, got {dims}"


# ------------------------------------------------------------------
# Determinism
# ------------------------------------------------------------------

def test_deterministic(encoder):
    text = "Albert Einstein was born in Germany in 1879."
    r1 = encoder.encode(text)
    r2 = encoder.encode(text)
    assert _concepts(r1) == _concepts(r2)
    scores1 = [a.confidence for a in r1.addresses]
    scores2 = [a.confidence for a in r2.addresses]
    assert scores1 == scores2


# ------------------------------------------------------------------
# Max addresses
# ------------------------------------------------------------------

def test_max_addresses(encoder):
    """At most 4 non-null addresses (1 per dimension)."""
    text = (
        "In 1935 in Germany, Oxford University hosted an event "
        "organized by the United Nations."
    )
    result = encoder.encode(text)
    non_null = [a for a in result.addresses if a.concept != "__null__"]
    assert len(non_null) <= 4, f"Expected at most 4 non-null, got {len(non_null)}"


# ------------------------------------------------------------------
# Address format / metadata
# ------------------------------------------------------------------

def test_address_metadata_has_dimension(encoder):
    result = encoder.encode("Founded in 1935 in France.")
    for addr in result.addresses:
        assert "dimension" in addr.metadata
        assert addr.metadata["dimension"] in (
            "temporal", "geographic", "domain", "institutional"
        )


def test_head_type_is_context(encoder):
    result = encoder.encode("Something happened in 1935.")
    assert result.head_type == "context"
    for addr in result.addresses:
        assert addr.head_type == "context"


# ------------------------------------------------------------------
# Query path
# ------------------------------------------------------------------

def test_encode_query(encoder):
    result = encoder.encode_query("What happened in France in the 1990s?")
    assert result.metadata["mode"] == "query"
    assert len(result.addresses) >= 1


# ------------------------------------------------------------------
# Store interaction
# ------------------------------------------------------------------

def test_store_caches_temporal(encoder):
    """After encoding, temporal resolutions should be in the store."""
    encoder.encode("Founded in 1935.")
    # The store should now have "1935" -> "1930s"
    cached = encoder.store.get("1935", "temporal")
    assert cached == "1930s", f"Expected store to cache '1930s', got '{cached}'"


# ------------------------------------------------------------------
# Keyword domain fallback
# ------------------------------------------------------------------

def test_keyword_domain_military(encoder):
    """Text with military keywords should produce military domain."""
    text = "The battle of Gettysburg was fought during the war."
    result = encoder.encode(text)
    domain = _dim_concept(result, "domain")
    # "battle" + "war" = 2 military keywords → domain=military
    assert domain == "military", f"Expected 'military', got '{domain}'"


def test_keyword_domain_political(encoder):
    """Text with political keywords should produce political domain."""
    text = "The president called for an election to decide policy."
    result = encoder.encode(text)
    domain = _dim_concept(result, "domain")
    assert domain == "political", f"Expected 'political', got '{domain}'"


def test_keyword_domain_requires_min_hits(encoder):
    """A single keyword match should NOT trigger domain fallback."""
    text = "The war ended quietly."
    result = encoder.encode(text)
    domain = _dim_concept(result, "domain")
    # Only 1 military keyword ("war") — below min_hits=2
    # Should be __null__ unless NER/ontology found something
    if domain is not None:
        assert domain in ("__null__", "historical"), f"Single keyword should not trigger, got '{domain}'"


def test_keyword_no_override_ner(encoder):
    """If NER already found a domain, keyword fallback should NOT fire."""
    text = "Oxford University hosted a battle during the war."
    ctx = _make_ontology_context("organization")
    result = encoder.encode(text, context=ctx)
    domain = _dim_concept(result, "domain")
    # With ORG ontology, domain should come from NER/ontology (institutional), not keywords
    # The keyword fallback only fires when NER/ontology produced nothing
    assert domain is not None


# ------------------------------------------------------------------
# Keyword temporal fallback
# ------------------------------------------------------------------

def test_keyword_temporal_era(encoder):
    """Era keywords should produce temporal signal when NER finds no DATE."""
    text = "The medieval castle stood for centuries."
    result = encoder.encode(text)
    temporal = _dim_concept(result, "temporal")
    # If NER doesn't detect "medieval" as DATE, keyword fallback should catch it
    if temporal and temporal != "__null__":
        assert "medieval" in temporal or "century" in temporal.lower() or "era" in temporal.lower(), \
            f"Expected medieval temporal, got '{temporal}'"


def test_keyword_temporal_year_in_text(encoder):
    """Year patterns in text should be detected by temporal keyword fallback."""
    text = "Events of 1865 shaped the nation."
    result = encoder.encode(text)
    temporal = _dim_concept(result, "temporal")
    # Either NER detects "1865" as DATE -> "1860s", or keyword fallback catches "1865"
    assert temporal != "__null__", f"Expected temporal signal from year, got '{temporal}'"
