"""Ingest and query steer attribute evidence by the same ontologies, and the
canonicalizer's injected descriptors never become query attribute values."""
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from src.encoders.registry import EncodingResult, SemanticAddress
from src.encoders.sentence_ontology import annotate_sentence_ontologies, attach_sentence_ontologies


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service()


@pytest.fixture(scope="module")
def ont():
    from nam_heads.impls.ontology.codebook.head import OntologyEncoder
    return OntologyEncoder(head_type="ontology")


def test_query_path_annotates_sentence_ontologies_like_ingest(nlp, ont):
    from src.encoders.registry import MultiHeadEncoder
    from nam_heads.impls.attribute.default.head import AttributeEncoder
    text = "Kenia Sinclair is a Jamaican athlete."
    doc = nlp.parse(text)
    # ingest: the ontology stage
    ing = ont.encode(text, {"spacy_doc": doc})
    attach_sentence_ontologies(ont, doc, ing)
    assert annotate_sentence_ontologies is attach_sentence_ontologies
    # query: MultiHeadEncoder.encode_query with the same doc
    mhe = MultiHeadEncoder.__new__(MultiHeadEncoder)
    mhe.encoders = {"ontology": ont,
                    "attribute": AttributeEncoder(head_type="attribute",
                                                  config={"category_store_backend": "memory"})}
    mhe._head_order = ["ontology"]
    mhe._executor = None
    res = MultiHeadEncoder.encode_query(mhe, text, {"spacy_doc": doc})
    assert res["ontology"].metadata["sentence_ontologies"] == ing.metadata["sentence_ontologies"]
    ing_attr = mhe.encoders["attribute"].encode(text, {"spacy_doc": doc, "ontology_result": ing})
    assert {a.concept for a in res["attribute"].addresses} == {a.concept for a in ing_attr.addresses}


def test_annotate_is_safe_without_doc_or_metadata(ont):
    assert attach_sentence_ontologies(ont, None, EncodingResult("ontology", [], None)) == {}
    assert attach_sentence_ontologies(None, None, EncodingResult("ontology", [], {})) == {}


def test_injected_descriptors_are_dropped_from_query_attributes():
    from src.query.coordinator import QueryCoordinator
    from src.query.canonicalizer import CanonicalQuery
    can = CanonicalQuery(text="biographical Abraham Lincoln?", original="Who was Abraham Lincoln?",
                         intent="IDENTITY", entity_descriptors=["biographical"])
    enc = {"attribute": EncodingResult("attribute", [
        SemanticAddress("attribute", "biographical", 0.6, {}),
        SemanticAddress("attribute", "american", 0.9, {})], {})}
    QueryCoordinator._drop_injected_attribute_values(enc, can)
    assert [a.concept for a in enc["attribute"].addresses] == ["american"]
    # only the descriptor -> NULL
    enc = {"attribute": EncodingResult("attribute", [SemanticAddress("attribute", "biographical", 0.6, {})], {})}
    QueryCoordinator._drop_injected_attribute_values(enc, can)
    assert [a.concept for a in enc["attribute"].addresses] == ["__null__"]
    # a descriptor the user typed is kept
    can2 = CanonicalQuery(text="geographic geographic survey", original="geographic survey",
                          intent="GENERAL", entity_descriptors=["geographic"])
    enc = {"attribute": EncodingResult("attribute", [SemanticAddress("attribute", "geographic", 0.6, {})], {})}
    QueryCoordinator._drop_injected_attribute_values(enc, can2)
    assert [a.concept for a in enc["attribute"].addresses] == ["geographic"]
