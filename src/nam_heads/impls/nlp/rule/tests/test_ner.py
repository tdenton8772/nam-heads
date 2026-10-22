"""Tests for the rule-based NER recognizer."""

import pytest
from ..tokenizer import Tokenizer
from ..pos_tagger import POSTagger
from ..ner import NERRecognizer
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def pipeline():
    tokenizer = Tokenizer()
    tagger = POSTagger(DATA_DIR)
    ner = NERRecognizer(DATA_DIR)
    return tokenizer, tagger, ner


def _recognize(pipeline, text):
    tokenizer, tagger, ner = pipeline
    tokens, sents = tokenizer.tokenize(text)
    tagger.tag(tokens, sents)
    entities = ner.recognize(tokens, sents)
    return tokens, entities


class TestDateTimeDetection:
    def test_month_year(self, pipeline):
        _, entities = _recognize(pipeline, "It happened in January 2024.")
        date_ents = [e for e in entities if e["label"] == "DATE"]
        assert len(date_ents) >= 1
        assert "January" in date_ents[0]["text"]

    def test_month_day_year(self, pipeline):
        _, entities = _recognize(pipeline, "On March 15, 2023 it happened.")
        date_ents = [e for e in entities if e["label"] == "DATE"]
        assert len(date_ents) >= 1

    def test_day_of_week(self, pipeline):
        _, entities = _recognize(pipeline, "We meet on Monday.")
        date_ents = [e for e in entities if e["label"] == "DATE"]
        assert len(date_ents) >= 1
        assert "Monday" in date_ents[0]["text"]

    def test_year_alone(self, pipeline):
        _, entities = _recognize(pipeline, "It was built in 1990.")
        date_ents = [e for e in entities if e["label"] == "DATE"]
        assert len(date_ents) >= 1


class TestGazetteerMatching:
    def test_country(self, pipeline):
        _, entities = _recognize(pipeline, "He traveled to France last year.")
        gpe_ents = [e for e in entities if e["label"] == "GPE"]
        assert any("France" in e["text"] for e in gpe_ents)

    def test_city(self, pipeline):
        _, entities = _recognize(pipeline, "She lives in Detroit.")
        gpe_ents = [e for e in entities if e["label"] == "GPE"]
        assert any("Detroit" in e["text"] for e in gpe_ents)

    def test_multi_word_city(self, pipeline):
        _, entities = _recognize(pipeline, "I visited New York last summer.")
        gpe_ents = [e for e in entities if e["label"] == "GPE"]
        assert any("New York" in e["text"] for e in gpe_ents)

    def test_organization(self, pipeline):
        _, entities = _recognize(pipeline, "She works at NASA.")
        org_ents = [e for e in entities if e["label"] == "ORG"]
        assert any("NASA" in e["text"] for e in org_ents)

    def test_nationality(self, pipeline):
        _, entities = _recognize(pipeline, "The French president spoke.")
        norp_ents = [e for e in entities if e["label"] == "NORP"]
        assert any("French" in e["text"] for e in norp_ents)


class TestPatternGrouping:
    def test_person_name(self, pipeline):
        _, entities = _recognize(pipeline, "I met John Smith yesterday.")
        person_ents = [e for e in entities if e["label"] == "PERSON"]
        assert len(person_ents) >= 1

    def test_multi_word_org(self, pipeline):
        _, entities = _recognize(pipeline, "The United Nations met today.")
        org_ents = [e for e in entities if e["label"] == "ORG"]
        assert any("United Nations" in e["text"] for e in org_ents)


class TestIOBTags:
    def test_iob_single_token(self, pipeline):
        tokens, entities = _recognize(pipeline, "She visited France.")
        france = [t for t in tokens if t.text == "France"][0]
        assert france.ent_iob == "B"

    def test_iob_multi_token(self, pipeline):
        tokens, entities = _recognize(pipeline, "He visited New York.")
        ny_tokens = [t for t in tokens if t.text in ("New", "York") and t.ent_type]
        if len(ny_tokens) >= 2:
            assert ny_tokens[0].ent_iob == "B"
            assert ny_tokens[1].ent_iob == "I"

    def test_outside_tokens(self, pipeline):
        tokens, _ = _recognize(pipeline, "The cat sat.")
        for t in tokens:
            if not t.ent_type:
                assert t.ent_iob == "O"


class TestGazetteerOnly:
    def test_gazetteer_match(self, pipeline):
        """NER finds entities from its gazetteers alone."""
        _, entities = _recognize(pipeline, "He went to France.")
        assert len(entities) >= 1
