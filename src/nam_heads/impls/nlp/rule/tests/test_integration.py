"""Integration tests for the full rule-based NLP pipeline."""

import pytest
from .. import RuleNLPService
from src.nlp.interfaces import ParsedDocument, Token, Entity, Sentence


@pytest.fixture
def svc():
    service = RuleNLPService()
    service.start()
    yield service
    service.stop()


class TestRuleNLPService:
    def test_backend_name(self, svc):
        assert svc.backend_name == "rule"

    def test_is_running(self, svc):
        assert svc.is_running

    def test_stop(self, svc):
        svc.stop()
        assert not svc.is_running


class TestParseBasic:
    def test_returns_parsed_document(self, svc):
        doc = svc.parse("The cat sat.")
        assert isinstance(doc, ParsedDocument)
        assert len(doc.tokens) > 0
        assert len(doc.sentences) > 0

    def test_empty_string(self, svc):
        doc = svc.parse("")
        assert isinstance(doc, ParsedDocument)
        assert len(doc.tokens) == 0

    def test_whitespace_only(self, svc):
        doc = svc.parse("   ")
        assert isinstance(doc, ParsedDocument)
        assert len(doc.tokens) == 0


class TestParseBatch:
    def test_batch(self, svc):
        texts = ["The cat sat.", "Dogs bark loudly."]
        docs = svc.parse_batch(texts)
        assert len(docs) == 2
        assert all(isinstance(d, ParsedDocument) for d in docs)

    def test_empty_batch(self, svc):
        docs = svc.parse_batch([])
        assert docs == []


class TestTokenProperties:
    def test_head_facing_token_properties(self, svc):
        """Every token property the heads read (pos_, lemma_, dep_, ...) works."""
        doc = svc.parse("The angry technician repaired the pump.")
        for tok in doc.tokens:
            assert isinstance(tok.pos_, str)
            assert isinstance(tok.tag_, str)
            assert isinstance(tok.lemma_, str)
            assert isinstance(tok.dep_, str)
            assert isinstance(tok.ent_type_, str)
            assert isinstance(tok.ent_iob_, str)

    def test_head_property(self, svc):
        """token.head should return another Token."""
        doc = svc.parse("The cat sat.")
        for tok in doc.tokens:
            head = tok.head
            assert isinstance(head, Token)

    def test_children_property(self, svc):
        """token.children should yield Token objects."""
        doc = svc.parse("The angry technician repaired the pump.")
        root = [t for t in doc.tokens if t.dep_ == "ROOT"][0]
        children = list(root.children)
        assert len(children) >= 1
        assert all(isinstance(c, Token) for c in children)

    def test_morph_property(self, svc):
        """token.morph.get() should work for passive detection."""
        doc = svc.parse("The pump was repaired.")
        repaired = [t for t in doc.tokens if t.text == "repaired"][0]
        voice = repaired.morph.get("Voice")
        assert voice == ["Pass"]

    def test_idx_property(self, svc):
        """token.i should equal token.idx."""
        doc = svc.parse("The cat sat.")
        for tok in doc.tokens:
            assert tok.i == tok.idx


class TestEntitySpans:
    def test_ents_property(self, svc):
        """doc.ents should return entity spans with label_ property."""
        doc = svc.parse("He visited France in January.")
        for ent in doc.ents:
            assert hasattr(ent, "label_")
            assert hasattr(ent, "text")
            assert hasattr(ent, "start")
            assert hasattr(ent, "end")


class TestSentenceSpans:
    def test_sents_property(self, svc):
        """doc.sents should yield sentence spans."""
        doc = svc.parse("He ran. She walked.")
        sents = list(doc.sents)
        assert len(sents) == 2
        for sent in sents:
            assert hasattr(sent, "text")
            assert hasattr(sent, "start")
            assert hasattr(sent, "end")


class TestSlicing:
    def test_doc_getitem(self, svc):
        doc = svc.parse("The cat sat on the mat.")
        tok = doc[0]
        assert isinstance(tok, Token)
        assert tok.text == "The"

    def test_doc_slice(self, svc):
        doc = svc.parse("The cat sat on the mat.")
        span = doc[0:3]
        assert hasattr(span, "text")
        assert "The" in span.text

    def test_doc_iter(self, svc):
        doc = svc.parse("The cat sat.")
        texts = [t.text for t in doc]
        assert "cat" in texts

    def test_doc_len(self, svc):
        doc = svc.parse("The cat sat.")
        assert len(doc) > 0


class TestEndToEnd:
    def test_standard_sentence(self, svc):
        """Full pipeline on the benchmark sentence from the plan."""
        doc = svc.parse("The angry technician repaired the pump in Detroit.")
        # Should have tokens, entities, sentences
        assert len(doc.tokens) >= 8
        assert len(doc.sentences) == 1

        # Check key parse features
        root = [t for t in doc.tokens if t.dep_ == "ROOT"]
        assert len(root) == 1
        assert root[0].text == "repaired"
        assert root[0].lemma_ == "repair"

        # Subject
        subj = [t for t in doc.tokens if t.dep_ == "nsubj"]
        assert len(subj) == 1
        assert subj[0].text == "technician"

        # Object
        obj = [t for t in doc.tokens if t.dep_ == "dobj"]
        assert len(obj) == 1
        assert obj[0].text == "pump"

        # Adjective modifier
        amod = [t for t in doc.tokens if t.dep_ == "amod"]
        assert len(amod) >= 1
        assert amod[0].text == "angry"

    def test_passive_sentence(self, svc):
        """Passive voice should be detected."""
        doc = svc.parse("The pump was repaired by the technician.")
        repaired = [t for t in doc.tokens if t.text == "repaired"][0]
        assert repaired.dep_ == "ROOT"
        assert repaired.morph.get("Voice") == ["Pass"]
        assert repaired.tag_ == "VBN"

        subj = [t for t in doc.tokens if t.dep_ == "nsubjpass"]
        assert len(subj) == 1

    def test_complex_sentence(self, svc):
        """Test a more complex sentence with multiple clauses."""
        doc = svc.parse("The scientist discovered the formula because she studied hard.")
        assert len(doc.sentences) == 1
        root = [t for t in doc.tokens if t.dep_ == "ROOT"]
        assert len(root) == 1

    def test_deterministic(self, svc):
        """Same input should always produce same output."""
        text = "The angry technician repaired the pump in Detroit."
        doc1 = svc.parse(text)
        doc2 = svc.parse(text)

        for t1, t2 in zip(doc1.tokens, doc2.tokens):
            assert t1.text == t2.text
            assert t1.pos == t2.pos
            assert t1.tag == t2.tag
            assert t1.dep == t2.dep
            assert t1.lemma == t2.lemma
            assert t1.head_idx == t2.head_idx
            assert t1.ent_type == t2.ent_type
            assert t1.ent_iob == t2.ent_iob


class TestFactoryRegistration:
    def test_rule_backend_available(self):
        from src.nlp.factory import list_backends
        assert "rule" in list_backends()

    def test_create_via_factory(self):
        from src.nlp.factory import create_nlp_service
        svc = create_nlp_service(backend="rule")
        assert svc.backend_name == "rule"
