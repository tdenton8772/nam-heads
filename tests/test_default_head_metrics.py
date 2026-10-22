"""Tests for encoder head metrics and config_summary interfaces."""

# Each head gets an in-memory store through its config.
# Nothing here touches os.environ: a module-level NAM_SKIP_HEAD_NLP=1 set at
# collection leaked into every later test in the process (the registry's NLP
# service and its rule fallback both return None under it), and a default
# config points the axis heads' stores at the session KV on 127.0.0.1:11210.
ENTITY_CFG = {"db_backend": "memory"}
ATTRIBUTE_CFG = {"category_store_backend": "memory"}
AFFORDANCE_CFG = {"category_store_backend": "memory"}
CONTEXT_CFG = {"context_store_backend": "memory"}


class TestEncoderHeadBaseMetrics:
    """Test base EncoderHead counter and metrics behaviour."""

    def test_initial_counters_zero(self):
        from src.encoders.registry import EncoderHead

        class Stub(EncoderHead):
            def _initialize(self):
                pass
            def encode(self, text, context=None):
                pass
            def encode_query(self, query, context=None):
                pass

        head = Stub(head_type="stub")
        assert head._records_processed == 0
        assert head._errors == 0
        assert head._total_encode_ms == 0.0

    def test_get_metrics_returns_base_keys(self):
        from src.encoders.registry import EncoderHead

        class Stub(EncoderHead):
            def _initialize(self):
                pass
            def encode(self, text, context=None):
                pass
            def encode_query(self, query, context=None):
                pass

        head = Stub(head_type="stub")
        m = head.get_metrics()
        assert m["head_type"] == "stub"
        assert m["records_processed"] == 0
        assert m["errors"] == 0
        assert m["avg_encode_ms"] == 0.0

    def test_counters_affect_metrics(self):
        from src.encoders.registry import EncoderHead

        class Stub(EncoderHead):
            def _initialize(self):
                pass
            def encode(self, text, context=None):
                pass
            def encode_query(self, query, context=None):
                pass

        head = Stub(head_type="stub")
        head._records_processed = 10
        head._total_encode_ms = 50.0
        head._errors = 2
        m = head.get_metrics()
        assert m["records_processed"] == 10
        assert m["errors"] == 2
        assert m["avg_encode_ms"] == 5.0

    def test_config_summary_default_empty(self):
        from src.encoders.registry import EncoderHead

        class Stub(EncoderHead):
            def _initialize(self):
                pass
            def encode(self, text, context=None):
                pass
            def encode_query(self, query, context=None):
                pass

        head = Stub(head_type="stub")
        assert head.config_summary() == {}


class TestOntologyHeadMetrics:
    def test_config_summary_keys(self):
        from nam_heads.impls.ontology.codebook.head import OntologyEncoder
        enc = OntologyEncoder(head_type="ontology")
        cs = enc.config_summary()
        assert {"codebook", "classes", "accept"} <= set(cs)
        assert isinstance(cs["classes"], int) and cs["classes"] > 0

    def test_get_metrics_returns_base_keys(self):
        from nam_heads.impls.ontology.codebook.head import OntologyEncoder
        enc = OntologyEncoder(head_type="ontology")
        m = enc.get_metrics()
        assert m["head_type"] == "ontology"
        assert "records_processed" in m
        assert "avg_encode_ms" in m


class TestEntityHeadMetrics:
    def test_config_summary_keys(self):
        from nam_heads.impls.entity.default.head import EntityEncoder
        enc = EntityEncoder(head_type="entity", config=ENTITY_CFG)
        cs = enc.config_summary()
        assert "max_entities_per_record" in cs
        assert "min_entity_confidence" in cs
        assert "enable_fuzzy_resolution" in cs
        assert "storage_backend" in cs

    def test_get_metrics_returns_base_plus_store(self):
        from nam_heads.impls.entity.default.head import EntityEncoder
        enc = EntityEncoder(head_type="entity", config=ENTITY_CFG)
        m = enc.get_metrics()
        assert m["head_type"] == "entity"
        assert "records_processed" in m


class TestAttributeHeadMetrics:
    def test_config_summary_keys(self):
        from nam_heads.impls.attribute.default.head import AttributeEncoder
        enc = AttributeEncoder(head_type="attribute", config=ATTRIBUTE_CFG)
        cs = enc.config_summary()
        assert "max_attributes" in cs
        assert "ontology_aware" in cs
        assert "category_store_backend" in cs

    def test_get_metrics_returns_category_store_size(self):
        from nam_heads.impls.attribute.default.head import AttributeEncoder
        enc = AttributeEncoder(head_type="attribute", config=ATTRIBUTE_CFG)
        m = enc.get_metrics()
        assert "category_store_size" in m
        assert isinstance(m["category_store_size"], int)


class TestAffordanceHeadMetrics:
    def test_config_summary_keys(self):
        from nam_heads.impls.affordance.default.head import AffordanceEncoder
        enc = AffordanceEncoder(head_type="affordance", config=AFFORDANCE_CFG)
        cs = enc.config_summary()
        assert "max_affordances" in cs
        assert "ontology_aware" in cs
        assert "category_store_backend" in cs

    def test_get_metrics_returns_category_store_size(self):
        from nam_heads.impls.affordance.default.head import AffordanceEncoder
        enc = AffordanceEncoder(head_type="affordance", config=AFFORDANCE_CFG)
        m = enc.get_metrics()
        assert "category_store_size" in m


class TestContextHeadMetrics:
    def test_config_summary_keys(self):
        from nam_heads.impls.context.default.head import ContextEncoder
        enc = ContextEncoder(head_type="context", config=CONTEXT_CFG)
        cs = enc.config_summary()
        assert "dimensions" in cs
        assert "max_per_dimension" in cs
        assert "context_store_backend" in cs
        assert isinstance(cs["dimensions"], list)

    def test_get_metrics_returns_context_store_size(self):
        from nam_heads.impls.context.default.head import ContextEncoder
        enc = ContextEncoder(head_type="context", config=CONTEXT_CFG)
        m = enc.get_metrics()
        assert "context_store_size" in m


class TestNLPServiceMetrics:
    def test_rule_backend_get_metrics(self):
        from nam_heads.impls.nlp.rule import RuleNLPService
        svc = RuleNLPService()
        svc.start()
        m = svc.get_metrics()
        assert m["head_type"] == "nlp"
        assert m["backend"] == "rule"
        assert m["running"] is True
        assert m["records_processed"] == 0
        assert m["avg_parse_ms"] == 0.0

    def test_rule_backend_config_summary(self):
        from nam_heads.impls.nlp.rule import RuleNLPService
        svc = RuleNLPService()
        cs = svc.config_summary()
        assert cs["backend"] == "rule"
        assert cs["deterministic"] is True

    def test_rule_backend_counters_increment(self):
        from nam_heads.impls.nlp.rule import RuleNLPService
        svc = RuleNLPService()
        svc.start()
        # Simulate counter increment (as NATS worker would do)
        svc._records_processed = 5
        svc._total_parse_ms = 10.0
        svc._errors = 1
        m = svc.get_metrics()
        assert m["records_processed"] == 5
        assert m["errors"] == 1
        assert m["avg_parse_ms"] == 2.0

    def test_nlp_service_abc_defaults(self):
        """NLPService ABC provides safe default get_metrics/config_summary."""
        from src.nlp.interfaces import NLPService

        class MinimalNLP(NLPService):
            def parse(self, text):
                return None
            def parse_batch(self, texts):
                return [None] * len(texts)
            def start(self):
                pass
            def stop(self):
                pass
            @property
            def is_running(self):
                return False
            @property
            def backend_name(self):
                return "minimal"

        svc = MinimalNLP()
        m = svc.get_metrics()
        assert m["head_type"] == "nlp"
        assert m["backend"] == "minimal"
        cs = svc.config_summary()
        assert cs["backend"] == "minimal"
