"""Test that batch encode fallback path creates ent:: records in store."""
import pytest
from unittest.mock import MagicMock, patch, call

from nam_heads.impls.entity.default.head import EntityEncoder, _hash_entity_id
from nam_heads.impls.entity.default.normalization import EntityCandidate


class TestBatchFallbackCreatesEntity:
    """When no NLP candidates are extracted from a text, the batch encode
    falls back to a title-case bigram or first-token entity. This fallback
    entity MUST be written to the entity store so that ent:: records exist
    for entity resolution at query time."""

    def _make_head(self, store=None):
        """Create an EntityEncoder with all attributes needed by encode_batch_texts."""
        if store is None:
            store = MagicMock()
        head = EntityEncoder.__new__(EntityEncoder)
        head.store = store
        head.config = {}
        head.create_new_entities = True
        head.head_type = "entity"
        head.enable_fuzzy = False
        head.fuzzy_max_candidates = 5
        head.max_entities_per_record = 5
        head.min_confidence = 0.3
        head._rule_nlp = None
        return head

    def _run_fallback(self, store, texts=None, contexts=None):
        """Run encode_batch_texts with no NLP candidates to trigger fallback."""
        head = self._make_head(store)
        if texts is None:
            texts = ["Xyzzy Qwerty"]
        if contexts is None:
            contexts = [{"spacy_doc": None}]
        with patch("nam_heads.impls.entity.default.head.extract_candidates", return_value=[]):
            results = head.encode_batch_texts(texts, contexts=contexts)
        return results

    def test_batch_fallback_calls_create_entity(self):
        """Fallback entity in batch encode must call store.create_entity()."""
        store = MagicMock()
        results = self._run_fallback(store)

        assert len(results) == 1
        assert len(results[0].addresses) > 0
        assert store.create_entity.called, \
            "Batch fallback must call store.create_entity() for entity resolution"

    def test_batch_fallback_entity_id_matches_address(self):
        """The entity_id written to store must match the one in the address."""
        store = MagicMock()
        results = self._run_fallback(store)

        addr_entity_id = results[0].addresses[0].concept
        create_call = store.create_entity.call_args
        store_entity_id = create_call.kwargs.get("entity_id") or create_call[1].get("entity_id")

        assert store_entity_id == addr_entity_id, \
            f"Store entity_id {store_entity_id} must match address entity_id {addr_entity_id}"

    def test_batch_fallback_writes_surface_forms(self):
        """Fallback entity must include surface_forms in the store write."""
        store = MagicMock()
        results = self._run_fallback(store)

        create_call = store.create_entity.call_args
        surface_forms = create_call.kwargs.get("surface_forms") or create_call[1].get("surface_forms")
        assert surface_forms is not None and len(surface_forms) > 0, \
            "Fallback entity must include at least one surface form"

    def test_batch_fallback_metadata_has_entity_type(self):
        """Fallback entity store write must include entity_type in metadata."""
        store = MagicMock()
        results = self._run_fallback(store)

        create_call = store.create_entity.call_args
        metadata = create_call.kwargs.get("metadata") or create_call[1].get("metadata")
        assert "entity_type" in metadata, "Metadata must include entity_type"
        assert metadata.get("fallback") is True, "Metadata must mark as fallback"

    def test_non_fallback_batch_does_not_double_create(self):
        """When NLP candidates are found, no extra create_entity call should happen."""
        store = MagicMock()
        head = self._make_head(store)

        texts = ["Abraham Lincoln"]
        contexts = [{"spacy_doc": None}]

        cand = EntityCandidate(
            text="Abraham Lincoln",
            normalized="abraham_lincoln",
            pos_pattern="PROPN PROPN",
            token_count=2,
            is_propn=True,
            confidence=0.9,
        )
        with patch("nam_heads.impls.entity.default.head.extract_candidates", return_value=[cand]):
            results = head.encode_batch_texts(texts, contexts=contexts)

        assert len(results[0].addresses) > 0
        # Fallback create_entity should NOT be called (batch create handles it)
        assert not store.create_entity.called, \
            "Non-fallback path should use create_entities_batch, not create_entity"
