"""EntityEncoder.resolve_query_entities: the one query-side entity resolver.

It must return only mentions that exist in the store (never the unresolved
hash or the fallback guess), and it must find stored entities through the
exact surf:: keys ingest writes, scoped first then unscoped.
"""

import pytest

from ..head import EntityEncoder, _hash_entity_id
from ..storage.factory import create_entity_store


@pytest.fixture
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 20
    enc.create_new_entities = True
    enc.min_confidence = 0.35
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


def _store_entity(enc, normalized, entity_type="person"):
    eid = _hash_entity_id(entity_type, normalized)
    enc.store.create_entity(entity_id=eid, surface_forms=[normalized],
                            metadata={"entity_type": entity_type})
    return eid


class TestResolveQueryEntities:
    def test_empty_store_resolves_nothing(self, encoder):
        """encode_query still emits unresolved hashes; the resolver drops them."""
        assert any(a.metadata.get("unresolved")
                   for a in encoder.encode_query("Abraham Lincoln").addresses)
        assert encoder.resolve_query_entities("Abraham Lincoln") == []

    def test_stored_entity_resolves(self, encoder):
        eid = _store_entity(encoder, "abraham_lincoln")
        out = encoder.resolve_query_entities("Abraham Lincoln")
        assert [e["entity_id"] for e in out] == [eid]
        assert out[0]["normalized"] == "abraham_lincoln"
        assert out[0]["entity_type"]

    def test_question_prefix_stripped(self, encoder):
        eid = _store_entity(encoder, "ada_lovelace")
        out = encoder.resolve_query_entities("who was Ada Lovelace")
        assert eid in [e["entity_id"] for e in out]

    def test_query_never_creates_entities(self, encoder):
        encoder.resolve_query_entities("Zzyzx Quorble")
        assert encoder.store.resolve_surface_form("zzyzx_quorble") is None

    def test_ordered_by_confidence_and_deduplicated(self, encoder):
        _store_entity(encoder, "abraham_lincoln")
        out = encoder.resolve_query_entities("Abraham Lincoln and Abraham Lincoln")
        ids = [e["entity_id"] for e in out]
        assert len(ids) == len(set(ids))
        confs = [e["confidence"] for e in out]
        assert confs == sorted(confs, reverse=True)


class TestResolveExact:
    def test_scoped_key_first(self, encoder, monkeypatch):
        calls = []

        def fake(surface, entity_type=None):
            calls.append((surface, entity_type))
            return "scoped" if entity_type == "person" else "unscoped"
        monkeypatch.setattr(encoder.store, "resolve_surface_form", fake)
        assert encoder._resolve_exact("x_y", "x_y", "person") == "scoped"
        assert calls == [("x_y", "person")]

    def test_falls_back_to_unscoped_then_normalized(self, encoder, monkeypatch):
        calls = []

        def fake(surface, entity_type=None):
            calls.append((surface, entity_type))
            return "hit" if (surface, entity_type) == ("polydor_records", None) else None
        monkeypatch.setattr(encoder.store, "resolve_surface_form", fake)
        assert encoder._resolve_exact("polydor_record", "polydor_records",
                                      "compound_noun") == "hit"
        assert calls == [("polydor_record", "compound_noun"), ("polydor_record", None),
                         ("polydor_records", "compound_noun"), ("polydor_records", None)]

    def test_miss_returns_none(self, encoder):
        assert encoder._resolve_exact("nope", "nope", "person") is None


class TestSpanScan:
    def test_text_spans_longest_first_trimmed_at_stop_words(self):
        spans = EntityEncoder._text_spans("What was the Apple II?")
        assert "apple_ii" in spans
        assert not any(sp.startswith(("the_", "what_", "was_")) for sp in spans)
        assert not any(sp.endswith("_the") for sp in spans)
        assert all("_" in sp for sp in spans)  # never single words

    def test_adjective_edges_allowed(self):
        assert "alexander_the_great" in EntityEncoder._text_spans("Alexander the Great")
        assert "new_york" in EntityEncoder._text_spans("trains in New York")
        assert "blood_on_the_tracks" in EntityEncoder._text_spans(
            "What label released Blood on the Tracks?")

    def test_text_spans_cap_and_empty(self):
        assert EntityEncoder._text_spans("") == []
        assert EntityEncoder._text_spans("anarchism") == []
        long = " ".join(f"w{i}x" for i in range(10))
        assert max(len(sp.split("_")) for sp in EntityEncoder._text_spans(long)) == \
            EntityEncoder._MAX_SPAN_TOKENS

    def test_stored_entity_inside_unresolved_chunk(self, encoder, monkeypatch):
        """'Polydor Records artists' is one noun chunk; polydor_records is stored."""
        from src.encoders.registry import EncodingResult, SemanticAddress
        encoder.store.create_entity(entity_id="63aa55edcc8d",
                                    surface_forms=["polydor_records"],
                                    metadata={"entity_type": "compound_noun"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: EncodingResult(
            head_type="entity", metadata={}, addresses=[SemanticAddress(
                head_type="entity", concept="1c1fd9d8a429", confidence=0.2,
                metadata={"normalized": "polydor_records_artists", "unresolved": True,
                          "entity_type": "compound_noun"})]))
        out = encoder.resolve_query_entities("Polydor Records artists")
        assert [(e["entity_id"], e["normalized"], e["entity_type"]) for e in out] == [
            ("63aa55edcc8d", "polydor_records", "compound_noun")]

    def test_apple_ii_beats_bare_numeral(self, encoder, monkeypatch):
        """Live 2026-10-04: the head proposed only 'ii' (stored, typed person)
        for 'What was the Apple II?' and graph was seeded from it."""
        from src.encoders.registry import EncodingResult, SemanticAddress
        encoder.store.create_entity(entity_id="5d7f49449ab2", surface_forms=["ii"],
                                    metadata={"entity_type": "person"})
        encoder.store.create_entity(entity_id="1f85c8cb83ce", surface_forms=["apple_ii"],
                                    metadata={"entity_type": "person"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: EncodingResult(
            head_type="entity", metadata={}, addresses=[SemanticAddress(
                head_type="entity", concept="5d7f49449ab2", confidence=0.45,
                metadata={"normalized": "ii", "surface_form": "II", "entity_type": "person"})]))
        out = encoder.resolve_query_entities("What was the Apple II?")
        assert [e["entity_id"] for e in out] == ["1f85c8cb83ce"]

    def test_longer_entity_ranks_first(self, encoder, monkeypatch):
        from src.encoders.registry import EncodingResult, SemanticAddress
        encoder.store.create_entity(entity_id="ab", surface_forms=["abraham_lincoln"],
                                    metadata={"entity_type": "person"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: EncodingResult(
            head_type="entity", metadata={}, addresses=[SemanticAddress(
                head_type="entity", concept="fo", confidence=0.9,
                metadata={"normalized": "theatre", "surface_form": "Theatre",
                          "entity_type": "compound_noun"})]))
        out = encoder.resolve_query_entities("abraham lincoln theatre")
        assert [e["entity_id"] for e in out] == ["ab", "fo"]

    def test_single_word_spans_never_tried(self, encoder, monkeypatch):
        from src.encoders.registry import EncodingResult, SemanticAddress
        encoder.store.create_entity(entity_id="art", surface_forms=["artists"],
                                    metadata={"entity_type": "compound_noun"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: EncodingResult(
            head_type="entity", metadata={}, addresses=[SemanticAddress(
                head_type="entity", concept="x", confidence=0.2,
                metadata={"normalized": "polydor_artists", "unresolved": True})]))
        assert encoder.resolve_query_entities("polydor artists") == []

    def test_span_scan_is_one_batch(self, encoder, monkeypatch):
        calls = []
        orig = encoder.store.resolve_surface_forms_batch
        monkeypatch.setattr(encoder.store, "resolve_surface_forms_batch",
                            lambda q: calls.append(q) or orig(q))
        encoder.resolve_query_entities("the battle of hastings in england")
        assert len(calls) == 1


class TestWeakMentions:
    @pytest.mark.parametrize("norm,surface", [
        ("ii", "II"), ("iv", "IV"), ("xiv", "XIV"), ("1984", "1984"),
        ("a", "a"), ("of", "of"), ("ab", "Ab"),
        ("tv", "TV"), ("st", "St"), ("dr", "Dr"), ("un", "un")])  # temporal_filter.is_fragment
    def test_weak(self, norm, surface):
        assert EntityEncoder._is_weak_mention(norm, surface)

    @pytest.mark.parametrize("norm,surface", [
        ("uk", "UK"), ("eu", "EU"), ("apple_ii", "Apple II"), ("linnaeus", "Linnaeus"),
        ("civil", "civil"), ("mild", "mild"), ("did", "did"), ("dim", "dim"),
        ("u2", "U2"), ("f1", "F1"),   # letter-digit names ingest keeps (temporal_filter)
        ("un", "UN"), ("dc", "DC"), ("md", "MD"), ("gm", "GM")])  # capitals, not numerals
    def test_not_weak(self, norm, surface):
        assert not EntityEncoder._is_weak_mention(norm, surface)

    def test_empty_is_not_a_numeral(self):
        assert EntityEncoder._is_weak_mention("", "")  # empty: <= 2 chars, no acronym

    def test_contains_run(self):
        assert EntityEncoder._contains_run(["apple", "ii"], ["ii"])
        assert not EntityEncoder._contains_run(["apple", "ii"], ["apple", "iii"])


def _fake_head_result(*mentions):
    """mentions: (entity_id, normalized, surface, entity_type, confidence)"""
    from src.encoders.registry import EncodingResult, SemanticAddress
    return EncodingResult(head_type="entity", metadata={}, addresses=[
        SemanticAddress(head_type="entity", concept=eid, confidence=conf,
                        metadata={"normalized": n, "surface_form": sf, "entity_type": t})
        for eid, n, sf, t, conf in mentions])


class TestSpecificBeatsGeneric:
    """Demo run 3, live 2026-10-04: generic nouns seeded the graph."""

    def test_is_specific(self):
        sp = EntityEncoder._is_specific
        assert sp({"normalized": "blood_on_the_tracks", "surface_form": "blood on the tracks",
                   "entity_type": "person"})
        assert sp({"normalized": "sidon", "surface_form": "Sidon", "entity_type": "person"})
        assert sp({"normalized": "uk", "surface_form": "UK", "entity_type": "person"})
        assert not sp({"normalized": "label", "surface_form": "label",
                       "entity_type": "compound_noun"})
        assert not sp({"normalized": "blood", "surface_form": "Blood",
                       "entity_type": "rejection_generic_noun"})
        assert not sp({"normalized": "x", "surface_form": "", "entity_type": ""})

    def test_blood_on_the_tracks_drops_label(self, encoder, monkeypatch):
        encoder.store.create_entity(entity_id="6617162d825a",
                                    surface_forms=["blood_on_the_tracks"],
                                    metadata={"entity_type": "person"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: _fake_head_result(
            ("1aca80e8b55c", "label", "label", "compound_noun", 0.25),
            ("3e8e40c0af5f", "blood", "Blood", "rejection_generic_noun", 0.25)))
        out = encoder.resolve_query_entities("What label released Blood on the Tracks?")
        assert [e["entity_id"] for e in out] == ["6617162d825a"]

    def test_alexander_the_great_and_sidon(self, encoder, monkeypatch):
        encoder.store.create_entity(entity_id="28c34e555703",
                                    surface_forms=["alexander_the_great"],
                                    metadata={"entity_type": "person"})
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: _fake_head_result(
            ("9f2a08fde416", "alexander", "Alexander", "person", 0.45),
            ("826035a0e120", "sidon", "Sidon", "person", 0.45),
            ("e81dfe69841a", "king", "king", "quality", 0.25)))
        out = encoder.resolve_query_entities("Who did Alexander the Great make king of Sidon?")
        assert [e["normalized"] for e in out] == ["alexander_the_great", "sidon"]

    def test_three_names_all_kept_generics_dropped(self, encoder, monkeypatch):
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: _fake_head_result(
            ("1a", "lincoln", "Lincoln", "person", 0.45),
            ("50", "garfield", "Garfield", "person", 0.45),
            ("96", "mckinley", "McKinley", "person", 0.45),
            ("vb", "book", "book", "rejection_generic_noun", 0.25),
            ("47", "assassinations", "assassinations", "rejection_generic_noun", 0.25)))
        out = encoder.resolve_query_entities(
            "Which book is about the assassinations of Lincoln, Garfield and McKinley?")
        assert [e["normalized"] for e in out] == ["lincoln", "garfield", "mckinley"]

    def test_only_generics_are_kept_when_nothing_specific(self, encoder, monkeypatch):
        monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: _fake_head_result(
            ("g1", "geology", "geology", "compound_noun", 0.4)))
        out = encoder.resolve_query_entities("volcanic geology")
        assert [e["normalized"] for e in out] == ["geology"]


def test_question_word_chunk_dropped(encoder, monkeypatch):
    """Ingest stored 'what_label'; resolving the original question finds it."""
    encoder.store.create_entity(entity_id="6617162d825a", surface_forms=["blood_on_the_tracks"],
                                metadata={"entity_type": "person"})
    monkeypatch.setattr(encoder, "encode_query", lambda t, context=None: _fake_head_result(
        ("ad9690717d21", "what_label", "What label", "compound_noun", 0.4),
        ("x1", "the_beatles", "the Beatles", "compound_noun", 0.4)))
    out = encoder.resolve_query_entities("What label released Blood on the Tracks?")
    assert [e["normalized"] for e in out] == ["blood_on_the_tracks", "the_beatles"]
