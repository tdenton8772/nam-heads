"""Temporal values and 1-2 character fragments are never entities.

Owner rule: dates, years, decades and centuries are temporal context, never
entities; "st"-style fragments are dropped (temporal_filter.py). The same
predicate runs on ingest (batch and single path) and on the query path, and
the context head still carries the temporal value.
"""

import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from ..head import EntityEncoder, _hash_entity_id
from ..normalization import extract_candidates, normalize_mention
from ..storage.factory import create_entity_store
from ..temporal_filter import (
    is_capital_acronym, is_designation, is_non_entity_mention, is_temporal_value)


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    svc = get_nlp_service(backend="rule")
    assert svc.backend_name == "rule"
    return svc


@pytest.fixture
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.15
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


def _drop(text, labels=None):
    return is_non_entity_mention(text, normalize_mention(text),
                                 frozenset(labels) if labels is not None else None)


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------

JUNK = [
    # years, decades, ranges -- by form, whatever the NER says
    ("1980", {"DATE"}), ("1980", None), ("1980s", {"DATE"}), ("1980's", {"DATE"}),
    ("the 1980s", None),                       # query full-text candidate form
    ("80s", {"O"}), ("early 2000s", {"DATE", "O"}), ("late 1990s", {"DATE", "O"}),
    ("mid-1980s", {"O"}), ("1914-1918", {"O"}), ("2004-05", {"O"}),
    # centuries and millennia, including the bare word the owner named
    ("19th century", {"DATE"}), ("19th-century", {"O"}), ("mid-20th century", {"O"}),
    ("nineteenth century", {"O"}), ("1st century BC", {"DATE", "PERSON"}),
    ("2nd millennium BCE", {"O"}),
    ("century", {"DATE"}), ("century", {"O"}), ("centuries", {"O"}), ("half century", {"O"}),
    # era markers and era years
    ("AD", {"O"}), ("BC", {"PERSON"}), ("BCE", None), ("800 AD", {"O"}), ("44 BC", None),
    # dates: by form, and anything wholly inside a rule-NLP DATE/TIME span
    ("12 March 1985", {"DATE"}), ("March 1985", None), ("March 15", None),
    ("March", {"DATE"}), ("Monday", {"DATE"}), ("3:00", {"TIME"}),
    # 1-2 character fragments; the rule NER calls a lone PROPN PERSON by default
    ("St", {"PERSON"}), ("st", None), ("Dr", {"PERSON"}), ("Jr", {"PERSON"}),
    ("Li", {"PERSON"}), ("Wu", {"PERSON"}), ("Co", {"O"}), ("re", {"O"}), ("10", {"O"}),
    ("un", {"O"}), ("dc", None), ("e", {"O"}), ("C", {"PERSON"}),
    # two capitals on the stoplist: Roman numerals II-XX, era markers, TV
    ("II", {"PERSON"}), ("IV", {"PERSON"}), ("VI", None), ("IX", {"O"}), ("XI", {"PERSON"}),
    ("XV", {"PERSON"}), ("XX", None), ("TV", {"O"}), ("CE", {"PERSON"}),
    ("ΑΝΤΙΟΧΕΩΝ", {"PERSON"}),                  # normalizes to "" -- one hash for all
    ("ΑΒ", {"PERSON"}),                         # two non-ASCII capitals: also ""
]

KEEP = [
    # real names that contain a number
    ("Apollo 11", {"PERSON", "O"}), ("Boeing 747", {"ORG", "O"}), ("Louis XIV", {"PERSON"}),
    ("Mariner 10", {"O"}), ("1990 FIFA World Cup", {"DATE", "ORG", "O"}),
    ("World War II", {"O", "PERSON"}), ("San Francisco 49ers", {"GPE", "O"}),
    ("COVID-19 pandemic", {"O", "PERSON"}),
    # 2-character names: two capitals as written, a designation, or a gazetteer type
    ("UN", {"O"}), ("NY", {"PERSON"}), ("DC", {"O"}), ("D.C.", None), ("GM", {"O"}),
    ("LA", {"PERSON"}), ("IL", None), ("MA", {"PERSON"}), ("MD", {"O"}), ("MI", None),
    ("U2", {"PERSON"}), ("U2", None), ("F1", {"O"}), ("K-2", {"PERSON"}),
    ("UK", {"GPE"}), ("US", {"GPE"}), ("EU", {"GPE"}), ("BP", {"ORG"}),
    ("uk", {"GPE"}),                             # gazetteer type, any case
    # a temporal word inside a larger name
    ("19th-century England", {"O", "GPE"}), ("1960s America", {"DATE", "PERSON"}),
    ("Century Media", {"O"}), ("Sun Yat-sen", {"DATE", "PERSON"}),
    # rule-NER DATE false positives
    ("Sun", {"DATE"}), ("Jan", {"DATE"}), ("march", {"DATE"}),
    # not in the rule: bare numbers of 3 digits, bare ordinals, years as a word
    ("747", {"O"}), ("5th", {"O"}), ("years", {"O"}), ("decades", {"O"}),
]


@pytest.mark.parametrize("text,labels", JUNK)
def test_junk_is_not_an_entity(text, labels):
    assert _drop(text, labels), text


@pytest.mark.parametrize("text,labels", KEEP)
def test_real_entity_is_kept(text, labels):
    assert not _drop(text, labels), text


def test_temporal_value_form_only():
    for v in ("1980", "1980s", "the_1980s", "19th_century", "century", "late_1990s",
              "march_1985", "15_march_1985", "44_bc", "ad_800"):
        assert is_temporal_value(v), v
    for v in ("", "march", "apollo_11", "1990_fifa_world_cup", "19th_century_england",
              "century_media", "747", "years"):
        assert not is_temporal_value(v), v


def test_designation():
    assert is_designation("U2") and is_designation("K-2") and is_designation(" F1 ")
    assert not is_designation("2C") and not is_designation("St") and not is_designation("u2")


def test_capital_acronym():
    for s in ("UN", "NY", "DC", "D.C.", "GM", "MD", "U.N."):
        assert is_capital_acronym(s), s
    for s in ("II", "IV", "XX", "TV", "AD", "BC", "St", "Dr", "Li", "un", "U", "USA", "ΑΒ", "U2"):
        assert not is_capital_acronym(s), s


# ---------------------------------------------------------------------------
# Candidate extraction carries the NER labels the predicate reads
# ---------------------------------------------------------------------------

def test_candidates_carry_ner_labels(nlp):
    text = "The band grew popular in the 1980s and toured the UK. World War II ended."
    by_norm = {c.normalized: c for c in extract_candidates(text, spacy_doc=nlp.parse(text))}
    assert by_norm["1980s"].ner_labels == frozenset({"DATE"})   # "the" is not content
    assert by_norm["uk"].ner_labels == frozenset({"GPE"})
    # "World War II" is one name span, labelled over all its tokens; its
    # capitalised prefix means no separate "II" tail candidate.
    assert by_norm["world_war_ii"].ner_labels == frozenset({"O", "PERSON"})
    assert "ii" not in by_norm
    assert by_norm["band"].ner_labels == frozenset({"O"})


def test_regex_fallback_candidates_have_no_labels():
    cands = extract_candidates("Ada Lovelace wrote in 1843", spacy_doc=None)
    assert cands and all(c.ner_labels is None for c in cands)


# ---------------------------------------------------------------------------
# Through the head: ingest (batch + single), query, parity
# ---------------------------------------------------------------------------

TEXT = ("The band grew popular in the 1980s and toured the UK. "
        "Notre-Dame Cathedral was restored in the 19th century, and the century ended. "
        "World War II ended. U2 released an album on 12 March 1985. "
        "He lived on St. Mark's Square in the early 2000s.")
JUNK_NORMS = {"1980s", "century", "ii", "st", "march", "early_2000s"}
REAL_NORMS = {"uk", "u2", "world_war_ii", "notre_dame_cathedral"}
# Canonical ids ingest used to write for the junk: "1980s" singularizes to "1980".
JUNK_IDS = {_hash_entity_id("", n) for n in ("1980", "1980s", "century", "ii", "st", "march")}


def _norms(result):
    return {(a.metadata or {}).get("normalized") for a in result.addresses}


def test_ingest_batch_path_drops_temporal_and_fragments(encoder, nlp):
    res = encoder.encode_batch_texts([TEXT], [{"spacy_doc": nlp.parse(TEXT)}])[0]
    norms = _norms(res)
    assert not norms & JUNK_NORMS, norms & JUNK_NORMS
    assert REAL_NORMS <= norms, REAL_NORMS - norms
    assert not {a.concept for a in res.addresses} & JUNK_IDS
    # nothing junk reached the store either
    assert encoder.store.resolve_surface_form("1980s") is None
    assert encoder.store.resolve_surface_form("century") is None


def test_ingest_single_path_matches_batch(encoder, nlp):
    doc = nlp.parse(TEXT)
    single = encoder._encode_internal(TEXT, mode="ingest", context={"spacy_doc": doc})
    batch = encoder.encode_batch_texts([TEXT], [{"spacy_doc": doc}])[0]
    assert _norms(single) == _norms(batch)
    assert not _norms(single) & JUNK_NORMS


def test_title_keeps_its_parse_labels(encoder, nlp):
    """A prepended title replacing a parsed candidate keeps that candidate's
    NER labels: "UK" stays (GPE), "St" still goes."""
    text = "UK. The UK is a country in Europe."
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert "uk" in _norms(res)


def test_query_about_a_decade_looks_for_no_decade_partition(encoder):
    for q in ("the 1980s", "1980s", "the 19th century", "century"):
        res = encoder.encode_query(q)
        assert not _norms(res) & (JUNK_NORMS | {"the_1980s", "19th_century"}), (q, _norms(res))
        assert not {a.concept for a in res.addresses} & JUNK_IDS, q


def test_query_and_ingest_drop_the_same_mentions(encoder, nlp):
    ingest = _norms(encoder.encode_batch_texts([TEXT], [{"spacy_doc": nlp.parse(TEXT)}])[0])
    query = _norms(encoder.encode_query(TEXT))
    assert not query & JUNK_NORMS
    assert REAL_NORMS <= query
    assert (ingest & REAL_NORMS) == (query & REAL_NORMS)


def test_resolver_skips_temporal_spans_a_stale_store_still_holds(encoder):
    for n in ("late_1980s", "19th_century", "abraham_lincoln"):
        encoder.store.create_entity(entity_id=_hash_entity_id("", n), surface_forms=[n],
                                    metadata={"entity_type": "compound"})
    out = encoder.resolve_query_entities("Abraham Lincoln in the late 1980s and the 19th century")
    got = {e["normalized"] for e in out}
    assert "abraham_lincoln" in got
    assert not got & {"late_1980s", "19th_century"}


def test_resolver_keeps_a_designation(encoder):
    eid = _hash_entity_id("", "u2")
    encoder.store.create_entity(entity_id=eid, surface_forms=["u2"],
                                metadata={"entity_type": "identifier"})
    assert eid in [e["entity_id"] for e in encoder.resolve_query_entities("U2")]


ACRONYM_TEXT = ("The UN met in DC. GM sold cars in NY. The TV was off. "
                "He visited St. Mark with Dr. Li and Wu. World War II ended.")
ACRONYMS_KEPT = {"un", "dc", "gm", "ny"}
FRAGMENTS_DROPPED = {"tv", "st", "dr", "li", "wu", "ii"}


def test_ingest_keeps_capital_acronyms_drops_fragments(encoder, nlp):
    doc = nlp.parse(ACRONYM_TEXT)
    for res in (encoder.encode_batch_texts([ACRONYM_TEXT], [{"spacy_doc": doc}])[0],
                encoder._encode_internal(ACRONYM_TEXT, mode="ingest", context={"spacy_doc": doc})):
        norms = _norms(res)
        assert ACRONYMS_KEPT <= norms, ACRONYMS_KEPT - norms
        assert not norms & FRAGMENTS_DROPPED, norms & FRAGMENTS_DROPPED


def test_query_keeps_capital_acronyms_drops_fragments(encoder):
    norms = _norms(encoder.encode_query(ACRONYM_TEXT))
    assert ACRONYMS_KEPT <= norms, ACRONYMS_KEPT - norms
    assert not norms & FRAGMENTS_DROPPED, norms & FRAGMENTS_DROPPED
    for q in ("UN", "GM"):
        assert _norms(encoder.encode_query(q)) == {q.lower()}, q
    for q in ("TV", "II", "St", "Dr", "un"):
        assert not _norms(encoder.encode_query(q)), q


def test_resolver_returns_stored_capital_acronyms(encoder, nlp):
    """Ingest writes them; the resolver (canonicalizer seed) finds them,
    including DC/MD, which the Roman-numeral pattern would have taken."""
    encoder.encode_batch_texts([ACRONYM_TEXT], [{"spacy_doc": nlp.parse(ACRONYM_TEXT)}])
    for q in ("UN", "DC", "GM", "NY"):
        got = [e["normalized"] for e in encoder.resolve_query_entities(f"news from the {q}")]
        assert q.lower() in got, (q, got)
    eid = _hash_entity_id("", "md")
    encoder.store.create_entity(entity_id=eid, surface_forms=["md"],
                                metadata={"entity_type": "identifier"})
    assert eid in [e["entity_id"] for e in encoder.resolve_query_entities("MD")]


def test_fallback_never_picks_a_temporal_token(encoder):
    assert encoder._fallback_entity("1980s 1990s") is None
    assert encoder._fallback_entity("in 1980 st") is None   # "in", "st": fragments
    assert encoder._fallback_entity("1980 widgets") == "widgets"


def test_record_of_only_temporal_mentions_gets_no_temporal_entity(encoder, nlp):
    text = "in the 1980s and the 1990s"
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert not _norms(res) & {"1980s", "1990s"}


# ---------------------------------------------------------------------------
# The context head still carries the temporal value
# ---------------------------------------------------------------------------

def test_context_head_still_gets_the_temporal_value(nlp, monkeypatch):
    from nam_heads.impls.context.default.head import ContextEncoder
    monkeypatch.delenv("NAM_CONTEXT_HEAD", raising=False)
    doc = nlp.parse(TEXT)
    enc = ContextEncoder(head_type="context", config={"context_store_backend": "memory"})
    assert enc.head_mode == "particular"
    got = [a.concept for a in enc.encode_batch_texts([TEXT], [{"spacy_doc": doc}])[0].addresses]
    assert any("D1980" in c for c in got), got           # the 1980s
    assert any(c.startswith("t:C19") for c in got), got  # the 19th century
    assert any("Y1985" in c for c in got), got           # 12 March 1985


def test_legacy_context_head_still_gets_the_temporal_value(nlp):
    from nam_heads.impls.context.default.head import ContextEncoder
    text = "The band grew popular in the 1980s."
    enc = ContextEncoder(head_type="context", config={
        "context_store_backend": "memory", "head": "legacy"})
    got = [a.concept for a in enc.encode(text, {"spacy_doc": nlp.parse(text)}).addresses]
    assert "1980s" in got, got
