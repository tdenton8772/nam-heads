"""The record title is the subject: one name candidate, read from ctx["title"].

run_head puts the record's title into every head's context (nats_worker,
commit a5ce7f47). The head no longer guesses a title from the text: the old
regex read only ASCII titles of <= 6 words without ( , . & -- and took any
short first sentence ("Armadillos are mammals.") for a title.
"""

import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import hashlib

import pytest

from ..head import (
    EntityEncoder, _ends_capitalised, _record_title, _title_prefix_tokens, _with_title,
)
from ..normalization import EntityCandidate, extract_candidates, normalize_mention
from ..storage.factory import create_entity_store


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    return get_nlp_service(backend="rule")


@pytest.fixture
def encoder():
    enc = EntityEncoder.__new__(EntityEncoder)
    enc.head_type = "entity"
    enc.config = {}
    enc.enable_diagnostics = False
    enc.max_entities_per_record = 25
    enc.create_new_entities = True
    enc.min_confidence = 0.20
    enc.enable_fuzzy = False
    enc.fuzzy_max_candidates = 50
    enc._rule_nlp = None
    enc.store = create_entity_store(backend="memory", fallback="memory")
    return enc


def _norms(result, meaningful_only=True):
    return {(a.metadata or {}).get("normalized") for a in result.addresses
            if not meaningful_only or (a.metadata or {}).get("meaningful", True)}


def _ingest(encoder, nlp, title, payload):
    """The production form: the adapter's "<title>. " prefix + ctx title."""
    text = f"{title}. {payload}" if title else payload
    return encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text), "title": title}])[0]


# ---------------------------------------------------------------------------
# _record_title: the name the title gives its subject
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,name", [
    ("Nial", "Nial"),
    ("Yves Rocher (company)", "Yves Rocher"),
    ("Cobalt 60 (band)", "Cobalt 60"),
    ("Barnet and Camden (London Assembly constituency)", "Barnet and Camden"),
    ("Santa Margarita, California", "Santa Margarita"),
    ("Francesco II d'Este, Duke of Modena", "Francesco II d'Este"),
    ("Crosby, Stills, Nash & Young", "Crosby, Stills, Nash & Young"),   # 2+ commas: whole
    ("Juno Award for International Entertainer of the Year",
     "Juno Award for International Entertainer of the Year"),           # > 6 words
    ("Just Glü It", "Just Glü It"),                                      # non-ASCII
    ("Great Wyrley F.C.", "Great Wyrley F.C."),
    ("", ""), (None, ""), ("  ", ""),
])
def test_record_title(title, name):
    assert _record_title(title) == name


def test_title_normalizes_like_the_text():
    assert normalize_mention(_record_title("Just Glü It")) == "just_glu_it"
    assert normalize_mention(_record_title("Great Wyrley F.C.")) == "great_wyrley_fc"
    assert normalize_mention(_record_title("Yves Rocher (company)")) == "yves_rocher"


# ---------------------------------------------------------------------------
# _title_prefix_tokens: the adapter's "<title>. " prefix
# ---------------------------------------------------------------------------

def test_prefix_tokens_counted(nlp):
    doc = nlp.parse("Battle of Lostwithiel. The battle was fought in 1644.")
    assert _title_prefix_tokens(doc, "Battle of Lostwithiel") == 4     # 3 words + "."
    assert [t.text for t in list(doc)[:4]] == ["Battle", "of", "Lostwithiel", "."]


def test_prefix_tokens_with_disambiguator_and_abbreviation(nlp):
    doc = nlp.parse("Yves Rocher (company). Yves Rocher is a French company.")
    k = _title_prefix_tokens(doc, "Yves Rocher (company)")
    assert k > 0 and list(doc)[k].text == "Yves"
    doc = nlp.parse("Great Wyrley F.C.. Great Wyrley F.C. is a football club.")
    k = _title_prefix_tokens(doc, "Great Wyrley F.C.")
    assert k > 0 and list(doc)[k].text == "Great"


def test_no_prefix_when_the_payload_leads_with_the_title(nlp):
    # The adapter does not prepend when the payload already starts with the title.
    doc = nlp.parse("Anarchism is a political philosophy.")
    assert _title_prefix_tokens(doc, "Anarchism") == 0
    doc = nlp.parse("Yves Rocher is a French company.")
    assert _title_prefix_tokens(doc, "Yves Rocher (company)") == 0
    assert _title_prefix_tokens(doc, "") == 0
    assert _title_prefix_tokens(None, "Yves Rocher") == 0


def test_prefix_candidates_are_skipped(nlp):
    text = "Politics of Croatia. Croatia is a parliamentary republic."
    doc = nlp.parse(text)
    skip = _title_prefix_tokens(doc, "Politics of Croatia")
    norms = [c.normalized for c in extract_candidates(text, spacy_doc=doc, skip_tokens=skip)]
    assert "politics" not in norms                    # a cut-off piece of the title
    assert "croatia" in norms                         # the body still speaks


# ---------------------------------------------------------------------------
# _with_title: promote or insert, always in front, always a name
# ---------------------------------------------------------------------------

def test_title_promotes_a_matching_mention_and_keeps_its_surface():
    body = [EntityCandidate(text="career", normalized="career", confidence=0.25),
            EntityCandidate(text="capital punishment", normalized="capital_punishment",
                            confidence=0.4, pos_pattern="NOUN_NOUN", token_count=2, start=7)]
    out = _with_title(body, "Capital punishment")
    assert [c.normalized for c in out] == ["capital_punishment", "career"]
    t = out[0]
    assert t.text == "capital punishment"         # matches the parse token by token
    assert t.is_name and t.is_propn and t.confidence == 0.6 and t.start == 7
    assert len(out) == 2


def test_title_inserted_when_the_text_never_names_it():
    out = _with_title([EntityCandidate(text="programming language",
                                       normalized="programming_language", confidence=0.4)], "Nial")
    assert out[0].text == "Nial" and out[0].normalized == "nial" and out[0].is_name
    assert out[0].pos_pattern == "PROPN" and out[0].confidence == 0.6


def test_no_title_no_change():
    body = [EntityCandidate(text="x y", normalized="x_y", confidence=0.4)]
    assert _with_title(body, "") == body and _with_title(body, None) == body


def test_ends_capitalised():
    assert _ends_capitalised("Angolan Armed Forces")
    assert _ends_capitalised("Apollo 15")
    assert _ends_capitalised("Great Wyrley F . C .")
    assert not _ends_capitalised("Board games")
    assert not _ends_capitalised("capital punishment")


# ---------------------------------------------------------------------------
# Through the head
# ---------------------------------------------------------------------------

def test_titles_the_regex_could_not_read_are_subjects(encoder, nlp):
    for title, payload, want in [
        ("Juno Award for International Entertainer of the Year",
         "The award honours touring acts.", "juno_award_for_international_entertainer_of_the_year"),
        ("Just Glü It", "The album was released in 1985.", "just_glu_it"),
        ("Cobalt 60 (band)", "The band formed in Paris.", "cobalt_60"),
        ("Barnet and Camden (London Assembly constituency)", "It elects one member.", "barnet_and_camden"),
    ]:
        assert want in _norms(_ingest(encoder, nlp, title, payload)), title


def test_a_short_first_sentence_is_not_a_title(encoder, nlp):
    text = "Armadillos are mammals. They dig burrows in South America."
    res = encoder.encode_batch_texts([text], [{"spacy_doc": nlp.parse(text)}])[0]
    assert "armadillos_are_mammals" not in _norms(res, meaningful_only=False)


def test_title_wins_over_a_codebook_rejection(encoder, nlp):
    """A common-noun title ("Allele") is what the record is about: emitted
    and meaningful whatever the codebook says about the bare noun."""
    res = _ingest(encoder, nlp, "Allele", "An allele is a variant of a gene.")
    assert "allele" in _norms(res)


def test_title_name_keeps_its_plural_form(encoder, nlp):
    res = _ingest(encoder, nlp, "Angolan Armed Forces",
                  "The Angolan Armed Forces were formed in 1991.")
    by_norm = {(a.metadata or {})["normalized"]: a.concept for a in res.addresses}
    assert by_norm["angolan_armed_forces"] == hashlib.sha256(
        b"angolan_armed_forces").hexdigest()[:12]


def test_batch_and_single_paths_agree_with_a_title(encoder, nlp):
    title = "Battle of Lostwithiel"
    text = f"{title}. The Battle of Lostwithiel was fought in Cornwall during the First English Civil War."
    ctx = {"spacy_doc": nlp.parse(text), "title": title}
    batch = encoder.encode_batch_texts([text], [ctx])[0]
    single = encoder._encode_internal(text, mode="ingest", context=ctx)
    assert [a.concept for a in batch.addresses] == [a.concept for a in single.addresses]
    assert batch.addresses[0].metadata["normalized"] == "battle_of_lostwithiel"
    assert "battle" not in _norms(batch, meaningful_only=False)   # title sentence not re-read


# ---------------------------------------------------------------------------
# Codebook and hashing (unchanged behaviour)
# ---------------------------------------------------------------------------

class TestTitleSurvivesCodebook:
    """Title-injected candidates must survive codebook rejection."""

    def test_anarchism_title_not_rejected(self):
        from nam_heads.impls.entity.default.codebook.lookup import get_codebook
        cb = get_codebook()
        etype = cb.classify(
            "anarchism", "PROPN", "Anarchism", 1, True, False, False, 0.6, 0.0,
        )
        assert not cb.is_rejection_type(etype)


class TestQueryPrefixStripping:
    """Query prefix stripping exposes entity mentions to NLP."""

    @pytest.mark.parametrize("q,bare", [
        ("tell me about Nial", "Nial"),
        ("what is Manufacturing Consent", "Manufacturing Consent"),
        ("who is Albert Einstein", "Albert Einstein"),
        ("Manufacturing Consent", "Manufacturing Consent"),
        ("show me Liquid-crystal display", "Liquid-crystal display"),
        ("search for Nial", "Nial"),
        ("Tell Me About Nial", "Nial"),
    ])
    def test_prefix(self, q, bare):
        assert EntityEncoder._QUERY_PREFIXES.sub("", q) == bare


class TestTitleHashConsistency:
    """The title's id is the id a query for the same name computes."""

    def test_nial_hash_matches(self):
        ingest_hash = hashlib.sha256(normalize_mention("Nial").encode("utf-8")).hexdigest()[:12]
        assert ingest_hash == "89827f61d6e2"

    def test_strom_thurmond_hash_matches(self):
        norm = normalize_mention("Strom Thurmond")
        assert norm == "strom_thurmond"
        assert hashlib.sha256(norm.encode("utf-8")).hexdigest()[:12] == "578b261ef1af"
