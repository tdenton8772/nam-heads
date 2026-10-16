"""What text of a parsed record the gaius head embeds (rule-backend parse)."""

from nam_heads.impls.ontology.gaius.text import (
    defining_phrase, detokenize, is_real, record_text, title_of)


def _sents(nlp, text):
    return [list(s) for s in nlp.parse(text).sents]


def test_detokenize_rejoins_punctuation():
    assert detokenize(["Paris", "is", "a", "city", "(", "France", ")", "."]) == "Paris is a city (France)."
    assert detokenize(["the", "state", "'s", "capital", ","]) == "the state's capital,"


def test_title_fragment_is_recognised_and_spaced(nlp):
    s = _sents(nlp, "Albert_Einstein. Albert Einstein was a physicist.")
    assert title_of(s[0]) == "Albert Einstein"
    assert title_of(s[1]) is None


def test_pointer_and_fragment_sentences_are_not_real(nlp):
    assert not is_real(_sents(nlp, "Mercury may refer to a planet.")[0])
    assert not is_real(_sents(nlp, "See also the list below.")[0])
    assert not is_real(_sents(nlp, "Hi there.")[0])
    assert is_real(_sents(nlp, "Mercury is the smallest planet.")[0])


def test_copular_phrase_cut_after_its_last_noun(nlp):
    ph = defining_phrase(_sents(nlp, "Albert Einstein was a German-born theoretical physicist "
                                     "who is widely held to be great.")[0])
    assert ph.text == "a German-born theoretical physicist"
    assert ph.head == "physicist"


def test_light_noun_hands_the_head_to_its_complement(nlp):
    ph = defining_phrase(_sents(nlp, "Agrotis ipsilon is a species of moth in the family Noctuidae.")[0])
    assert ph.text == "a species of moth"
    assert ph.head == "moth"
    ph = defining_phrase(_sents(nlp, "Polydor is a list of artists.")[0])
    assert ph.head == "list"                     # not a light noun: keeps its head


def test_subject_phrase_when_there_is_no_copula(nlp):
    ph = defining_phrase(_sents(nlp, "The Battle of Waterloo was fought on Sunday 18 June 1815.")[0])
    assert ph.text == "The Battle of Waterloo"
    assert ph.head == "Battle"


def test_no_phrase_without_a_noun(nlp):
    assert defining_phrase(_sents(nlp, "He ran quickly away.")[0]) is None


def test_record_query_prefers_the_lead_phrase(nlp):
    rt = record_text(nlp.parse("Albert_Einstein. Albert Einstein was a German-born theoretical physicist. "
                               "He received the Nobel Prize."))
    assert rt.title == "Albert Einstein"
    assert [i for i, _ in rt.sentences] == [1, 2]
    assert rt.query_text() == "a German-born theoretical physicist"
    assert rt.phrases[2] is None or rt.phrases[2].text


def test_record_query_falls_back_to_title_and_lead(nlp):
    rt = record_text(nlp.parse("Waterloo_Station. He ran quickly to it every day."))
    assert rt.query_text() == "Waterloo Station. He ran quickly to it every day."


def test_record_without_text_has_no_query(nlp):
    rt = record_text(nlp.parse("Hi."))
    assert rt.query_text() == ""
    assert rt.lead is None
