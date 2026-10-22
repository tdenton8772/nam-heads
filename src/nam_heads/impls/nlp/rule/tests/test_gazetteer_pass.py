"""NERRecognizer._gazetteer_pass with the merged lookup keeps the per-label
semantics: longest span first, first label in dict order wins, '_' lists are
not labels, first+last name PERSON rule, and a replaced gazetteer is seen."""
import types

from nam_heads.impls.nlp.rule.ner import NERRecognizer


def _ner(gaz):
    n = NERRecognizer.__new__(NERRecognizer)
    n._gazetteers = gaz
    return n


def _toks(*words, pos="PROPN"):
    return [types.SimpleNamespace(text=w, pos=pos, ent_type="", ent_iob="O") for w in words]


def _run(ner, toks):
    claimed = [False] * len(toks)
    ents = []
    ner._gazetteer_pass(toks, claimed, ents)
    return [(e["text"], e["label"]) for e in ents]


def test_longest_span_wins():
    ner = _ner({"GPE": {"new york", "new york city"}, "ORG": {"york"}})
    assert _run(ner, _toks("New", "York", "City", "is", "big")) == [("New York City", "GPE")]


def test_first_label_in_dict_order_wins():
    ner = _ner({"ORG": {"apple"}, "PRODUCT": {"apple"}})
    assert _run(ner, _toks("Apple")) == [("Apple", "ORG")]
    ner2 = _ner({"PRODUCT": {"apple"}, "ORG": {"apple"}})
    assert _run(ner2, _toks("Apple")) == [("Apple", "PRODUCT")]


def test_underscore_lists_are_not_labels():
    ner = _ner({"_first_names": {"john"}, "_last_names": {"smith"}, "GPE": {"paris"}})
    assert _run(ner, _toks("john", "Paris")) == [("Paris", "GPE")]


def test_first_last_name_person_rule():
    ner = _ner({"_first_names": {"john"}, "_last_names": {"smith"}})
    assert _run(ner, _toks("John", "Smith", "ran")) == [("John Smith", "PERSON")]
    assert _run(ner, _toks("John", "Smith", pos="NOUN")) == []   # needs PROPN


def test_claimed_tokens_are_skipped():
    ner = _ner({"GPE": {"paris"}})
    toks = _toks("Paris", "Paris")
    claimed = [True, False]
    ents = []
    ner._gazetteer_pass(toks, claimed, ents)
    assert [(e["text"], e["label"]) for e in ents] == [("Paris", "GPE")]


def test_replaced_gazetteer_is_picked_up():
    ner = _ner({"GPE": {"paris"}})
    assert _run(ner, _toks("Paris")) == [("Paris", "GPE")]
    ner._gazetteers = {"GPE": {"paris"}, "LOC": {"rome"}}
    assert _run(ner, _toks("Rome")) == [("Rome", "LOC")]


def test_multi_token_entry_found_only_via_its_first_word():
    ner = _ner({"ORG": {"united nations"}, "GPE": {"nations"}})
    # "United" is the first word of an entry: the 2-token span is tried first.
    assert _run(ner, _toks("United", "Nations")) == [("United Nations", "ORG")]
    # "The" starts no entry: only single tokens are tried from there.
    assert _run(ner, _toks("The", "Nations")) == [("Nations", "GPE")]


def test_token_with_internal_space_skips_the_shortcut():
    ner = _ner({"GPE": {"new york city"}})
    assert _run(ner, _toks("New York", "City")) == [("New York City", "GPE")]
