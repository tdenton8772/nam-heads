"""The record title as main subject; pronouns, copula variants and kind-noun
anaphors resolving to it; the person / non-person and kind guards.

Runs on the production NLP (rule backend): every case is the parse the
pipeline sees, including its defects (curly "She’s" split in three, a 3sg
verb tagged NOUN, participles tagged VERB).
"""
import os

os.environ.setdefault("NAM_NLP_BACKEND", "rule")

import pytest

from nam_heads.impls.attribute.default import predication as P


@pytest.fixture(scope="module")
def nlp():
    from src.nlp import get_nlp_service
    svc = get_nlp_service()
    assert svc.backend_name == "rule"
    return svc


def run(nlp, text, title="", **kw):
    doc = nlp.parse(text)
    vals = {av.value: av for av in P.attribute_values(doc, title=title, **kw)}
    return doc, vals


def subj(doc, av):
    """The subjects a value is predicated of (bare-modifier readings aside)."""
    return sorted({" ".join(doc[i].text for i in range(s, e))
                   for (s, e), link, _k, _si in av.occurrences if link != "modifier"})


# ------------------------------------------------------------- the title

def test_split_title_strips_the_qualifier():
    assert P.split_title("Yves Rocher (company)") == (["yves", "rocher"], "company")
    assert P.split_title("Mad Love (2001 film)") == (["mad", "love"], "film")
    assert P.split_title("Brigham Young") == (["brigham", "young"], "")
    assert P.split_title("") == ([], "")


def test_title_mentions_ignore_tokenization_and_take_a_plural(nlp):
    doc = nlp.parse("Great Wyrley F.C.. Great Wyrley F.C. is a club.")
    spans = P.title_spans(doc, "Great Wyrley F.C.")
    assert len(spans) == 2 and [doc[i].text for i in range(*spans[0])] == ["Great", "Wyrley", "F", ".", "C"]
    doc = nlp.parse("Air shows are public events.")
    assert P.title_spans(doc, "Air show") == [(0, 2)]
    assert P.title_spans(doc, "Boat show") == []


def test_pronoun_resolves_to_the_title_without_a_copular_subject(nlp):
    text = ("Brigham Young. If you're wondering who Brigham Young was, he was an American "
            "religious leader born in 1801 who became the second president of the LDS Church.")
    doc, vals = run(nlp, text, "Brigham Young")
    for v in ("american", "religious", "leader", "president"):
        assert subj(doc, vals[v]) == ["Brigham Young"], v
    assert vals["leader"].link == "pronoun"
    # without the title the same "he" has nothing to resolve to
    doc, plain = run(nlp, text)
    assert subj(doc, plain["leader"]) == ["he"]


# ------------------------------------------------------------- copula forms

def test_curly_contraction_is_a_copula_after_a_pronoun(nlp):
    doc, vals = run(nlp, "Svetlana Ishmouratova. She’s a Russian biathlete.", "Svetlana Ishmouratova")
    assert subj(doc, vals["russian"]) == ["Svetlana Ishmouratova"]
    assert vals["russian"].link == "pronoun"


@pytest.mark.parametrize("apos", ["'", "’"])
def test_possessive_s_is_not_a_copula(nlp, apos):
    doc, vals = run(nlp, f"Gary{apos}s impact was big.")
    assert "impact" not in vals            # not "Gary is impact"
    assert subj(doc, vals["big"]) == ["impact"]   # the possessor is not the subject


def test_become_and_remain_are_copulas(nlp):
    doc, vals = run(nlp, "Brussels remains the de facto capital of the European Union.")
    assert subj(doc, vals["capital"]) == ["Brussels"]
    assert "de" not in vals and "facto" not in vals
    doc, vals = run(nlp, "Tillie Klimek. She became notorious.", "Tillie Klimek")
    assert subj(doc, vals["notorious"]) == ["Tillie Klimek"]


def test_auxiliaries_and_a_leading_adverb_before_the_copula(nlp):
    doc, vals = run(nlp, "Historically, Rusyns have been farmers.")
    assert subj(doc, vals["farmer"]) == ["Rusyns"]


def test_leading_participial_clause_with_several_commas(nlp):
    text = ("Yves Rocher (company). Founded by entrepreneur Yves Rocher in 1965 in La Gacilly, France, "
            "the company was a mail order business.")
    doc, vals = run(nlp, text, "Yves Rocher (company)")
    assert subj(doc, vals["business"]) == ["Yves Rocher"]       # "the company" -> the title (qualifier)
    assert subj(doc, vals["company"]) == ["Yves Rocher"]


# ------------------------------------------------------------- as-role

def test_as_role_verbs_reflexive_and_aside(nlp):
    doc, vals = run(nlp, "Brussels. It has established itself as the dense part of Belgium.", "Brussels")
    assert subj(doc, vals["dense"]) == ["Brussels"]
    text = ("Brassicaceae. This plant family, commonly known as the mustard or cabbage family, "
            "is vital to agriculture.")
    doc, vals = run(nlp, text, "Brassicaceae")
    for v in ("plant", "mustard", "cabbage"):
        assert subj(doc, vals[v]) == ["Brassicaceae"], v
    assert "vital" not in vals              # "vital to": relational, and evaluative


def test_fronted_as_needs_a_person_head(nlp):
    doc, vals = run(nlp, "As mayor of La Gacilly, Rocher aimed to provide employment.")
    assert subj(doc, vals["mayor"]) == ["Rocher"] and vals["mayor"].link == "fronted_as"
    doc, vals = run(nlp, "As a consequence, Rocher moved.")
    assert "consequence" not in vals


# ------------------------------------------------------------- anaphors

def test_demonstrative_subject_anaphor(nlp):
    text = "RAF East Kirkby. Situated in Lincolnshire, this airfield was a bomber base."
    doc, vals = run(nlp, text, "RAF East Kirkby")
    for v in ("airfield", "bomber", "base"):
        assert subj(doc, vals[v]) == ["RAF East Kirkby"], v
    assert vals["airfield"].link == "anaphor"


def test_misread_verb_after_the_anaphor(nlp):
    doc, vals = run(nlp, "Alligatoridae. This family exemplifies how shifts shaped biodiversity.",
                    "Alligatoridae")
    assert subj(doc, vals["family"]) == ["Alligatoridae"]


def test_demonstrative_object_anaphor(nlp):
    text = "AM 738 4to. Around 1680 a scribe fashioned this unusually tall manuscript."
    doc, vals = run(nlp, text, "AM 738 4to")
    assert subj(doc, vals["manuscript"]) == ["AM 738 4to"]
    assert subj(doc, vals["tall"]) == ["AM 738 4to"]
    assert "unusually" not in vals


def test_definite_anaphor_needs_support_from_the_text_or_the_store(nlp):
    # "the battle": the title names it
    doc, vals = run(nlp, "Battle of Lostwithiel. The battle secured South West England.",
                    "Battle of Lostwithiel")
    assert subj(doc, vals["battle"]) == ["Battle of Lostwithiel"]
    # "the gold shield": nothing ties it to the coat of arms -- not a value of it
    title = "Coat of arms of the Turks and Caicos Islands"
    text = title + ". The gold shield depicts a conch shell."
    doc, vals = run(nlp, text, title)
    assert "shield" not in vals
    # ... unless the store knows "shield" as a kind (learned kind flag)
    doc, vals = run(nlp, text, title, is_kind=lambda v: True if v == "shield" else None)
    assert subj(doc, vals["shield"]) == [title]


def test_anaphor_noun_named_something_else_earlier(nlp):
    text = ("Battle of Lostwithiel. It happened during the First English Civil War. "
            "The war ended in 1651.")
    doc, vals = run(nlp, text, "Battle of Lostwithiel", is_kind=lambda v: True)
    assert "war" not in vals or subj(doc, vals["war"]) != ["Battle of Lostwithiel"]
    text = "Toxicodendron rydbergii. Its trifoliate leaves develop. In autumn these leaves turn yellow."
    doc, vals = run(nlp, text, "Toxicodendron rydbergii")
    assert "leaf" not in vals or subj(doc, vals["leaf"]) != ["Toxicodendron rydbergii"]


def test_abstract_anaphor_is_rejected_by_the_kind_flag(nlp):
    text = "Bodhidharma. This story illustrates how influences shaped Buddhism."
    abstract = lambda v: False if v == "story" else None
    doc, vals = run(nlp, text, "Bodhidharma", is_kind=abstract)
    assert "story" not in vals


def test_person_guard_on_anaphors(nlp):
    # a PERSON-labelled title plus her/his in the text: the main subject is a
    # person, so "the company" is not it
    text = ("Susan Williams-Ellis. Portmeirion Pottery was established by Susan Williams-Ellis. "
            "Her work was admired. The company remains a celebrated name.")
    doc = nlp.parse(text)
    # Precondition on the deterministic rule NER: fail, never skip.
    assert all(P._ner_map(doc).get(k) == "PERSON"
               for k in range(*P.title_spans(doc, "Susan Williams-Ellis")[0])), \
        "rule NER no longer labels the title PERSON"
    doc, vals = run(nlp, text, "Susan Williams-Ellis")
    assert "company" not in vals
    assert subj(doc, vals["celebrated"]) == ["company"]     # the predicate stays on the anaphor


def test_pronoun_guard_uses_the_title_qualifier(nlp):
    text = "Yves Rocher (company). He was a French botanist."
    doc, vals = run(nlp, text, "Yves Rocher (company)")
    assert subj(doc, vals["botanist"]) == ["He"]          # a company is not "he"
    text = "Anita Hill. Anita Hill is an American lawyer. It was a stone bridge."
    doc, vals = run(nlp, text, "Anita Hill")
    assert subj(doc, vals["lawyer"]) == ["Anita Hill"]
    assert subj(doc, vals["bridge"]) == ["It"]            # a person is not "it"


# ------------------------------------------------------------- appositive / title / kinds

def test_wider_appositives(nlp):
    doc, vals = run(nlp, "Zhang Sixun, a Chinese astronomer and mechanical engineer, was born in Bazhong.")
    for v in ("chinese", "astronomer", "mechanical", "engineer"):
        assert subj(doc, vals[v]) == ["Zhang Sixun"], v
    doc, vals = run(nlp, "Acapulco, a major seaport city, is in Mexico.")
    assert subj(doc, vals["seaport"]) == ["Acapulco"] and subj(doc, vals["city"]) == ["Acapulco"]


def test_capitalised_title_words(nlp):
    doc, vals = run(nlp, "In 1644 King Charles marched west. Emperor Ashoka ruled India.")
    assert subj(doc, vals["king"]) == ["Charles"]
    assert subj(doc, vals["emperor"]) == ["Ashoka"]
    doc, vals = run(nlp, "General Motors is big.")
    assert "general" not in vals                          # no PERSON name after it


def test_plural_title_is_shared_by_coordinated_names(nlp):
    doc, vals = run(nlp, "The work of creators Len Wein, Marv Wolfman, and John Warner was popular.")
    assert subj(doc, vals["creator"]) == ["John Warner", "Len Wein", "Marv Wolfman"]


def test_participle_before_a_title_noun(nlp):
    doc, vals = run(nlp, "Gary replaced the injured actor Terrell Davis.")
    assert subj(doc, vals["injured"]) == ["Terrell Davis"] and subj(doc, vals["actor"]) == ["Terrell Davis"]


def test_kind_of(nlp):
    doc, vals = run(nlp, "The army camped around the town of Lostwithiel in 1644.")
    assert subj(doc, vals["town"]) == ["Lostwithiel"]
    doc, vals = run(nlp, "Visitors come to see the ruins of Alinda.")
    assert "ruin" not in vals                             # plural: parts of X, not X
    doc, vals = run(nlp, "He was crowned as the king of France.")
    assert "France" not in [s for av in vals.values() for s in subj(doc, av)]
    doc, vals = run(nlp, "Brussels lies in the heart of Belgium.", is_kind=lambda v: False if v == "heart" else None)
    assert "heart" not in vals


def test_hearst(nlp):
    doc, vals = run(nlp, "The fossil record, including genera like Leidyosuchus, is rich.")
    assert subj(doc, vals["genus"]) == ["Leidyosuchus"]
    doc, vals = run(nlp, "It was shown in modern media like Disney+'s Werewolf by Night.")
    assert "medium" not in vals                           # the possessor is not the instance
    doc, vals = run(nlp, "They formed a coalition including Labour.")
    assert "coalition" not in vals                        # singular: a group X belongs to


# ------------------------------------------------------------- predicate NP details

def test_possessor_inside_a_predicate_np(nlp):
    doc, vals = run(nlp, "The FAA is Angola's national military.")
    assert "national" in vals and "angola" not in vals


def test_participle_opening_a_predicate_np_keeps_its_own_form(nlp):
    doc, vals = run(nlp, "Botswana is a landlocked country.")
    assert {"landlocked", "country"} <= set(vals)
    doc, vals = run(nlp, "The armadillo is an armored mammal.")
    assert {"armored", "mammal"} <= set(vals)
    doc, vals = run(nlp, "Isle of the Ape. Gygax wrote this challenging module.", "Isle of the Ape")
    assert "challenging" in vals and "challenge" not in vals


def test_capitalised_heads_name_unless_they_are_roles(nlp):
    doc, vals = run(nlp, "Afghanistan, known as the Heart of Asia, is a landlocked country.")
    assert "heart" not in vals and "country" in vals
    doc, vals = run(nlp, "Francesco II d'Este was Duke of Modena.")
    assert "duke" in vals


def test_name_internal_capitalised_modifier(nlp):
    doc, vals = run(nlp, "RAF Fiskerton was a Royal Air Force bomber station.")
    assert "royal" not in vals and {"bomber", "station"} <= set(vals)
    doc, vals = run(nlp, "Anacharsis was a Scythian prince.")
    assert "norp" in vals["scythian"].kinds


def test_ing_nouns_keep_their_form(nlp):
    doc, vals = run(nlp, "The valley served as a vital meeting place. It was a long hearing.")
    assert "meeting" in vals and "meet" not in vals
    assert "hearing" in vals and "hear" not in vals


def test_extraction_survives_a_trailing_curly_possessive(nlp):
    doc = nlp.parse("The king’s ’s was here. Gary’")
    P.attribute_values(doc, title="Gary")       # must not raise
