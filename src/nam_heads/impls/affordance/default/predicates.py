"""Predicate extraction for the affordance axis.

The affordance of an entity bundle is the predicate the text links the entity
to: what it did, what was done to it, what it is used for ("born", "found",
"release", "know_as"). This module finds every such predicate in a parsed
document and gives it ONE canonical string, so the same meaning yields the
same value at ingest and at query time.

Value space: the repaired verb lemma, or a curated phrasal value where the
particle changes the meaning ("know_as", "grow_up"), or the copular values
"classify"/"describe". No class taxonomy is applied: values stay
high-cardinality on purpose.

What it repairs (all deterministic):
- Rule-lemmatizer damage: "called" -> "cal", "joined" -> "joine",
  "changed" -> "chang", "became" -> "became". ``normalize_verb`` regenerates
  base candidates from the surface text and keeps the first real verb
  (``data/verb_bases.json``, every WordNet verb lemma -- membership only).
- British -ise / -our spellings fold to -ize / -or (spelling, not meaning).
- Proper-name tokens the tagger calls VERB ("United" in "United States",
  "Marvel") are skipped.
- Aspectual verbs ("began playing", "continued to serve") yield to the verb
  they govern, also across an object ("helps us appreciate", "aimed to
  provide", "led the military to intervene").
- Discourse-frame verbs ("reflects", "highlighting", "if you're wondering")
  are dropped: they describe the passage, not an entity (config
  ``frame_verbs`` always, ``frame_verbs_positional`` in a framing position).
- Verb + object idioms take the idiom's value or none ("shed light",
  "play a role" -> none; "laid the groundwork" -> establish).
- Participles inside a noun phrase ("a landlocked country", "his enduring
  presence") are attributes, not predicates.

Synonym folding is the shared fold (``data/verb_synonyms.json``,
definitions.VERB_SYNONYMS), rebuilt sense-preserving by build_synonyms.py:
two verbs fold only when they share their dominant sense. Light verbs come
from config (``light_verbs``); phrasal values are read before the light-verb
and synonym steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

_DATA = os.path.join(os.path.dirname(__file__), "data")
# The rule lemmatizer's exception table, carried in this head's own data (a copy of the rule parser's
# data/lemma_exceptions.json): a head never reads another plugin's files.
_RULE_EXC_FILE = os.path.join(_DATA, "lemma_exceptions.json")

NULL = "__null__"

BE_FORMS: FrozenSet[str] = frozenset({"be", "am", "is", "are", "was", "were", "been", "being"})

# Irregular past forms the rule lemmatizer's exception table lacks.
IRREGULAR: Dict[str, str] = {
    "became": "become", "underwent": "undergo", "overcame": "overcome",
    "foresaw": "foresee", "withdrew": "withdraw", "forbade": "forbid",
    "outgrew": "outgrow", "undertook": "undertake", "overtook": "overtake",
    "rebuilt": "rebuild", "retook": "retake", "reran": "rerun",
}

def config_light_verbs() -> FrozenSet[str]:
    """The configured light-verb list (affordance config.py light_verbs)."""
    from .config import DEFAULT_CONFIG
    return frozenset(DEFAULT_CONFIG.get("light_verbs", []))


def config_frame_verbs() -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(always, positional) discourse-frame verbs (config.py frame_verbs,
    frame_verbs_positional)."""
    from .config import DEFAULT_CONFIG
    return (frozenset(DEFAULT_CONFIG.get("frame_verbs", [])),
            frozenset(DEFAULT_CONFIG.get("frame_verbs_positional", [])))


def config_emit_copular() -> bool:
    from .config import DEFAULT_CONFIG
    return bool(DEFAULT_CONFIG.get("emit_copular", False))


# Aspectual verbs: when they govern another verb ("began playing",
# "continued to serve") the governed verb carries the meaning.
ASPECTUAL: FrozenSet[str] = frozenset({
    "begin", "start", "continue", "cease", "stop", "finish", "end",
    "resume", "keep", "go", "try", "attempt", "help", "decide", "agree",
    "want", "need", "seek", "fail", "manage", "refuse", "plan",
    "hope", "tend",
})

# Verbs that yield to a verb they govern across a short object ("helps us
# appreciate", "aimed to provide", "allowed researchers to calculate", "led
# the military to intervene"): the governed verb is what happened. "lead"
# yields only across an object ("led to" is the causal phrasal value).
ASPECTUAL_ACROSS_OBJECT: FrozenSet[str] = frozenset({
    "help", "aim", "seek", "allow", "enable", "lead",
})

# Verb + object idioms: (base, object word) -> value, or "" for none. The
# object is found within IDIOM_WINDOW tokens after the verb (or, for
# IDIOM_OBJECT_BEFORE, within 3 tokens before it: "the role language plays").
IDIOMS: Dict[Tuple[str, str], str] = {
    ("shed", "light"): "",
    ("play", "role"): "", ("play", "roles"): "", ("play", "part"): "",
    ("hold", "place"): "",
    ("bear", "fruit"): "",
    ("pave", "way"): "",
    ("mark", "milestone"): "", ("establish", "milestone"): "", ("reach", "milestone"): "",
    ("achieve", "feat"): "",
    ("stand", "testament"): "", ("serve", "testament"): "", ("remain", "testament"): "",
    ("serve", "reminder"): "", ("remain", "reminder"): "",
    ("lay", "groundwork"): "establish", ("lay", "foundation"): "establish",
    ("lay", "foundations"): "establish",
    ("bring", "life"): "create",
}
IDIOM_WINDOW = 6
IDIOM_OBJECT_BEFORE: FrozenSet[Tuple[str, str]] = frozenset({("play", "role"), ("play", "roles")})

# Object-NP starts (a transitive verb's object begins with one of these).
_OBJ_START = frozenset({"DET", "PRON", "NUM", "PROPN", "ADJ"})
_POSSESSIVE = frozenset({"his", "her", "its", "their", "our", "my", "your"})
# Possessives that are never an object pronoun ("helped her win" is a verb).
_POSSESSIVE_ONLY = _POSSESSIVE - {"her"}
# Relative words the tagger marks DET/PRON: the verb after one heads a
# relative clause ("a book that quickly gained acclaim"), it is no modifier.
_RELATIVE = frozenset({"which", "that", "who", "whom", "whose", "what"})
_WH = frozenset({"how", "why", "what", "whether", "when", "where", "which", "who"})

# A framing subject: the reader ("you", "we", "us") or a span of time
# ("April 2 has witnessed", "the 1990s marked").
_READER = frozenset({"you", "we", "us"})
_TIME_NOUNS = frozenset({"century", "centuries", "decade", "decades", "era", "period",
                         "year", "years", "day", "days", "age", "ages", "time", "times"})
_MONTHS = frozenset({"january", "february", "march", "april", "may", "june", "july",
                     "august", "september", "october", "november", "december"})
# A present-tense frame verb whose object is "a/an ...", "more than ..." or a
# wh-clause describes the topic ("represents a fascinating intersection",
# "illustrates how dynasties managed"); "illustrate"/"demonstrate" in the
# present tense always do.
_FRAME_OBJECT = frozenset({"a", "an", "more", "how", "why", "what", "whether"})
_PRESENT_TENSE_FRAME = frozenset({"illustrate", "demonstrate"})

# (base, particle) -> value, where the particle changes the meaning.
PHRASAL: Dict[Tuple[str, str], str] = {
    ("know", "as"): "know_as",
    ("know", "for"): "know_for",
    ("name", "after"): "name_after",
    ("name", "for"): "name_after",
    ("grow", "up"): "grow_up",
    ("take", "part"): "participate",
    ("take", "over"): "take_over",
    ("set", "up"): "set_up",
    ("carry", "out"): "carry_out",
    ("pass", "away"): "die",
    ("break", "out"): "break_out",
    ("break", "up"): "break_up",
    ("make", "up"): "make_up",
    ("step", "down"): "step_down",
    ("give", "up"): "give_up",
    ("bring", "up"): "bring_up",
    ("give", "birth"): "give_birth",
    ("come", "from"): "come_from",
    ("go", "on"): "go_on",
    ("refer", "to"): "refer_to",
    ("lead", "to"): "lead_to",
    ("result", "in"): "result_in",
    ("take", "place"): "take_place",
    ("give", "rise"): "give_rise",
    ("result", "from"): "result_from",
}

COPULAR_NOUN = "classify"
COPULAR_ADJ = "describe"


@lru_cache(maxsize=1)
def verb_bases() -> FrozenSet[str]:
    """The SEED verb list (data/verb_bases.json). At runtime it reaches the
    extractor through the affordance store's seeding path (predicate_store);
    this loader serves the seed builder and SeedLexicon."""
    with open(os.path.join(_DATA, "verb_bases.json")) as f:
        return frozenset(json.load(f))


def _rule_exceptions_path() -> Optional[str]:
    """This head's copy of the rule parser's lemma_exceptions.json (None when it is missing)."""
    return _RULE_EXC_FILE if os.path.isfile(_RULE_EXC_FILE) else None


@lru_cache(maxsize=1)
def _rule_exceptions() -> Dict[str, str]:
    path = _rule_exceptions_path()
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except OSError:
        return {}


class Lexicon:
    """What extraction consults. ``predicate_store.StoreLexicon`` backs it
    with the affordance store (seed + learned entries); ``SeedLexicon`` uses
    the seed data alone (offline tools, tests, a store without docs)."""

    def is_base(self, word: str) -> bool:
        raise NotImplementedError

    def synonym(self, word: str) -> str:
        return word

    def phrasal(self, base: str, particle: str) -> Optional[str]:
        return None

    def note_verb(self, surface: str, tag: str, base: str, known: bool, particle: str) -> None:
        """Observation hook, called once per predicate verb token."""


@lru_cache(maxsize=1)
def seed_synonyms() -> Dict[str, str]:
    """The SEED synonym fold: data/verb_synonyms.json (sense-preserving
    WordNet fold + curated pairs, build_synonyms.py) -- the same table the
    legacy path and the bundler use (definitions.VERB_SYNONYMS)."""
    from .definitions import VERB_SYNONYMS
    return VERB_SYNONYMS


@lru_cache(maxsize=1)
def seed_unfolded() -> FrozenSet[str]:
    """Words an older synonym table folded that the current one keeps as
    themselves (data/verb_synonyms_unfolded.json, build_synonyms.py). The
    store seeds an identity synonym doc for each, replacing the old fold."""
    try:
        with open(os.path.join(_DATA, "verb_synonyms_unfolded.json")) as f:
            return frozenset(json.load(f)) - frozenset(seed_synonyms())
    except OSError:
        return frozenset()


class SeedLexicon(Lexicon):
    def is_base(self, word: str) -> bool:
        return word in verb_bases()

    def synonym(self, word: str) -> str:
        return seed_synonyms().get(word, word)

    def phrasal(self, base: str, particle: str) -> Optional[str]:
        return PHRASAL.get((base, particle))


_SEED = SeedLexicon()


class UnfoldedSeedLexicon(SeedLexicon):
    """The seed lexicon without the synonym fold (NAM_AFFORDANCE_SENSES: the sense table resolves meaning)."""

    def synonym(self, word: str) -> str:
        return word


_SEED_UNFOLDED = UnfoldedSeedLexicon()


def _candidates(t: str) -> List[str]:
    """Base-form candidates for an inflected surface form, most likely first."""
    out = [t]
    if t.endswith("ies") and len(t) > 4:
        out.append(t[:-3] + "y")
    if t.endswith("ied") and len(t) > 4:
        out.append(t[:-3] + "y")
    if t.endswith("ing") and len(t) > 4:
        s = t[:-3]
        out += [s, s + "e"]
        if len(s) > 2 and s[-1] == s[-2]:
            out.append(s[:-1])
        if s.endswith("y"):
            out.append(s[:-1] + "ie")
    if t.endswith("ed") and len(t) > 3:
        out += [t[:-1], t[:-2]]
        s = t[:-2]
        if len(s) > 2 and s[-1] == s[-2]:
            out.append(s[:-1])
    if t.endswith("es") and len(t) > 3:
        out += [t[:-1], t[:-2]]
    elif t.endswith("s") and not t.endswith("ss") and len(t) > 2:
        out.append(t[:-1])
    return out


def surface_candidates(text: str) -> List[str]:
    """Every base form normalize_verb may test for ``text`` (store prefetch)."""
    t = text.lower()
    out = list(_candidates(t))
    if t in IRREGULAR:
        out.append(IRREGULAR[t])
    exc = _rule_exceptions().get(t)
    if exc:
        out.append(exc)
    for c in list(out):
        if c.endswith("ise") or c.endswith("yse"):
            out.append(c[:-2] + "ze")
        if c.endswith("our"):
            out.append(c[:-3] + "or")
    return list(dict.fromkeys(out))


def resolve_base(text: str, rule_lemma: str = "", lexicon: Optional[Lexicon] = None) -> Tuple[str, bool]:
    """(base form, known) for a verb token; known=False means no candidate
    was a known verb and the rule lemma was kept."""
    lex = lexicon or _SEED
    t = text.lower()
    known = True
    if t in IRREGULAR:
        base = IRREGULAR[t]
    else:
        exc = _rule_exceptions().get(t)
        if exc and not t.endswith("s"):
            base = exc                       # born -> bear, led -> lead
        else:
            base = next((c for c in _candidates(t) if lex.is_base(c)), None)
            if base is None:
                base = (rule_lemma or t).lower()
                known = False
    if base.endswith("ise") and lex.is_base(base[:-2] + "ze"):
        base = base[:-2] + "ze"
    if base.endswith("yse") and lex.is_base(base[:-2] + "ze"):
        base = base[:-2] + "ze"
    if base.endswith("our") and lex.is_base(base[:-3] + "or"):
        base = base[:-3] + "or"                # armour -> armor, honour -> honor
    return base, known


def normalize_verb(text: str, rule_lemma: str = "", lexicon: Optional[Lexicon] = None) -> str:
    """Canonical (repaired, synonym-folded) base form of a verb token."""
    lex = lexicon or _SEED
    return lex.synonym(resolve_base(text, rule_lemma, lex)[0])


@dataclass(frozen=True)
class Predicate:
    value: str          # the canonical affordance value
    token_i: int        # token index of the predicate head in the doc
    lemma: str          # repaired base verb (or the copula)
    particle: str       # particle used for a phrasal value, else ""
    passive: bool
    kind: str           # "verb" | "phrasal" | "copular"


def _sentence_bounds(doc) -> List[Tuple[int, int]]:
    sents = getattr(doc, "sents", None)
    out = []
    if sents:
        for s in sents:
            out.append((s.start, s.end))
    if not out:
        out = [(0, len(doc))]
    return out


def _ent_token_set(doc) -> set:
    s = set()
    for ent in getattr(doc, "ents", None) or []:
        for tok in ent:
            s.add(tok.i)
    return s


def _particle(doc, i: int, end: int) -> str:
    """The particle/preposition/object noun right after a verb (one ADV skipped)."""
    j = i + 1
    if j < end and doc[j].pos_ == "ADV" and doc[j].text.lower() not in ("up", "out", "down", "away", "over", "on"):
        j += 1
    if j < end:
        return doc[j].text.lower()
    return ""


def _adjective_after_be(doc, j: int) -> bool:
    """A finite/base VERB tag right after a be-form is not an auxiliary
    construction (those take VBN/VBG): the rule tagger's VERB for an
    adjective ("The pump is red")."""
    if j == 0 or doc[j - 1].text.lower() not in BE_FORMS:
        return False
    if getattr(doc[j], "tag_", "") == "VBG":
        return False
    # The tagger also marks participles VBD ("was built", "was found"): a
    # word that is an inflected form of a known verb is a verb.
    text = doc[j].text
    base, known = resolve_base(text, getattr(doc[j], "lemma_", ""))
    return not known or (base == text.lower() and not text.lower().endswith("ed"))


def _copular_value(doc, i: int, end: int) -> Optional[str]:
    """'is a/an/the ... NOUN' -> classify; 'is ADJ' -> describe; aux use -> None."""
    saw_adj = False
    for j in range(i + 1, min(i + 6, end)):
        p = doc[j].pos_
        if p == "VERB" and _adjective_after_be(doc, j):
            saw_adj = True                   # tagger's VERB for an adjective ("is red")
            continue
        if p in ("VERB", "AUX"):
            return None                      # passive / progressive auxiliary
        if p in ("NOUN", "PROPN"):
            return COPULAR_NOUN
        if p == "ADJ":
            saw_adj = True
            continue
        if p in ("DET", "ADV", "NUM", "PART"):
            continue
        break
    return COPULAR_ADJ if saw_adj else None


DEFAULT_FUNCTION_VERBS: FrozenSet[str] = frozenset({"use", "intend"})


def _object_follows(doc, j: int, end: int) -> bool:
    """Token j starts a noun phrase (what a transitive verb's object begins
    with): a determiner, pronoun, number, proper noun, adjective or
    possessive."""
    if j >= end:
        return False
    t = doc[j]
    return t.pos_ in _OBJ_START or t.text.lower() in _POSSESSIVE


def _plural_noun(w: str) -> bool:
    """A plural common noun by its form ("researchers", "organizations"):
    before a bare verb form it is the verb's subject ("researchers analyze",
    "professionals emphasize vigilance"), where a singular noun makes a
    compound ("the water supply")."""
    w = w.lower()
    return len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is"))


def _bare_verb(tok) -> bool:
    """A lower-case word that is its own verb base ("shape", "oversee"):
    after "help" or an aspectual verb's "to" it is an infinitive even where
    the tagger says NOUN ("helped shape the community", "continue to
    impact crops")."""
    w = tok.text
    return tok.pos_ in ("VERB", "NOUN") and w.islower() and w.isalpha() and w in verb_bases()


def _attributive_participle(doc, i: int, start: int, end: int, tag: str) -> bool:
    """A participle inside a noun phrase, before its noun: "a landlocked
    country", "his enduring presence", "their most celebrated moment". The
    participle directly precedes a common noun and follows a determiner,
    adjective, number or possessive -- or an adverb that itself follows one
    of those or a preposition ("its ethnically mixed population"); a past
    participle may also follow a preposition directly ("to learned
    actions"). After a bare adverb it is a verb ("architects not only
    designed buildings"), after a relative word too ("a book that quickly
    gained acclaim")."""
    low = doc[i].text.lower()
    if not (tag in ("VBN", "VBG") or (tag == "VBD" and low.endswith("ed"))):
        return False
    if i + 1 >= end or doc[i + 1].pos_ != "NOUN" or i == start:
        return False
    prev = doc[i - 1]
    if prev.pos_ == "ADV":
        if i - 2 < start:
            return False
        pp = doc[i - 2]
        if pp.text.lower() in _RELATIVE:
            return False
        if pp.pos_ == "ADP":
            # "with newly built roads" is a modifier; "by dramatically
            # increasing production" is a gerund with its object.
            return tag != "VBG"
        return pp.pos_ in ("DET", "ADJ", "NUM") or pp.text.lower() in _POSSESSIVE
    if prev.text.lower() in _RELATIVE:
        return False
    if prev.pos_ == "ADP":
        # "to learned actions", "by trained soldiers"; a gerund after a
        # preposition takes an object ("for digging burrows").
        return tag != "VBG"
    return prev.pos_ in ("DET", "ADJ", "NUM") or prev.text.lower() in _POSSESSIVE


def _idiom_value(doc, i: int, start: int, end: int, base: str) -> Optional[str]:
    """The idiom value for verb ``base`` at i ("" = no value), or None when
    no idiom object is in reach. The object is looked for in the next
    IDIOM_WINDOW tokens of the clause (a comma, semicolon, colon or
    coordinating conjunction ends it: "brought a richly developed character
    to life"), or for "play a role" also just before the verb ("the role
    language plays"); "the role of" is acting, not an idiom."""
    for j in range(i + 1, min(end, i + 1 + IDIOM_WINDOW)):
        t = doc[j]
        if t.text in (",", ";", ":") or t.pos_ == "CCONJ":
            break
        hit = IDIOMS.get((base, t.text.lower()))
        if hit is not None:
            if j + 1 < end and doc[j + 1].text.lower() == "of" and base == "play":
                return None
            return hit
    for j in range(max(start, i - 3), i):
        w = doc[j].text.lower()
        if (base, w) in IDIOM_OBJECT_BEFORE:
            return IDIOMS[(base, w)]
    return None


def frame_position(doc, i: int, start: int, end: int) -> bool:
    """Is the verb at i in a framing position (positional frame verbs)?

    - a ", V-ing" sentence adjunct: "..., marking the first time";
    - the reader as subject: "you", "we" or "us" in the 4 tokens before;
    - a span of time as subject: "April 2 has witnessed", "the 1990s
      witnessed" (a number, month or century/decade/era/year noun right
      before the verb, auxiliaries and adverbs skipped);
    - copular remain / stand: "remains a significant focus", "stands as a
      testament", "remains notable";
    - "the story of X begins";
    - the present tense with an "a/an", "more than" or wh-clause object
      ("represents a fascinating intersection", "illustrates how"), and
      illustrate / demonstrate in the present tense with any object.
    """
    low = doc[i].text.lower()
    if getattr(doc[i], "tag_", "") in ("VBZ", "VBP"):
        base = resolve_base(doc[i].text, getattr(doc[i], "lemma_", ""))[0]
        if base in _PRESENT_TENSE_FRAME:
            return True
        if i + 1 < end and doc[i + 1].text.lower() in _FRAME_OBJECT:
            return True
    if i > start and doc[i - 1].text == "," and low.endswith("ing"):
        return True
    if any(doc[j].text.lower() in _READER for j in range(max(start, i - 4), i)):
        return True
    j = i - 1
    while j >= start and doc[j].pos_ in ("AUX", "ADV"):
        j -= 1
    if j >= start and (doc[j].pos_ == "NUM" or doc[j].text[:1].isdigit()
                       or doc[j].text.lower() in _TIME_NOUNS or doc[j].text.lower() in _MONTHS):
        return True
    if low.startswith(("remain", "stand", "stood")) and i + 1 < end:
        nxt = doc[i + 1]
        if nxt.text.lower() in ("a", "an", "the", "as", "one") or nxt.pos_ in ("ADJ", "ADV"):
            return True
    if low.startswith("begin") and any(doc[k].text.lower() == "story" for k in range(start, i)):
        return True
    return False


def extract_predicates(doc, emit_copular: Optional[bool] = None,
                       light: Optional[FrozenSet[str]] = None,
                       sentence_profile: Optional[Callable[[int], Dict[str, Any]]] = None,
                       lexicon: Optional[Lexicon] = None,
                       frame: Optional[Tuple[FrozenSet[str], FrozenSet[str]]] = None,
                       ) -> List[Predicate]:
    """Every predicate in ``doc`` with its canonical value, in token order.

    ``sentence_profile(si)`` returns the ontology-conditioned extraction
    profile for sentence ``si`` (affordance/profiles.py). Its
    ``function_verbs`` yield to the verb they govern ("used to treat" ->
    "treat", "designed to carry" -> "carry"): what the thing is for.
    ``emit_copular``, ``light`` and ``frame`` default to the config
    (emit_copular, light_verbs, frame_verbs / frame_verbs_positional).
    """
    if doc is None:
        return []
    lex = lexicon or _SEED
    if emit_copular is None:
        emit_copular = config_emit_copular()
    if light is None:
        light = config_light_verbs()
    frame_always, frame_pos = frame if frame is not None else config_frame_verbs()
    ents = _ent_token_set(doc)
    out: List[Predicate] = []
    for si, (start, end) in enumerate(_sentence_bounds(doc)):
        prof = sentence_profile(si) if sentence_profile else None
        fverbs = (prof or {}).get("function_verbs", DEFAULT_FUNCTION_VERBS)
        for i in range(start, end):
            tok = doc[i]
            text = tok.text
            low = text.lower()
            if emit_copular and tok.pos_ in ("AUX", "VERB") and low in BE_FORMS:
                v = _copular_value(doc, i, end)
                if v:
                    out.append(Predicate(v, i, "be", "", False, "copular"))
                continue
            if tok.pos_ != "VERB":
                continue
            # The tagger's VERB for an adjective after a be-form ("is red"):
            # not a verb, whether or not copular values are emitted.
            if _adjective_after_be(doc, i):
                continue
            # Proper-name tokens tagged VERB: "United" States, "Marvel".
            if i != start and text[:1].isupper():
                continue
            if i in ents and text[:1].isupper():
                continue
            # Participial modifier of a noun ("the abandoned station"): an
            # attribute, not a predicate.
            if getattr(tok, "dep_", "") == "amod":
                continue
            tag = getattr(tok, "tag_", "")
            prev = doc[i - 1] if i > start else None
            nxt_tok = doc[i + 1] if i + 1 < end else None
            obj_or_adp = _object_follows(doc, i + 1, end) or (
                nxt_tok is not None and (nxt_tok.pos_ == "ADP" or nxt_tok.text.lower() in ("that", "as")
                                         or getattr(nxt_tok, "tag_", "") == "VBG"))
            # Gerund used as a noun: "comic book writing", "a year of planning".
            # After a noun, a gerund with an object or a preposition (not
            # "of") is a reduced relative clause, a verb: "disorders
            # affecting voluntary muscles", "a book chronicling the history".
            if tag == "VBG" and prev is not None:
                if prev.pos_ in ("DET", "ADJ", "NOUN", "PROPN") or prev.text.lower() in ("of", "'s"):
                    if not (prev.pos_ in ("NOUN", "PROPN") and (
                            _object_follows(doc, i + 1, end) or (
                                nxt_tok is not None and nxt_tok.pos_ == "ADP"
                                and nxt_tok.text.lower() != "of"))):
                        continue
            # Base form right after a determiner/adjective is a noun ("a
            # construction permit"); right after a noun or a verb it is a noun
            # or an adverb the tagger missed ("recorded live") -- unless an
            # object, a preposition, "as", a gerund or a "that" clause
            # follows: "courts accept them", "helps improve critical
            # thinking", "farms dominate with", "the Centre ensure that",
            # "Society of America emphasize studying".
            # Only a bare base form qualifies (the surface is its own base,
            # before any synonym fold): "Metcalfe sang" is a past tense the
            # tagger left as VB.
            # A possessive makes it a noun whatever follows ("our grasp of").
            if tag in ("VB", "VBP") and prev is not None and (
                    prev.pos_ in ("NOUN", "PROPN", "ADJ", "DET") or prev.text.lower() in _POSSESSIVE_ONLY or
                    (prev.pos_ == "VERB" and prev.text.lower() != "to")) \
                    and resolve_base(text, getattr(tok, "lemma_", ""), lex)[0] == low:
                plural_subject = prev.pos_ in ("NOUN", "PROPN") and _plural_noun(prev.text)
                if prev.pos_ in ("DET", "ADJ") or prev.text.lower() in _POSSESSIVE_ONLY \
                        or not (obj_or_adp or plural_subject or (
                            nxt_tok is not None and nxt_tok.text.lower() in _WH)):
                    continue
            # Participle between a modifier and its noun is an adjective
            # ("heavy armoured car", "a tired WPTA news operation").
            if tag == "VBN" and nxt_tok is not None and nxt_tok.pos_ in ("NOUN", "PROPN") \
                    and prev is not None and prev.pos_ in ("ADJ", "DET", "ADV"):
                continue
            # Participle inside a noun phrase ("a landlocked country").
            if _attributive_participle(doc, i, start, end, tag):
                continue
            # Gerund in "along with shipping".
            if tag == "VBG" and prev is not None and prev.text.lower() == "with" \
                    and i - 2 >= start and doc[i - 2].text.lower() == "along":
                continue
            base, known = resolve_base(text, getattr(tok, "lemma_", ""), lex)
            # "found" after do-support is the base form of found, not the
            # past of find ("What did they found?").
            if low == "found" and any(doc[k].text.lower() in ("do", "does", "did")
                                      for k in range(max(start, i - 6), i)):
                base = "found"
            if not base.isalpha() or len(base) < 2:
                continue
            nxt = _particle(doc, i, end)
            # Aspectual verb governing another verb: yield to that verb.
            if base in ASPECTUAL:
                j = i + 1
                to = j < end and doc[j].text.lower() == "to"
                if to:
                    j += 1
                if j < end and (doc[j].pos_ == "VERB" or (
                        (to or base == "help") and _bare_verb(doc[j]))):
                    continue
            # ... also across a short object, to an infinitive: "helps us
            # appreciate", "aimed to provide", "led the military to
            # intervene" ("enables the partitioning of" governs no verb).
            if base in ASPECTUAL_ACROSS_OBJECT:
                j, k = i + 1, 0
                while j < end and k < 4 and doc[j].pos_ in ("PRON", "DET", "ADJ", "NOUN", "PROPN"):
                    j += 1
                    k += 1
                to = j < end and doc[j].text.lower() == "to"
                if to:
                    j += 1
                if j < end and doc[j].pos_ == "VERB" and (to or getattr(doc[j], "tag_", "") == "VB") \
                        and (base != "lead" or k > 0):
                    continue
            # Function verb governing another verb ("used to treat",
            # "used for cutting"): the governed verb is the function.
            if base in fverbs:
                j = i + 1
                if j + 1 < end and doc[j].text.lower() in ("to", "for", "in") \
                        and doc[j + 1].pos_ == "VERB":
                    continue
            lex.note_verb(low, tag, base, known, nxt)
            morph = getattr(tok, "morph", None)
            passive = bool(morph is not None and hasattr(morph, "get")
                           and morph.get("Voice") == ["Pass"])
            # Phrasal value before folding: the particle reads the verb as
            # written ("pick up" is not "choose up").
            ph = lex.phrasal(base, nxt)
            if ph:
                out.append(Predicate(ph, i, base, nxt, passive, "phrasal"))
                continue
            # Verb + object idiom: its own value or none ("shed light").
            idiom = _idiom_value(doc, i, start, end, base)
            if idiom is not None:
                if idiom:
                    out.append(Predicate(idiom, i, base, "", passive, "idiom"))
                continue
            if base in light or low in light:
                continue
            value = lex.synonym(base)
            if value in light:
                continue
            if base in frame_always or value in frame_always:
                continue
            if (base in frame_pos or value in frame_pos) and frame_position(doc, i, start, end):
                continue
            out.append(Predicate(value, i, base, "", passive, "verb"))
    return out
