"""What the attribute store learns from predications, and when it trusts it.

Every word the predication path keeps (before the ontology steer) is an
observation of a value. Observations are merged into the value's global
store row:

    kinds        structural evidence kinds it was seen as (norp, noun, adj,
                 noun_mod; a role noun is recorded as ``noun``)
    ontologies   the ontology labels of the sentences it was read from
    classes      the same as BFO/CCO class CURIEs (registry bindings)
    person_support  up to ROLE_SUPPORT record fingerprints in which the
                 value was the head noun of a copular/relative/as-role/
                 pronoun predication, in a sentence whose ontology is a
                 cco:Person, whose subject contains a PERSON-labelled token,
                 with a surface person cue (he/she, who, lifespan)
    role         True when it names a kind of person: from the seed
                 (person_nouns.txt, role_source=seed) or learned once
                 person_support reaches ROLE_SUPPORT (role_source=learned)
    kind_support up to KIND_SUPPORT record fingerprints in which the value
                 was the head noun of a copular/relative/appositive
                 predication of a NAMED subject ("Kucura is a village",
                 "Acapulco, a seaport city"), not followed by of/to/for (the
                 relational "X is the capital of Y" is not evidence)
    kind         whether the noun names a kind of thing a subject can be
                 an instance of. False from the seed (abstract_nouns.txt,
                 kind_source=seed: discourse, abstract and relational nouns
                 -- "story", "impact", "heart", "capital"); True once
                 kind_support reaches KIND_SUPPORT (kind_source=learned),
                 which overrides the seed: the corpus predicating the noun
                 of named subjects is evidence that it is used as a kind.
                 kind_status() reads it; a role noun is always a kind.
                 Extraction asks it before resolving "this/the <noun>" to
                 the main subject and before reading "the <noun> of X" /
                 "<nouns> such as X".

Thresholds: ROLE_SUPPORT = KIND_SUPPORT = 3 distinct records. The support
lists keep the smallest fingerprints, so they depend only on WHICH records
gave evidence -- not on their order, batching or repetition.

Convergence: every field is a set union or a bounded min-set, so merging is
commutative, associative and idempotent. Observations are taken before the
ontology steer and kinds are structural, so the observation set only grows
as the learned role set grows. Hence any two runs over the same records
reach the same store once each has made two passes (the second pass sees
the final role set). A row's category: a seeded/curated row (confidence
1.0) keeps its own; otherwise ``role`` if it is a role noun, else the best
category of its first observed ontology label (sorted) -- a function of
the merged row, not of arrival order.
"""

import hashlib
from typing import Dict, Iterable, Optional, Set

from nam_heads.ontology import bindings as B

ROLE_SUPPORT = 3
SUPPORT_LINKS = frozenset({"copular", "relative", "as_role", "pronoun"})
KIND_SUPPORT = 3
KIND_LINKS = frozenset({"copular", "relative", "appositive"})
# A predicate noun taking one of these complements relates its subject to
# something ("the capital of", "a testament to", "the basis for"): not kind evidence.
RELATIONAL_PREPS = frozenset({"of", "to", "for"})
LEARNED_CONFIDENCE = 0.5


def fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def is_role(doc: Optional[dict]) -> bool:
    return bool(doc) and doc.get("role") is True


def kind_status(doc: Optional[dict]) -> Optional[bool]:
    """True: the noun names a kind (learned, or a role noun); False: an
    abstract / relational noun (seed, not yet overridden by evidence);
    None: nothing known."""
    if not doc:
        return None
    k = doc.get("kind")
    if k is True or doc.get("role") is True:
        return True
    if k is False:
        return False
    return None


def structural_kind(kind: str) -> str:
    return "noun" if kind == "role_noun" else kind


class Observation:
    __slots__ = ("kinds", "ontologies", "classes", "support", "kind_support")

    def __init__(self):
        self.kinds: Set[str] = set()
        self.ontologies: Set[str] = set()
        self.classes: Set[str] = set()
        self.support: Set[str] = set()
        self.kind_support: Set[str] = set()


def _category_for(doc: dict) -> str:
    from .definitions import get_best_category_for_ontology
    if doc.get("role"):
        return "role"
    for label in doc.get("ontologies", []):
        cat = get_best_category_for_ontology(label)
        if cat:
            return cat
    return "uncategorized"


def merge(old: Optional[dict], obs: Observation) -> dict:
    """The row after merging one observation set (pure, deterministic)."""
    new = dict(old) if old else {}
    new["kinds"] = sorted(set(new.get("kinds", [])) | obs.kinds)
    new["ontologies"] = sorted(set(new.get("ontologies", [])) | obs.ontologies)
    new["classes"] = sorted(set(new.get("classes", [])) | obs.classes)
    support = sorted(set(new.get("person_support", [])) | obs.support)[:ROLE_SUPPORT]
    if support:
        new["person_support"] = support
    if new.get("role_source") != "seed" and len(support) >= ROLE_SUPPORT:
        new["role"], new["role_source"] = True, "learned"
    ksupport = sorted(set(new.get("kind_support", [])) | obs.kind_support)[:KIND_SUPPORT]
    if ksupport:
        new["kind_support"] = ksupport
    if len(ksupport) >= KIND_SUPPORT and new.get("kind") is not True:
        new["kind"], new["kind_source"] = True, "learned"
    curated = old is not None and float(old.get("confidence", 0)) >= 1.0
    if not curated:
        new["category"] = _category_for(new)
        new["confidence"] = LEARNED_CONFIDENCE
    return new


def person_evidence(link: str, word_role: str, kind: str, sentence_class: str,
                    subject_has_person_ner: bool, person_cue: bool = False,
                    followed_by_of: bool = False) -> bool:
    """Does this observation count toward learning that a noun names a person?

    The ontology and NER labels alone are too noisy (the ontology stage
    labels ~76% of rows person; NER labels album and town names PERSON):
    measured on 2,000 records they promoted album, band, village, town.
    So a surface person cue is also required (predication.person_cue).
    """
    if link not in SUPPORT_LINKS or word_role != "head" or structural_kind(kind) != "noun":
        return False
    # "the head of X", "the voice of X": a relational noun names the
    # subject's relation to X, and is not evidence that it names a person
    if followed_by_of:
        return False
    if not (subject_has_person_ner and person_cue):
        return False
    from nam_heads.ontology.registry import get_registry
    reg = get_registry()
    return reg.is_class(sentence_class) and reg.is_a(sentence_class, B.PERSON)


def kind_evidence(link: str, word_role: str, kind: str, named_subject: bool,
                  of_complement: bool = False, relational: bool = False) -> bool:
    """Does this observation count toward learning that a noun names a kind?

    The head noun of a copular / relative / appositive predication of a
    named subject ("Kucura is a village"), unless the predicate takes an
    of-complement or the noun a relational complement (RELATIONAL_PREPS:
    "the capital of Y", "a testament to X"): relational nouns are what
    kind_of / anaphor resolution must not accept.
    """
    return (link in KIND_LINKS and word_role == "head" and structural_kind(kind) == "noun"
            and named_subject and not of_complement and not relational)
