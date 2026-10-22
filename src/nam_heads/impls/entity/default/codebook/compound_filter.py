# Post-codebook filter for compound noun meaningfulness.
#
# After the main codebook classifies a candidate as compound_noun/compound/
# quality/long_description, this filter decides whether the compound
# represents a meaningful concept (keep) or contextual noise (reject).
#
# Meaningful: "quantum_physics", "civil_war", "human_rights"
# Noise:      "perpetuate_authority", "all_institutions", "important_role"
#
# Deterministic, no ML — pure rule-based on POS pattern + word lists.

from typing import Optional

# Types that trigger the compound filter.  Everything else passes through.
_COMPOUND_TYPES = frozenset({
    "compound_noun",
    "compound",
    "quality",
    "long_description",
})

# Rejection types from codebook — always meaningless, never index.
_REJECTION_TYPES = frozenset({
    "rejection_generic_noun",
    "rejection_low_salience",
})

# Generic modifiers: if the FIRST component is one of these, the compound
# is almost certainly a contextual description, not a named concept.
_GENERIC_MODIFIERS = frozenset({
    "all", "any", "both", "certain", "each", "entire", "every",
    "few", "first", "former", "further", "general", "good", "great",
    "high", "important", "large", "last", "latter", "long", "low",
    "main", "major", "many", "more", "most", "much", "new", "next",
    "old", "only", "other", "own", "particular", "possible",
    "previous", "real", "recent", "same", "several", "significant",
    "similar", "single", "small", "special", "such", "total",
    "various", "whole", "wide",
})

# Generic root words: if the LAST component is one of these, the compound
# is too vague to be an entity.  "important_role" → role is generic.
_GENERIC_ROOTS = frozenset({
    "area", "aspect", "case", "claim", "day",
    "effect", "element", "end", "example", "fact", "feature",
    "form", "group", "hand", "idea", "impact", "increase",
    "interest", "issue", "kind", "level", "line", "matter",
    "means", "measure", "method", "model", "movement",
    "name", "nature", "need", "number", "order", "part",
    "period", "place", "point", "position", "power",
    "process", "question", "range", "rate", "reason",
    "relationship", "replacement", "report", "result",
    "role", "set", "side", "situation", "sort", "source",
    "stage", "state", "step", "story", "structure",
    "study", "subject", "term",
    "thing", "time", "type", "use", "value", "view",
    "way", "word", "work", "year",
})

# Verb stems that appear as the first component when rule-NLP
# misclassifies a verb as NOUN inside a noun chunk.
_VERB_STEMS = frozenset({
    "abolish", "achieve", "advocate", "allow", "apply",
    "attempt", "avoid", "base", "become", "begin", "believe",
    "build", "call", "carry", "cause", "claim", "close",
    "consider", "contain", "continue", "control", "cover",
    "create", "deal", "define", "demonstrate", "describe",
    "develop", "establish", "expect", "explain", "express",
    "extend", "face", "fail", "find", "follow", "force",
    "give", "grow", "happen", "help", "hold", "include",
    "indicate", "involve", "keep", "know", "lead", "leave",
    "lose", "maintain", "make", "manage", "mark", "mean",
    "meet", "note", "obtain", "offer", "operate", "oppose",
    "pass", "pay", "perform", "perpetuate", "play", "point",
    "produce", "promote", "propose", "protect", "provide",
    "publish", "raise", "reach", "receive", "reduce",
    "reflect", "remain", "remove", "replace", "represent",
    "require", "result", "reveal", "run", "seek", "serve",
    "show", "spread", "stand", "start", "stop", "suggest",
    "support", "take", "tell", "tend", "think", "try",
    "turn", "understand", "use", "want", "win", "write",
})


def is_meaningful_compound(
    normalized: str,
    pos_pattern: str,
    text: str,
    token_count: int,
    is_propn: bool,
    entity_type: Optional[str] = None,
    is_name: bool = False,
    has_article: bool = False,
    concept: Optional[bool] = None,
) -> bool:
    """Return True if this compound noun is a meaningful concept worth indexing.

    Called AFTER main codebook classification, only for compound types.
    If entity_type is not in _COMPOUND_TYPES, always returns True (pass-through).

    ``is_name``: the mention is a name as written (EntityCandidate.is_name:
    a capitalised span such as "European Union", "English Civil War",
    "Royal Navy"). A name is always meaningful -- the article rule and the
    three-word rule are for lower-case descriptions.

    ``has_article``: the mention came with an article or demonstrative, which
    is not in ``pos_pattern`` (EntityCandidate.has_article).

    ``concept``: the concept gate's verdict (entity/concepts.py) for a
    lower-case phrase of two or more words -- True: a seed or learned
    concept, which the word-list rules below do not second-guess ("high
    school", "gross domestic product"); False: neither, never meaningful;
    None: the gate does not cover the mention.
    """
    if is_name:
        return True
    if concept is False:
        return False

    # rejection_low_salience: always reject — these are truly low-signal.
    if entity_type == "rejection_low_salience":
        return False

    # rejection_generic_noun: codebook cluster is too greedy — real compounds
    # like "acid_rain" land here because their feature vectors resemble common
    # nouns.  Fall through to the rule-based checks below instead of blind
    # rejection.  The rules will still catch actual generic nouns.

    if entity_type and entity_type not in _COMPOUND_TYPES | _REJECTION_TYPES:
        return True

    # Proper nouns always pass — named things are meaningful.
    # Also check POS pattern: rule-NLP sets is_propn from root token only,
    # but "Edwards Air Force Base" has root "Base" (NOUN).  If ANY token
    # in the chunk is PROPN, treat it as a named entity.
    if is_propn:
        return True

    parts = normalized.split("_")
    if not parts:
        return False

    first = parts[0]
    last = parts[-1]
    pos_tags = set(pos_pattern.split("_")) if pos_pattern != "UNK" else set()

    if "PROPN" in pos_tags:
        return True

    # Rule 1: a lower-case compound introduced by an article ("the decisive
    # engagement", "a unique blend") or holding a determiner ("all
    # institutions") is a description. Names never reach this rule.
    if has_article or "DET" in pos_tags:
        return False

    if concept:
        return True

    # Rule 2: VERB in POS → "perpetuate authority" → action phrase, not entity.
    if "VERB" in pos_tags:
        return False

    # Rule 3: Generic modifier as first component.
    if first in _GENERIC_MODIFIERS:
        return False

    # Rule 4: Generic root as last component.
    if last in _GENERIC_ROOTS:
        return False

    # Rule 5: First component looks like a verb stem (rule-NLP misclassifies).
    if first in _VERB_STEMS:
        return False

    # Rule 6: 3+ content words → almost always a description, not a concept.
    # Words as written: a hyphenated word is one word ("anti-aircraft gun"
    # is two, its normalized form anti_aircraft_gun has three parts).
    words = text.split() if text else parts
    if len(words) >= 3:
        return False

    return True
