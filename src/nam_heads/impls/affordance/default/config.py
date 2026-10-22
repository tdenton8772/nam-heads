# Affordance Encoder Configuration
import os as _os

DEFAULT_CONFIG = {
    # Emission: "predicate" (one address per predicate, with token positions,
    # linked per entity by the bundler) or "selected" (legacy primary +
    # secondary per record). See predicates.py.
    "emit_mode": _os.getenv("NAM_AFFORDANCE_EMIT", "predicate"),
    # Copular "X is a N" -> classify, "X is ADJ" -> describe. Off: the
    # values say THAT a passage describes something, not what the entity
    # does (hand audit 2026-10-07, wiki sample: 25 of 153 junk values).
    "emit_copular": _os.getenv("NAM_AFFORDANCE_COPULAR", "0") in ("1", "true", "yes"),

    # Predicate lexicon in the affordance store (predicate_store.py).
    "learn_predicates": _os.getenv("NAM_AFF_LEARN", "1") not in ("0", "false", "no"),
    "use_learned_predicates": _os.getenv("NAM_AFF_USE_LEARNED", "1") not in ("0", "false", "no"),

    # Predicate mode: at most this many predicate values per record (highest
    # score first), bounding head addresses and metadata per record.
    "max_predicate_values": int(_os.getenv("NAM_AFF_MAX_PREDICATES", "24")),

    # Fanout (selected mode only)
    "max_affordances": 2,   # hard cap 2 even if increased

    # Scoring weights (0-1)
    "weights": {
        "root": 1.0,
        "conj_root": 0.9,
        "relcl": 0.90,
        "acl": 0.80,
        "xcomp": 0.85,
        "ccomp": 0.85,
        "advcl": 0.50,
        "pcomp": 0.55,
        "main": 0.75,
        "other": 0.20,
        "regex": 0.40,
        "subj_bonus": 0.10,
        "dobj_bonus": 0.10,
        "purpose_bonus": 0.10,
        "passive_penalty": 0.20,
        "event_penalty": 1.00,  # firewall: push event/result/causal verbs below thresholds
        "light_penalty": 0.50,
        "copular": 0.60,
    },

    # Selection thresholds
    "absolute_min_confidence": 0.25,
    "soft_min_confidence": 0.08,
    "elbow_gap": 0.18,
    "cluster_tolerance": 0.07,

    # Firewalls: verbs that report THAT something happened, resulted or was
    # caused, not what an entity did. Filtered unless the ontology exempts
    # them (definitions.FIREWALL_ONTOLOGY_EXEMPTIONS). Entries are lemmas or
    # predicate-mode phrasal values. The incident vocabulary (outage, crash,
    # offline, ...) stays for incident text; the encyclopedic frame verbs
    # (take place, ensue, result in, give rise to, lead to) are added.
    # "lead"/"led" are NOT causal: in encyclopedic text they mean leading a
    # body ("led the army"); the causal reading is the phrasal "lead_to".
    "event_verbs": [
        "fail", "failed", "crash", "crashed", "occur", "occurred", "happen", "happened",
        "trigger", "triggered", "outage", "incident",
        "take_place", "ensue", "transpire", "befall", "unfold",
    ],
    "result_verbs": [
        "offline", "degraded", "unavailable", "resolved", "healthy", "down", "up",
        "result_in", "result_from",
    ],
    "causal_verbs": [
        "cause", "caused", "result", "resulted", "because", "therefore",
        "lead_to", "give_rise",
    ],

    # Discourse frame: verbs that describe the PASSAGE -- what it reflects,
    # highlights or reminds the reader of -- not what an entity did
    # ("This album highlights a cultural moment", "If you're wondering
    # who ..."). Applied by the extractor (predicates.py) to the base and the
    # folded value; no ontology exemption.
    "frame_verbs": [
        "reflect", "embody", "exemplify", "highlight", "underscore", "underline",
        "symbolize", "showcase", "remind", "wonder", "imagine", "appreciate",
        "epitomize", "encapsulate", "testify", "signify",
    ],
    # Frame only in a framing position (predicates.frame_position): a
    # ", V-ing" sentence adjunct ("..., marking the first time"), a you/we/us
    # subject, a time-span subject ("April 2 has witnessed"), copular
    # remain/stand ("remains a significant focus"), "the story of X begins".
    # Elsewhere they are predicates ("represented Ohio", "witnessed the
    # signing", "marked the border").
    "frame_verbs_positional": [
        "illustrate", "represent", "mark", "witness", "demonstrate", "explore",
        "begin", "stress", "remain", "stand", "recognize", "hint", "suggest",
    ],
    "light_verbs": [
        "be", "is", "was", "were", "am", "are",
        "have", "has", "had",
        "do", "does", "did",
        "tell", "say", "said", "know", "knew", "known",
        "think", "thought", "want", "wanted",
        "need", "needed", "get", "got", "gotten",
        "give", "gave", "given", "make", "made",
        "take", "took", "taken", "let",
        "show", "showed", "shown",
        # "found" is not listed: the rule lemmatizer gives "founded" the
        # lemma "found"; the past of find already lemmatizes to "find".
        "find", "see", "saw", "seen",
        "go", "went", "gone", "come", "came",
        "look", "looked", "seem", "seemed",
        "mean", "meant", "keep", "kept",
        "put", "set", "try", "tried",
        "call", "called", "ask", "asked",
    ],

    # Category store
    "category_store_backend": _os.getenv("NAM_AFFORDANCE_STORE_BACKEND", "couchbase_kv"),
    "category_store_fallback": "memory",
    "category_store_path": "./data/affordance_store/verb_categories.db",
    "category_store_seed": True,

    # Ontology-aware scoring
    "ontology_aware": True,
    "ontology_relevance_weight": 1.0,

    # NER scoping
    "ner_scoping_weight": 0.5,
}

# The parse is the rule NLP backend's (the nlp.rule plugin, nam_heads/impls/nlp/rule): in-repo,
# no package or model to install.
DEPENDENCIES = {
    "python": [],
    "system": [],
    "models": [],
}
