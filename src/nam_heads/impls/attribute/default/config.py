# Attribute Encoder Configuration
import os as _os

DEFAULT_CONFIG = {
    # Fanout and filtering
    "max_attributes": 3,
    "min_len": 2,

    # Optional attribute wordlist (list or path to txt/json)
    "wordlist": None,
    # Canonicalization helpers -- lemma-to-lemma folding
    "synonyms": {
        # size
        "big": "large",
        "huge": "large",
        "enormous": "large",
        "massive": "large",
        "gigantic": "large",
        "immense": "large",
        "tiny": "small",
        "miniature": "small",
        "little": "small",
        "minute": "small",
        # quality
        "great": "excellent",
        "superb": "excellent",
        "wonderful": "excellent",
        "outstanding": "excellent",
        "brilliant": "excellent",
        "terrible": "poor",
        "awful": "poor",
        "dreadful": "poor",
        "horrible": "poor",
        # speed
        "quick": "fast",
        "rapid": "fast",
        "swift": "fast",
        "speedy": "fast",
        # state
        "damaged": "broken",
        "shattered": "broken",
        "ruined": "broken",
        # emotion
        "joyful": "happy",
        "cheerful": "happy",
        "glad": "happy",
        "furious": "angry",
        "enraged": "angry",
        "irate": "angry",
        # other
        "beautiful": "attractive",
        "gorgeous": "attractive",
        "ugly": "unattractive",
        "hideous": "unattractive",
        "wealthy": "rich",
        "affluent": "rich",
    },
    "stopwords": [
        "good", "bad", "new", "old", "other", "different", "same",
        "many", "few", "various", "several",
        "certain", "particular", "own", "such", "whole", "entire",
        "main", "only", "very", "real", "actual",
        "likely", "possible", "available", "specific", "general",
    ],

    # Category store -- maps lemmas to semantic categories
    "category_store_backend": _os.getenv("NAM_ATTRIBUTE_STORE_BACKEND", "couchbase_kv"),
    "category_store_fallback": "memory",
    "category_store_path": "./data/attribute_store/categories.db",
    "category_store_seed": True,              # bootstrap from definitions.py on first run

    # Ontology-aware scoring
    "ontology_aware": True,
    "ontology_relevance_weight": 1.0,  # 0.0=off, 1.0=full multiplier delta

    # NER scoping
    "ner_scoping_weight": 0.5,  # 0.0=off, 1.0=full NER multiplier

    # Method weights (0.0-1.0) for scoring
    "weights": {
        "wordlist": 1.00,
        "amod": 0.90,
        "predicate": 0.90,
        "participial": 0.85,
        "compound_mod": 0.85,
        "adj": 0.70,
        "regex": 0.50,
        "stop_penalty": 0.30,  # subtracted if stopword hit
        "compound_phrase": 0.88,  # multi-token adjective phrases
        "noun_compound": 0.75,   # NOUN modifiers in compound dep
        "appositive": 0.80,     # appositive NOUN/PROPN modifiers
        "acl_participial": 0.82, # past participle in acl dep
        # predication-mode links (heads/attribute/predication.py)
        "copular": 0.90,         # X is a/an ADJ* NOUN
        "query": 0.90,           # a canonicalized query's leading NP
        "title": 0.85,           # the physicist Richard Feynman
        "as_role": 0.85,         # served as Minister
        "relative": 0.80,        # X, who was a ...
        "pronoun": 0.70,         # He was a ... (resolved to the main subject)
        "modifier": 0.60,        # ADJ before a noun
    },
    # Adaptive selection thresholds
    "absolute_min_confidence": 0.20,   # drop below this (unless best is weak fallback)
    "soft_min_confidence": 0.05,       # keep best one if all are weak but non-garbage
    "elbow_gap": 0.15,                 # cut at biggest gap > elbow_gap
    "cluster_tolerance": 0.05,         # keep scores within tolerance of top
}

# The parse is the rule NLP backend's (the nlp.rule plugin, nam_heads/impls/nlp/rule): in-repo,
# no package or model to install.
DEPENDENCIES = {
    "python": [],
    "system": [],
    "models": [],
}
