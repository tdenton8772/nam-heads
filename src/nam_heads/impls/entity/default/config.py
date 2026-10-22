import os as _os

DEFAULT_CONFIG = {
    # Storage -- override via NAM_ENTITY_STORE_BACKEND env var
    # Options: "memory" (dev), "lmdb" (ingest: the node cache, read-only),
    # "couchbase_kv" (nam-query: session bucket, node cache in front if set)
    "db_backend": _os.getenv("NAM_ENTITY_STORE_BACKEND", "memory"),
    "db_fallback": "memory",

    # Extraction
    "max_surface_forms": 50,          # hard cap per encode() call
    "min_surface_len": 3,

    # Resolution
    "resolution_threshold": 0.78,     # match existing entity if >= this
    "intra_record_cluster_threshold": 0.82,  # merge mentions in same record if >= this
    "min_token_overlap": 0.5,         # fast candidate filter

    # Output
    "max_entities_per_record": 25,    # hard cap
    "include_debug_metadata": False,  # keep metadata compact by default

    # Behavior
    "create_new_entities": True,      # if False, only emit matches to existing entities
    "min_entity_confidence": 0.20,

    # Fuzzy resolution (type-dispatched comparators)
    "enable_fuzzy_resolution": True,
    "fuzzy_resolution_max_candidates": 50,

    "enable_diagnostics": True,
}

DEPENDENCIES = {
    "python": [
        {
            "name": "rapidfuzz",
            "version": ">=3.0.0",
            "required": False,
            "note": "Jaro-Winkler similarity for fuzzy entity resolution (falls back to exact match)"
        },
        {
            "name": "metaphone",
            "version": ">=0.6",
            "required": False,
            "note": "Double Metaphone phonetic keys (falls back to Soundex)"
        },
    ],
    "system": [],
    # The parse is the rule NLP backend's (the nlp.rule plugin, nam_heads/impls/nlp/rule):
    # in-repo, no model to install.
    "models": []
}