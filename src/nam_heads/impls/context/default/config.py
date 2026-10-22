# Context encoder configuration (NER-driven extraction + store-backed normalization)
import os as _os

DEFAULT_CONFIG = {
    "default_null": "__null__",

    # Dimensions to extract
    "dimensions": ["temporal", "geographic", "domain", "institutional"],
    "max_per_dimension": 1,

    # Context store
    "context_store_backend": _os.getenv("NAM_CONTEXT_STORE_BACKEND", "couchbase_kv"),
    "context_store_fallback": "memory",
    "context_store_path": "./data/context_store/contexts.db",
    "context_store_seed": True,

    # Fragment splitting
    "split_fragments": True,
}

# The parse (NER, sentences) is the rule NLP backend's
# (the nlp.rule plugin, nam_heads/impls/nlp/rule): in-repo, no package or model to install.
DEPENDENCIES = {
    "python": [],
    "system": [],
    "models": [],
}
