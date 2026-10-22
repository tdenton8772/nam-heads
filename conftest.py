"""Run inside NAM when one is checked out next to this repo.

``NAM_SRC=/path/to/nam-0.0.1/nam pytest`` registers NAM's parser, axis
encoders and CPU budget as the host (``src.encoders.nam_host.register``), so
the tests that need a real parse run too. Without it they skip.
"""

import os
import sys

_nam = os.environ.get("NAM_SRC")
if _nam:
    sys.path.insert(0, _nam)
    from src.encoders import nam_host

    nam_host.register()


# The default heads (nlp.rule, entity, attribute, affordance, context) are written against NAM's host modules
# (src.storage, src.nlp, src.encoders.record_parse, ...): without NAM_SRC their packages cannot be imported,
# so their tests are not collected.
try:
    import src.nlp.interfaces  # noqa: F401
    _HAVE_NAM = True
except ImportError:
    _HAVE_NAM = False

collect_ignore = [] if _HAVE_NAM else [
    "src/nam_heads/impls/nlp", "src/nam_heads/impls/entity", "src/nam_heads/impls/attribute",
    "src/nam_heads/impls/affordance", "src/nam_heads/impls/context", "src/nam_heads/impls/_testing",
    "tests/test_default_head_metrics.py", "tests/test_default_store_factories.py",
]
