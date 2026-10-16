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
