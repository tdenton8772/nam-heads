"""test_context_head.py tests the legacy NER-driven extraction, which stays
selectable (NAM_CONTEXT_HEAD=legacy) for rollback; the default head is the
particular-evidence extractor (src/context_axis, tests in
context_axis/tests and test_head_selection.py). Pin that module to legacy."""
import os

import pytest

_LEGACY_MODULES = ("test_context_head",)


@pytest.fixture(autouse=True, scope="module")
def _legacy_context_head(request):
    if request.module.__name__.rsplit(".", 1)[-1] not in _LEGACY_MODULES:
        yield
        return
    old = os.environ.get("NAM_CONTEXT_HEAD")
    os.environ["NAM_CONTEXT_HEAD"] = "legacy"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("NAM_CONTEXT_HEAD", None)
        else:
            os.environ["NAM_CONTEXT_HEAD"] = old
