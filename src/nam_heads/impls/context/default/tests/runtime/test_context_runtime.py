"""
Runtime tests for the context head with its default extraction (the
particular-evidence extractor, src/context_axis): confidence bounds,
metadata shape, and null fallback. Texts are encoded without a doc, so the
head parses them (the rule parser).
"""

import pytest

from nam_heads.impls.context.default.head import ContextEncoder


def make_encoder():
    return ContextEncoder(
        head_type="context",
        config={
            "context_store_backend": "memory",
            "context_store_seed": True,
        },
    )


def test_null_when_no_match():
    enc = make_encoder()
    text = "No contextual signals here."
    res = enc.encode(text)
    assert all(a.concept == "__null__" for a in res.addresses)


def test_confidence_bounded():
    enc = make_encoder()
    res = enc.encode("Founded in 1935 in France.")
    for a in res.addresses:
        assert 0.0 <= a.confidence <= 1.0


def test_metadata_has_dimension():
    enc = make_encoder()
    res = enc.encode("Founded in 1935.")
    for a in res.addresses:
        assert "dimension" in a.metadata
        assert a.metadata["dimension"] in (
            "temporal", "geographic", "domain", "institutional"
        )


def test_mode_in_metadata():
    enc = make_encoder()
    res = enc.encode("Founded in 1935.")
    assert res.metadata["mode"] == "ingest"

    res_q = enc.encode_query("When was it founded?")
    assert res_q.metadata["mode"] == "query"


def test_address_count_in_metadata():
    enc = make_encoder()
    res = enc.encode("Founded in 1935 in Germany.")
    assert res.metadata["address_count"] == len(res.addresses)
